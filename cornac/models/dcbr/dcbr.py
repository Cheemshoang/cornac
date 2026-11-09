# Copyright 2026 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""PyTorch module of DCBR (imported only inside ``fit``)."""

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bundle_utils.graph import bi_norm, bipartite, row_norm
from ..bundle_utils.torch_utils import (
    bpr_loss, info_nce, lightgcn_propagate, perturb, sp_to_torch,
)


class Denoiser(nn.Module):
    """MLP that predicts a clean user-bundle row x_0 from a noised row x_t and step t."""

    def __init__(self, num_bundles, hidden_dims, time_emb_size, dropout):
        super().__init__()
        self.time_emb_size = time_emb_size
        self.time_proj = nn.Linear(time_emb_size, time_emb_size)
        dims = [num_bundles + time_emb_size] + list(hidden_dims)
        self.encoder = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        out_dims = list(reversed(hidden_dims)) + [num_bundles]
        self.decoder = nn.ModuleList(nn.Linear(a, b) for a, b in zip(out_dims[:-1], out_dims[1:]))
        self.dropout = nn.Dropout(dropout)

    def time_embedding(self, t):
        half = self.time_emb_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.time_emb_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.time_proj(emb)

    def forward(self, x, t):
        h = torch.cat([self.dropout(F.normalize(x, dim=-1)), self.time_embedding(t)], dim=-1)
        for layer in self.encoder:
            h = torch.tanh(layer(h))
        for k, layer in enumerate(self.decoder):
            h = layer(h)
            if k < len(self.decoder) - 1:
                h = torch.tanh(h)
        return h


class Diffusion(nn.Module):
    """Gaussian diffusion over user-bundle rows (x_0 prediction), DiffRec style."""

    def __init__(self, num_bundles, steps, noise_scale, noise_min, noise_max, hidden_dims,
                 time_emb_size, dropout):
        super().__init__()
        self.steps = steps
        self.denoiser = Denoiser(num_bundles, hidden_dims, time_emb_size, dropout)
        variance = noise_scale * np.linspace(noise_min, noise_max, steps, dtype=np.float64)
        alpha_bar = 1.0 - variance
        betas = 1.0 - alpha_bar / np.append(1.0, alpha_bar[:-1])
        betas = np.clip(betas, 1e-12, 0.999)
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)
        alpha_bar_prev = np.append(1.0, alpha_bar[:-1])
        buffers = {
            "sqrt_alpha_bar": np.sqrt(alpha_bar),
            "sqrt_one_minus_alpha_bar": np.sqrt(1.0 - alpha_bar),
            "post_coef_x0": betas * np.sqrt(alpha_bar_prev) / (1.0 - alpha_bar),
            "post_coef_xt": (1.0 - alpha_bar_prev) * np.sqrt(alphas) / (1.0 - alpha_bar),
        }
        for name, value in buffers.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def q_sample(self, x0, t, noise):
        return (self.sqrt_alpha_bar[t][:, None] * x0
                + self.sqrt_one_minus_alpha_bar[t][:, None] * noise)

    def training_loss(self, x0):
        t = torch.randint(0, self.steps, (x0.shape[0],), device=x0.device)
        x0_hat = self.denoiser(self.q_sample(x0, t, torch.randn_like(x0)), t)
        return ((x0_hat - x0) ** 2).mean(dim=1).mean(), x0_hat

    @torch.no_grad()
    def denoise(self, x0):
        """Reverse process from the clean row (no added noise), posterior means."""
        x = x0
        for step in reversed(range(self.steps)):
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            x0_hat = self.denoiser(x, t)
            x = self.post_coef_x0[step] * x0_hat + self.post_coef_xt[step] * x if step else x0_hat
        return x


class DCBRModule(nn.Module):
    """UI and BI views (scored) plus a UB view learned on a diffusion-rebuilt
    user-bundle graph, tied together by inter- and intra-view InfoNCE."""

    def __init__(self, ui, ub, bi, emb_size, num_layers, layer_weights, view_eps, omega,
                 c_lambda, c_temp, gammas, lambda_reg, diff_steps, noise_scale, noise_min,
                 noise_max, diff_dims, time_emb_size, diff_dropout, diff_lr, diff_batch_size,
                 blcc_weight, rebuild_k, rng, device):
        super().__init__()
        self.ub = ub.tocsr()
        self.num_users, self.num_items = ui.shape
        self.num_bundles = ub.shape[1]
        self.num_layers, self.layer_weights, self.view_eps = num_layers, layer_weights, view_eps
        self.omega, self.c_lambda, self.c_temp = omega, c_lambda, c_temp
        self.gammas, self.lambda_reg = gammas, lambda_reg
        self.diff_batch_size, self.blcc_weight = diff_batch_size, blcc_weight
        self.rebuild_k, self.rng, self.device = rebuild_k, rng, device

        self.users = nn.Parameter(torch.empty(self.num_users, emb_size))
        self.bundles = nn.Parameter(torch.empty(self.num_bundles, emb_size))
        self.items = nn.Parameter(torch.empty(self.num_items, emb_size))
        for weight in (self.users, self.bundles, self.items):
            nn.init.xavier_normal_(weight)

        self.ui_graph = sp_to_torch(bi_norm(bipartite(ui)), device)
        self.bi_graph = sp_to_torch(bi_norm(bipartite(bi)), device)
        self.bundle_agg = sp_to_torch(row_norm(bi), device)
        self.user_agg = sp_to_torch(row_norm(ui), device)
        self.diffusion = Diffusion(self.num_bundles, diff_steps, noise_scale, noise_min,
                                   noise_max, diff_dims, time_emb_size, diff_dropout)
        self.diff_optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=diff_lr)
        self.ub_graph = self._ub_graph(self.ub)

    def embedding_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith("diffusion.")]

    def _ub_graph(self, ub):
        eye = sp.identity(sum(ub.shape), format="csr")
        return sp_to_torch(bi_norm(bipartite(ub) + eye), self.device)

    def _rows(self, users):
        return torch.as_tensor(self.ub[users].toarray(), dtype=torch.float32,
                               device=self.device)

    def epoch_start(self, epoch):
        """Train the diffusion model one pass, then rebuild the UB view's graph."""
        with torch.no_grad():
            frozen_u, frozen_b = (x.detach() for x in self.fused(noise=False))
        self.diffusion.train()
        order = self.rng.permutation(self.num_users)
        for start in range(0, self.num_users, self.diff_batch_size):
            users = order[start: start + self.diff_batch_size]
            x0 = self._rows(users)
            loss, x0_hat = self.diffusion.training_loss(x0)
            if self.blcc_weight:
                # Bundle-level consistency: users aggregated through predicted rows
                # should match the (frozen) bundle representations.
                agg = x0_hat.T @ frozen_u[torch.as_tensor(users, device=self.device)]
                loss = loss + self.blcc_weight * F.mse_loss(agg, frozen_b)
            self.diff_optimizer.zero_grad()
            loss.backward()
            self.diff_optimizer.step()
        self.diffusion.eval()
        rows, cols = [], []
        for start in range(0, self.num_users, self.diff_batch_size):
            users = np.arange(start, min(start + self.diff_batch_size, self.num_users))
            scores = self.diffusion.denoise(self._rows(users))
            top = torch.topk(scores, min(self.rebuild_k, self.num_bundles), dim=1).indices
            rows.append(np.repeat(users, top.shape[1]))
            cols.append(top.cpu().numpy().ravel())
        rebuilt = sp.csr_matrix((np.ones(sum(len(r) for r in rows)),
                                 (np.concatenate(rows), np.concatenate(cols))),
                                shape=self.ub.shape)
        self.ub_graph = self._ub_graph((rebuilt > 0).astype(np.float64))

    def _view(self, graph, left, right, name, noise):
        eps = self.view_eps[name] if noise else 0.0
        feats = lightgcn_propagate(graph, torch.cat([left, right]), self.num_layers,
                                   layer_weights=self.layer_weights[name], normalize=True,
                                   noise_eps=eps)
        return feats[: left.shape[0]], feats[left.shape[0]:], eps

    def views(self, noise):
        u_ub, b_ub, _ = self._view(self.ub_graph, self.users, self.bundles, "UB", noise)
        u_ui, i_ui, eps = self._view(self.ui_graph, self.users, self.items, "UI", noise)
        b_ui = perturb(torch.sparse.mm(self.bundle_agg, i_ui), eps)
        b_bi, i_bi, eps = self._view(self.bi_graph, self.bundles, self.items, "BI", noise)
        u_bi = perturb(torch.sparse.mm(self.user_agg, i_bi), eps)
        return {"UB": (u_ub, b_ub), "UI": (u_ui, b_ui), "BI": (u_bi, b_bi)}

    def fused(self, noise, views=None):
        views = self.views(noise) if views is None else views
        (u_ui, b_ui), (u_bi, b_bi) = views["UI"], views["BI"]
        return (self.omega * u_ui + (1 - self.omega) * u_bi,
                self.omega * b_ui + (1 - self.omega) * b_bi)

    def loss(self, users, pos, neg):
        views = self.views(noise=True)
        u, b = self.fused(noise=True, views=views)
        bpr = bpr_loss((u[users] * b[pos]).sum(1), (u[users] * b[neg]).sum(1))
        reg = sum(p.pow(2).sum() for p in self.embedding_parameters())
        total = bpr + self.lambda_reg * reg
        parts = {"bpr": bpr.item()}
        if self.c_lambda:
            names = ("UB", "UI", "BI")
            inter = [info_nce(views[a][k][idx], views[c][k][idx], self.c_temp)
                     for k, idx in ((0, users), (1, pos))
                     for i, a in enumerate(names) for c in names[i + 1:]]
            intra = [info_nce(views[a][k][idx], views[a][k][idx], self.c_temp)
                     for k, idx in ((0, users), (1, pos)) for a in names]
            c_loss = (self.gammas[0] * sum(inter) / len(inter)
                      + self.gammas[1] * sum(intra) / len(intra))
            total = total + self.c_lambda * c_loss
            parts["contrastive"] = c_loss.item()
        return total, parts

    def embeddings(self):
        return self.fused(noise=False)
