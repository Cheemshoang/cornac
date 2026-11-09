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
"""PyTorch module of MultiCBR (imported only inside ``fit``)."""

import torch
import torch.nn as nn

from ..bundle_utils.graph import bi_norm, bipartite, row_norm
from ..bundle_utils.torch_utils import (
    bpr_loss, info_nce, lightgcn_propagate, perturb, sp_to_torch,
)


class MultiViewEncoder(nn.Module):
    """UB, UI and BI LightGCN views fused by fixed weights.

    - UB view: users and bundles from the user-bundle graph.
    - UI view: users and items from the user-item graph; bundles are the mean
      of their items.
    - BI view: bundles and items from the bundle-item graph; users are the mean
      of their interacted items.
    """

    def __init__(self, ui, ub, bi, emb_size, num_layers, layer_weights, view_eps,
                 device):
        super().__init__()
        self.num_users, self.num_items = ui.shape
        self.num_bundles = ub.shape[1]
        self.num_layers = num_layers
        self.layer_weights = layer_weights  # {"UB": [...], "UI": [...], "BI": [...]}
        self.view_eps = view_eps            # {"UB": eps, "UI": eps, "BI": eps}

        self.users = nn.Parameter(torch.empty(self.num_users, emb_size))
        self.bundles = nn.Parameter(torch.empty(self.num_bundles, emb_size))
        self.items = nn.Parameter(torch.empty(self.num_items, emb_size))
        for weight in (self.users, self.bundles, self.items):
            nn.init.xavier_normal_(weight)

        self.graphs = {
            "UB": sp_to_torch(bi_norm(bipartite(ub)), device),
            "UI": sp_to_torch(bi_norm(bipartite(ui)), device),
            "BI": sp_to_torch(bi_norm(bipartite(bi)), device),
        }
        self.bundle_agg = sp_to_torch(row_norm(bi), device)  # bundle <- its items
        self.user_agg = sp_to_torch(row_norm(ui), device)    # user <- their items

    def _view(self, name, left, right, noise):
        eps = self.view_eps[name] if noise else 0.0
        feats = lightgcn_propagate(
            self.graphs[name], torch.cat([left, right]), self.num_layers,
            layer_weights=self.layer_weights[name], normalize=True, noise_eps=eps,
        )
        return feats[: left.shape[0]], feats[left.shape[0]:], eps

    def views(self, noise):
        """{view: (users, bundles)} with noise augmentation when ``noise``."""
        u_ub, b_ub, _ = self._view("UB", self.users, self.bundles, noise)
        u_ui, i_ui, eps = self._view("UI", self.users, self.items, noise)
        b_ui = perturb(torch.sparse.mm(self.bundle_agg, i_ui), eps)
        b_bi, i_bi, eps = self._view("BI", self.bundles, self.items, noise)
        u_bi = perturb(torch.sparse.mm(self.user_agg, i_bi), eps)
        return {"UB": (u_ub, b_ub), "UI": (u_ui, b_ui), "BI": (u_bi, b_bi)}


class MultiCBRModule(nn.Module):
    """MultiCBR: early fusion of three views, BPR + contrastive regularization."""

    def __init__(self, ui, ub, bi, emb_size, num_layers, layer_weights, view_eps,
                 modal_weights, c_lambda, c_temp, official_variant, device):
        super().__init__()
        self.encoder = MultiViewEncoder(ui, ub, bi, emb_size, num_layers,
                                        layer_weights, view_eps, device)
        self.modal_weights = modal_weights  # {"UB": w, "UI": w, "BI": w}
        self.c_lambda, self.c_temp = c_lambda, c_temp
        self.official_variant = official_variant

    def fused(self, noise):
        views = self.encoder.views(noise)
        users = sum(self.modal_weights[v] * views[v][0] for v in views)
        bundles = sum(self.modal_weights[v] * views[v][1] for v in views)
        return users, bundles

    def loss(self, users, pos, neg):
        u, b = self.fused(noise=True)
        bpr = bpr_loss((u[users] * b[pos]).sum(1), (u[users] * b[neg]).sum(1))
        if self.c_lambda == 0:
            return bpr, {"bpr": bpr.item()}
        if self.official_variant:
            # Released code: each embedding is contrasted with itself.
            u2, b2 = u, b
        else:
            # Paper: a second, independently perturbed forward pass.
            u2, b2 = self.fused(noise=True)
        c_loss = (info_nce(u[users], u2[users], self.c_temp)
                  + info_nce(b[pos], b2[pos], self.c_temp)) / 2
        return bpr + self.c_lambda * c_loss, {"bpr": bpr.item(), "contrastive": c_loss.item()}

    def embeddings(self):
        return self.fused(noise=False)
