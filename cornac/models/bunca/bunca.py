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
"""PyTorch module of BunCa (imported only inside ``fit``)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bundle_utils.graph import bi_norm, binary, bipartite, edge_dropout, row_norm
from ..bundle_utils.torch_utils import (
    bpr_loss, info_nce, lightgcn_propagate, segment_softmax, sp_to_torch,
)


def cooccurrence(matrix, threshold):
    """Binary item-item graph: items co-occurring in at least ``threshold`` rows."""
    counts = (matrix.T @ matrix).tocsr()
    counts.setdiag(0)
    counts.data = (counts.data >= threshold).astype(np.float64)
    counts.eliminate_zeros()
    return counts


def shared_neighbors(matrix, threshold):
    """Binary row-row graph: rows sharing more than ``threshold`` columns."""
    counts = (matrix @ matrix.T).tocsr()
    counts.setdiag(0)
    counts.data = (counts.data > threshold).astype(np.float64)
    counts.eliminate_zeros()
    return counts


class AsymmetricAttention(nn.Module):
    """GATv2-style multi-head attention over a directed item-item graph.

    For an edge j -> i: e_ij = a^T LeakyReLU(W_t x_i + W_s x_j); weights are
    softmax-normalized over the in-neighbors of i, and heads are averaged. The
    output is the mean of the input and the attended messages.
    """

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads, self.emb_size = num_heads, emb_size
        self.source = nn.Linear(emb_size, num_heads * emb_size, bias=False)
        self.target = nn.Linear(emb_size, num_heads * emb_size, bias=False)
        self.att = nn.Parameter(torch.empty(num_heads, emb_size))
        nn.init.xavier_uniform_(self.att)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src, dst):
        if src.numel() == 0:
            return x
        n, h, d = x.shape[0], self.num_heads, self.emb_size
        xs = self.source(x).view(n, h, d)
        xt = self.target(x).view(n, h, d)
        scores = (F.leaky_relu(xt[dst] + xs[src], 0.2) * self.att).sum(-1)  # (E, H)
        alpha = self.dropout(segment_softmax(scores, dst, n)).unsqueeze(-1)
        out = torch.zeros(n, h, d, device=x.device, dtype=x.dtype)
        out = out.index_add(0, dst, alpha * xs[src]).mean(dim=1)
        return (x + out) / 2


class BunCaModule(nn.Module):
    """Coherent view (user-item and bundle-item sub-views over causation-enhanced
    items) and cohesive view (user-bundle graph with user-user and bundle-bundle
    edges), aligned with cross-view and native contrastive losses."""

    def __init__(self, ui, ub, bi, emb_size, num_layers, item_dropout, bundle_dropout,
                 agg_dropout, iui_threshold, ibi_threshold, hyper_threshold, num_heads,
                 attn_dropout, attn_weight, residual_weight, coherent_weights,
                 cohesive_weights, c_lambda, c_temp, contrast_weights, rng, device):
        super().__init__()
        self.ui, self.ub, self.bi = ui, ub, bi
        self.num_users, self.num_items = ui.shape
        self.num_bundles = ub.shape[1]
        self.num_layers = num_layers
        self.dropouts = (item_dropout, bundle_dropout, agg_dropout)
        self.attn_weight, self.residual_weight = attn_weight, residual_weight
        self.coherent_weights = coherent_weights  # (w_bundle_bc, w_user_bc)
        self.cohesive_weights = cohesive_weights  # (w_user, w_bundle)
        self.c_lambda, self.c_temp = c_lambda, c_temp
        self.contrast_weights = contrast_weights  # (u_cross, b_cross, u_native, b_native)
        self.rng, self.device = rng, device

        self.users = nn.Parameter(torch.empty(self.num_users, emb_size))
        self.bundles = nn.Parameter(torch.empty(self.num_bundles, emb_size))
        self.items = nn.Parameter(torch.empty(self.num_items, emb_size))
        for weight in (self.users, self.bundles, self.items):
            nn.init.xavier_normal_(weight)

        # Item-item causation graphs from training user-item and bundle-item data.
        self.item_edges = []
        for graph in (cooccurrence(ui, iui_threshold), cooccurrence(bi, ibi_threshold)):
            coo = graph.tocoo()  # edge col -> row
            self.item_edges.append((torch.as_tensor(coo.col, dtype=torch.long, device=device),
                                    torch.as_tensor(coo.row, dtype=torch.long, device=device)))
        self.attention = nn.ModuleList(
            AsymmetricAttention(emb_size, num_heads, attn_dropout) for _ in self.item_edges
        )

        ub_bin = binary(ub)
        self.uu = shared_neighbors(ub_bin, hyper_threshold)
        self.bb = shared_neighbors(ub_bin.T.tocsr(), hyper_threshold)
        self.clean = self._graphs(0.0, 0.0, 0.0)
        self.graphs = self.clean

    def _graphs(self, item_dropout, bundle_dropout, agg_dropout):
        ui = edge_dropout(self.ui, item_dropout, self.rng)
        ub = edge_dropout(self.ub, bundle_dropout, self.rng)
        bi = edge_dropout(self.bi, agg_dropout, self.rng)
        return {
            "UI": sp_to_torch(bi_norm(bipartite(ui)), self.device),
            "BI": sp_to_torch(bi_norm(bipartite(bi)), self.device),
            "UB": sp_to_torch(bi_norm(bipartite(ub, self.uu, self.bb)), self.device),
            "bundle_agg": sp_to_torch(row_norm(bi), self.device),
            "user_agg": sp_to_torch(row_norm(ui), self.device),
        }

    def epoch_start(self, epoch):
        self.graphs = self._graphs(*self.dropouts) if any(self.dropouts) else self.clean

    def enhanced_items(self):
        attended = [att(self.items, src, dst)
                    for att, (src, dst) in zip(self.attention, self.item_edges)]
        attended = torch.stack(attended).mean(0) if attended else self.items
        return self.attn_weight * attended + self.residual_weight * self.items

    def views(self, graphs):
        nu, nb = self.num_users, self.num_bundles
        items = self.enhanced_items()
        # Coherent view: user-preference (UI) and bundle-composition (BI) sub-views.
        feats = lightgcn_propagate(graphs["UI"], torch.cat([self.users, items]),
                                   self.num_layers, normalize=True)
        u_up, b_up = feats[:nu], torch.sparse.mm(graphs["bundle_agg"], feats[nu:])
        feats = lightgcn_propagate(graphs["BI"], torch.cat([self.bundles, items]),
                                   self.num_layers, normalize=True)
        b_bc, u_bc = feats[:nb], torch.sparse.mm(graphs["user_agg"], feats[nb:])
        w_b, w_u = self.coherent_weights
        coherent = ((1 - w_u) * u_up + w_u * u_bc, (1 - w_b) * b_up + w_b * b_bc)
        # Cohesive view: user-bundle graph with user-user and bundle-bundle edges.
        feats = lightgcn_propagate(graphs["UB"], torch.cat([self.users, self.bundles]),
                                   self.num_layers, normalize=True)
        cohesive = (self.cohesive_weights[0] * feats[:nu], self.cohesive_weights[1] * feats[nu:])
        return coherent, cohesive

    def loss(self, users, pos, neg):
        (u_h, b_h), (u_s, b_s) = self.views(self.graphs)
        pos_score = (u_h[users] * b_h[pos]).sum(1) + (u_s[users] * b_s[pos]).sum(1)
        neg_score = (u_h[users] * b_h[neg]).sum(1) + (u_s[users] * b_s[neg]).sum(1)
        bpr = bpr_loss(pos_score, neg_score)
        w_uc, w_bc, w_un, w_bn = self.contrast_weights
        terms = []
        if w_uc:
            terms.append(w_uc * info_nce(u_h[users], u_s[users], self.c_temp))
        if w_bc:
            terms.append(w_bc * info_nce(b_h[pos], b_s[pos], self.c_temp))
        if w_un:
            u = u_h[users] + u_s[users]
            terms.append(w_un * info_nce(u, u, self.c_temp))
        if w_bn:
            b = b_h[pos] + b_s[pos]
            terms.append(w_bn * info_nce(b, b, self.c_temp))
        if not terms or self.c_lambda == 0:
            return bpr, {"bpr": bpr.item()}
        c_loss = sum(terms)
        return bpr + self.c_lambda * c_loss, {"bpr": bpr.item(), "contrastive": c_loss.item()}

    def embeddings(self):
        (u_h, b_h), (u_s, b_s) = self.views(self.clean)
        return torch.cat([u_h, u_s], dim=1), torch.cat([b_h, b_s], dim=1)
