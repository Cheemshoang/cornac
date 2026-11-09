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
"""PyTorch module of CrossCBR (imported only inside ``fit``)."""

import torch
import torch.nn as nn

from ..bundle_utils.graph import bi_norm, bipartite, edge_dropout, row_norm
from ..bundle_utils.torch_utils import bpr_loss, info_nce, lightgcn_propagate, sp_to_torch


class CrossCBRModule(nn.Module):
    """Item view (LightGCN on user-item, bundles = mean of their items) and
    bundle view (LightGCN on user-bundle), aligned by cross-view InfoNCE."""

    def __init__(self, ui, ub, bi, emb_size, num_layers, item_dropout,
                 bundle_dropout, agg_dropout, c_lambda, c_temp, use_item_view,
                 use_bundle_view, rng, device):
        super().__init__()
        self.ui, self.ub, self.bi = ui, ub, bi
        self.num_users, self.num_items = ui.shape
        self.num_bundles = ub.shape[1]
        self.num_layers = num_layers
        self.dropouts = (item_dropout, bundle_dropout, agg_dropout)
        self.c_lambda, self.c_temp = c_lambda, c_temp
        self.use_item_view, self.use_bundle_view = use_item_view, use_bundle_view
        self.rng, self.device = rng, device

        self.users = nn.Parameter(torch.empty(self.num_users, emb_size))
        self.bundles = nn.Parameter(torch.empty(self.num_bundles, emb_size))
        self.items = nn.Parameter(torch.empty(self.num_items, emb_size))
        for weight in (self.users, self.bundles, self.items):
            nn.init.xavier_normal_(weight)

        self.clean = self._graphs(0.0, 0.0, 0.0)
        self.graphs = self.clean

    def _graphs(self, item_dropout, bundle_dropout, agg_dropout):
        ui = edge_dropout(self.ui, item_dropout, self.rng)
        ub = edge_dropout(self.ub, bundle_dropout, self.rng)
        bi = edge_dropout(self.bi, agg_dropout, self.rng)
        return (
            sp_to_torch(bi_norm(bipartite(ui)), self.device),
            sp_to_torch(bi_norm(bipartite(ub)), self.device),
            sp_to_torch(row_norm(bi), self.device),
        )

    def epoch_start(self, epoch):
        # Fresh edge-dropout graphs every epoch (training only).
        self.graphs = self._graphs(*self.dropouts) if any(self.dropouts) else self.clean

    def propagate(self, graphs):
        ui_graph, ub_graph, agg_graph = graphs
        nu = self.num_users
        out = {}
        if self.use_item_view:
            feats = lightgcn_propagate(
                ui_graph, torch.cat([self.users, self.items]), self.num_layers,
                normalize=True,
            )
            users_i, items_i = feats[:nu], feats[nu:]
            out["item"] = (users_i, torch.sparse.mm(agg_graph, items_i))
        if self.use_bundle_view:
            feats = lightgcn_propagate(
                ub_graph, torch.cat([self.users, self.bundles]), self.num_layers,
                normalize=True,
            )
            out["bundle"] = (feats[:nu], feats[nu:])
        return out

    def loss(self, users, pos, neg):
        views = self.propagate(self.graphs)
        pos_score = sum((u[users] * b[pos]).sum(1) for u, b in views.values())
        neg_score = sum((u[users] * b[neg]).sum(1) for u, b in views.values())
        bpr = bpr_loss(pos_score, neg_score)
        if len(views) < 2 or self.c_lambda == 0:
            return bpr, {"bpr": bpr.item()}
        (u_i, b_i), (u_b, b_b) = views["item"], views["bundle"]
        c_loss = (info_nce(u_i[users], u_b[users], self.c_temp)
                  + info_nce(b_i[pos], b_b[pos], self.c_temp)) / 2
        total = bpr + self.c_lambda * c_loss
        return total, {"bpr": bpr.item(), "contrastive": c_loss.item()}

    def embeddings(self):
        views = self.propagate(self.clean)
        users = torch.cat([u for u, _ in views.values()], dim=1)
        bundles = torch.cat([b for _, b in views.values()], dim=1)
        return users, bundles
