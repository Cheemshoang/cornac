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
"""PyTorch module of BGCN (imported only inside ``fit``)."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bundle_utils.graph import bi_norm, bipartite, edge_dropout, row_norm
from ..bundle_utils.torch_utils import bpr_loss, sp_to_torch


def bundle_overlap(bi):
    """Bundle-bundle overlap: cosine similarity of bundle item sets (diagonal 1)."""
    norms = np.sqrt(np.asarray(bi.multiply(bi).sum(axis=1)).ravel())
    scaled = sp.diags(1.0 / np.maximum(norms, 1e-8)) @ bi
    return (scaled @ scaled.T).tocsr()


class GraphConv(nn.Module):
    """Stacked graph convolutions e_{l+1} = LeakyReLU(W_l (G e_l) + b_l); the
    output concatenates the L2-normalized layers 0..L."""

    def __init__(self, emb_size, num_layers, message_dropout):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(emb_size, emb_size) for _ in range(num_layers))
        self.dropout = nn.Dropout(message_dropout)

    def forward(self, graph, x):
        outputs = [F.normalize(x, p=2, dim=1)]
        h = x
        for layer in self.layers:
            h = self.dropout(F.leaky_relu(layer(torch.sparse.mm(graph, h)), 0.1))
            outputs.append(F.normalize(h, p=2, dim=1))
        return torch.cat(outputs, dim=1)


class BGCNModule(nn.Module):
    """Item level (user-item graph with self-loops; bundles = mean of items) and
    bundle level (user-bundle graph with bundle-overlap edges), scores summed."""

    def __init__(self, ui, ub, bi, emb_size, num_layers, message_dropout,
                 node_dropout, lambda_reg, use_item_level, use_bundle_level, rng, device):
        super().__init__()
        self.num_users, self.num_items = ui.shape
        self.num_bundles = ub.shape[1]
        self.ui, self.ub, self.bi = ui, ub, bi
        self.node_dropout, self.lambda_reg = node_dropout, lambda_reg
        self.use_item_level, self.use_bundle_level = use_item_level, use_bundle_level
        self.rng, self.device = rng, device

        self.users = nn.Parameter(torch.empty(self.num_users, emb_size))
        self.items = nn.Parameter(torch.empty(self.num_items, emb_size))
        self.bundles = nn.Parameter(torch.empty(self.num_bundles, emb_size))
        for weight in (self.users, self.items, self.bundles):
            nn.init.xavier_normal_(weight)
        self.item_conv = GraphConv(emb_size, num_layers, message_dropout)
        self.bundle_conv = GraphConv(emb_size, num_layers, message_dropout)

        self.overlap = bundle_overlap(bi)
        self.agg = sp_to_torch(row_norm(bi), device)
        self.clean = self._graphs(0.0)
        self.graphs = self.clean

    def _graphs(self, ratio):
        ui = edge_dropout(self.ui, ratio, self.rng)
        ub = edge_dropout(self.ub, ratio, self.rng)
        eye_u = sp.identity(self.num_users, format="csr")
        item_graph = bipartite(ui, eye_u, sp.identity(self.num_items, format="csr"))
        bundle_graph = bipartite(ub, eye_u, self.overlap)
        return (sp_to_torch(bi_norm(item_graph), self.device),
                sp_to_torch(bi_norm(bundle_graph), self.device))

    def epoch_start(self, epoch):
        self.graphs = self._graphs(self.node_dropout) if self.node_dropout else self.clean

    def propagate(self, graphs):
        item_graph, bundle_graph = graphs
        nu = self.num_users
        out = {}
        if self.use_item_level:
            feats = self.item_conv(item_graph, torch.cat([self.users, self.items]))
            bundles = F.normalize(torch.sparse.mm(self.agg, feats[nu:]), p=2, dim=1)
            out["item"] = (feats[:nu], bundles)
        if self.use_bundle_level:
            feats = self.bundle_conv(bundle_graph, torch.cat([self.users, self.bundles]))
            out["bundle"] = (feats[:nu], feats[nu:])
        return out

    def loss(self, users, pos, neg):
        views = self.propagate(self.graphs)
        pos_score = sum((u[users] * b[pos]).sum(1) for u, b in views.values())
        neg_score = sum((u[users] * b[neg]).sum(1) for u, b in views.values())
        bpr = bpr_loss(pos_score, neg_score)
        reg = sum(u[users].pow(2).sum() + b[pos].pow(2).sum() + b[neg].pow(2).sum()
                  for u, b in views.values()) / (2 * len(users))
        return bpr + self.lambda_reg * reg, {"bpr": bpr.item()}

    def embeddings(self):
        views = self.propagate(self.clean)
        return (torch.cat([u for u, _ in views.values()], dim=1),
                torch.cat([b for _, b in views.values()], dim=1))
