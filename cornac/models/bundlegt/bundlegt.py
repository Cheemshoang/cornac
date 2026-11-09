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
"""PyTorch module of BundleGT (imported only inside ``fit``)."""

import math

import numpy as np
import torch
import torch.nn as nn

from ..bundle_utils.graph import bi_norm, bipartite
from ..bundle_utils.torch_utils import bpr_loss, lightgcn_propagate, sp_to_torch


def token_sequences(matrix, max_tokens):
    """Padded (rows x max_tokens) index matrix of each row's first nonzero columns.

    Padding uses index ``matrix.shape[1]``; the mask is True on real tokens.
    """
    rows = matrix.shape[0]
    pad = matrix.shape[1]
    tokens = np.full((rows, max(max_tokens, 1)), pad, dtype=np.int64)
    for r in range(rows):
        cols = matrix.indices[matrix.indptr[r]: matrix.indptr[r + 1]][:max_tokens]
        tokens[r, : len(cols)] = cols
    return tokens, tokens != pad


class SetAttention(nn.Module):
    """Single-head masked self-attention over a token set (values = tokens),
    mean-pooled over the real tokens."""

    def __init__(self, emb_size, trainable=True):
        super().__init__()
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        if not trainable:
            for weight in self.parameters():
                weight.requires_grad_(False)

    def forward(self, tokens, mask):
        q, k = self.query(tokens), self.key(tokens)
        scores = q @ k.transpose(1, 2) / math.sqrt(tokens.shape[-1])
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1))  # empty sets -> 0
        out = (attn @ tokens) * mask.unsqueeze(-1)
        return out.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)


class BundleGTModule(nn.Module):
    """LightGCN on user-item, then transformer layers that encode bundles from
    their items and users from their bundles, plus one-hop UB and BI messages."""

    def __init__(self, ui, ub, bi, emb_size, num_ui_layers, num_trans_layers, num_token,
                 num_user_token, alpha, gcn_norm, lambda_reg, official_variant, device):
        super().__init__()
        self.num_users, self.num_items = ui.shape
        self.num_bundles = ub.shape[1]
        self.num_ui_layers, self.gcn_norm = num_ui_layers, gcn_norm
        self.alpha, self.lambda_reg = alpha, lambda_reg
        self.official_variant = official_variant

        self.users = nn.Parameter(torch.empty(self.num_users, emb_size))
        self.bundles = nn.Parameter(torch.empty(self.num_bundles, emb_size))
        self.items = nn.Parameter(torch.empty(self.num_items, emb_size))
        for weight in (self.users, self.bundles, self.items):
            nn.init.xavier_uniform_(weight)

        trainable = not official_variant
        self.bundle_attn = nn.ModuleList(SetAttention(emb_size, trainable)
                                         for _ in range(num_trans_layers))
        self.user_attn = nn.ModuleList(SetAttention(emb_size, trainable)
                                       for _ in range(num_trans_layers))

        self.ui_graph = sp_to_torch(bi_norm(bipartite(ui)), device)
        self.ub_graph = sp_to_torch(bi_norm(bipartite(ub)), device)
        self.bi_graph = sp_to_torch(bi_norm(bipartite(bi)), device)
        bundle_tokens, bundle_mask = token_sequences(bi, num_token)
        user_tokens, user_mask = token_sequences(ub, num_user_token)
        self.bundle_tokens = torch.as_tensor(bundle_tokens, device=device)
        self.bundle_mask = torch.as_tensor(bundle_mask, device=device)
        self.user_tokens = torch.as_tensor(user_tokens, device=device)
        self.user_mask = torch.as_tensor(user_mask, device=device)

    def _ui_branch(self):
        # gcn_norm: L2-normalized propagated layers summed with the raw layer 0 (the
        # reference also divides by l+1, which the normalization cancels).
        feats = lightgcn_propagate(self.ui_graph, torch.cat([self.users, self.items]),
                                   self.num_ui_layers, normalize=self.gcn_norm)
        return feats[: self.num_users], feats[self.num_users:]

    @staticmethod
    def _message(graph, left, right, to_left):
        """One normalized message over a bipartite graph, from one side only."""
        if to_left:
            out = torch.sparse.mm(graph, torch.cat([torch.zeros_like(left), right]))
            return out[: left.shape[0]]
        out = torch.sparse.mm(graph, torch.cat([left, torch.zeros_like(right)]))
        return out[left.shape[0]:]

    def forward_all(self):
        """HGT layers (paper Eq. 4-15).

        Paper: the bundle residual aggregates the LightGCN items i^cf (Eq. 9) and the
        user residual aggregates the current layer's bundles b^(l) (Eq. 14).
        Released code (official_variant): both residuals are one-hop messages taken at
        the start of the layer, from the previous layer's items and bundles.
        """
        nu, nb = self.num_users, self.num_bundles
        u_cf, i_cf = self._ui_branch()
        u, b, i = self.users, self.bundles, self.items
        for bundle_attn, user_attn in zip(self.bundle_attn, self.user_attn):
            ub = torch.sparse.mm(self.ub_graph, torch.cat([u, b]))
            u_b2u, b_u2b = ub[:nu], ub[nu:]               # messages from u^(l-1), b^(l-1)
            bi = torch.sparse.mm(self.bi_graph, torch.cat([b, i]))
            b_i2b, i_b2i = bi[:nb], bi[nb:]
            i = i_cf + i_b2i                                # Eq. 4-5: tokens + positions
            item_table = torch.cat([i, i.new_zeros(1, i.shape[1])])
            b_pool = bundle_attn(item_table[self.bundle_tokens], self.bundle_mask)
            if not self.official_variant:                   # Eq. 9: aggregate i^cf
                b_i2b = self._message(self.bi_graph, b, i_cf, to_left=True)
            b = self.alpha * b_pool + (1 - self.alpha) * b_i2b + b_u2b  # Eq. 9-10
            bundle_table = torch.cat([b, b.new_zeros(1, b.shape[1])])
            u_pool = user_attn(bundle_table[self.user_tokens], self.user_mask)
            if not self.official_variant:                   # Eq. 14: aggregate b^(l)
                u_b2u = self._message(self.ub_graph, u, b, to_left=True)
            u = self.alpha * u_pool + (1 - self.alpha) * u_b2u + u_cf    # Eq. 14-15
        return u, b

    def loss(self, users, pos, neg):
        u, b = self.forward_all()
        bpr = bpr_loss((u[users] * b[pos]).sum(1), (u[users] * b[neg]).sum(1))
        tables = (self.users, self.bundles, self.items)
        if self.official_variant:  # released code: embedding tables only
            params = tables
        else:  # paper Eq. 18: all trainable parameters
            params = tables + tuple(p for p in self.parameters() if p.requires_grad
                                    and all(p is not t for t in tables))
        reg = sum(p.pow(2).sum() for p in params)
        return bpr + self.lambda_reg * reg, {"bpr": bpr.item()}

    def embeddings(self):
        return self.forward_all()
