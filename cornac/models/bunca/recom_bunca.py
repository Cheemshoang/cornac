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

from ..bundle_utils import BundleTorchRecommender
from ..bundle_utils.graph import binary, user_item_matrix


class BunCa(BundleTorchRecommender):
    """BunCa: Bundle Recommendation with Item-level Causation-enhanced Multi-view Learning.

    Item representations are enhanced by multi-head GATv2-style attention over
    two directed item-item graphs built from training co-occurrences: items
    sharing at least ``iui_threshold`` users and items sharing at least
    ``ibi_threshold`` bundles. Two views follow:

    - Coherent view: a user-preference sub-view (LightGCN on user-item, bundles
      as the mean of their items) and a bundle-composition sub-view (LightGCN
      on bundle-item, users as the mean of their items), mixed by
      ``coherent_weights``.
    - Cohesive view: LightGCN on the user-bundle graph extended with user-user
      and bundle-bundle edges (pairs sharing more than ``hyper_threshold``
      neighbors), scaled by ``cohesive_weights``.

    The score sums the two views' dot products. Training uses BPR plus
    ``c_lambda`` times a weighted sum of cross-view InfoNCE (coherent vs
    cohesive) and native InfoNCE (combined representation vs itself) for users
    and bundles, with CrossCBR-style edge dropout.

    Parameters
    ----------
    name: str, default: 'BunCa'

    emb_size: int, default: 64

    num_layers: int, default: 1

    num_epochs: int, default: 100

    learning_rate: float, default: 1e-3

    batch_size: int, default: 2048

    lambda_reg: float, default: 1e-4
        Adam weight decay.

    item_dropout, bundle_dropout, agg_dropout: float, default: 0.2
        Edge dropout of the user-item, user-bundle and bundle-item graphs.

    iui_threshold, ibi_threshold: int, default: 4
        Minimum shared users / bundles for an item-item edge.

    hyper_threshold: int, default: 4
        User-user and bundle-bundle edges need more than this many shared neighbors.

    num_heads: int, default: 2
        Attention heads (averaged).

    attn_dropout: float, default: 0.1
        Dropout on attention weights.

    attn_weight, residual_weight: float, default: 1.0
        Item features are ``attn_weight * attended + residual_weight * raw``.

    coherent_weights: (float, float), default: (0.9, 0.9)
        Weights of the bundle-composition sub-view for bundles and for users.

    cohesive_weights: (float, float), default: (1.0, 1.0)
        Scales of cohesive-view users and bundles.

    c_lambda: float, default: 0.04

    c_temp: float, default: 0.25

    contrast_weights: 4 floats, default: (0.25, 0.25, 0.25, 0.25)
        Weights of user cross-view, bundle cross-view, user native and bundle
        native InfoNCE terms.

    eval_every, patience, eval_k, device, trainable, verbose, seed:
        See :class:`cornac.models.bundle_utils.BundleTorchRecommender`.

    Notes
    -----
    Clean-room reimplementation from the paper. Defaults follow the paper with
    every component on (residual connection, all contrastive terms, user-user
    and bundle-bundle edges). The released configurations
    (https://github.com/bt-nghia/BunCa, commit dcff093) switch some of these
    off per dataset; ``examples/bundle_repro.py --official`` applies them.
    Attention is implemented in plain PyTorch (no torch-geometric).

    References
    ----------
    * Nguyen, H. S., Bui, T. N., Nguyen, L. H., Hoang, H., Nguyen, C. V. T.,
      Le, H. Q., & Le, D. T. (2024). Bundle Recommendation with Item-level
      Causation-enhanced Multi-view Learning. ECML-PKDD 2024.
    """

    def __init__(
        self,
        name="BunCa",
        emb_size=64,
        num_layers=1,
        num_epochs=100,
        learning_rate=1e-3,
        batch_size=2048,
        lambda_reg=1e-4,
        item_dropout=0.2,
        bundle_dropout=0.2,
        agg_dropout=0.2,
        iui_threshold=4,
        ibi_threshold=4,
        hyper_threshold=4,
        num_heads=2,
        attn_dropout=0.1,
        attn_weight=1.0,
        residual_weight=1.0,
        coherent_weights=(0.9, 0.9),
        cohesive_weights=(1.0, 1.0),
        c_lambda=0.04,
        c_temp=0.25,
        contrast_weights=(0.25, 0.25, 0.25, 0.25),
        eval_every=5,
        patience=None,
        eval_k=20,
        device=None,
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(
            name=name, emb_size=emb_size, num_epochs=num_epochs,
            learning_rate=learning_rate, batch_size=batch_size,
            lambda_reg=lambda_reg, eval_every=eval_every, patience=patience,
            eval_k=eval_k, device=device, trainable=trainable, verbose=verbose,
            seed=seed,
        )
        self.num_layers = num_layers
        self.item_dropout = item_dropout
        self.bundle_dropout = bundle_dropout
        self.agg_dropout = agg_dropout
        self.iui_threshold = iui_threshold
        self.ibi_threshold = ibi_threshold
        self.hyper_threshold = hyper_threshold
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.attn_weight = attn_weight
        self.residual_weight = residual_weight
        self.coherent_weights = tuple(coherent_weights)
        self.cohesive_weights = tuple(cohesive_weights)
        self.c_lambda = c_lambda
        self.c_temp = c_temp
        self.contrast_weights = tuple(contrast_weights)
        if len(self.coherent_weights) != 2 or len(self.cohesive_weights) != 2:
            raise ValueError("coherent_weights and cohesive_weights need 2 values")
        if len(self.contrast_weights) != 4:
            raise ValueError("contrast_weights needs 4 values")

    def _build_module(self, train_set, device):
        from .bunca import BunCaModule

        return BunCaModule(
            ui=user_item_matrix(train_set),
            ub=binary(train_set.csr_matrix),
            bi=binary(train_set.bundle_item_matrix),
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            item_dropout=self.item_dropout,
            bundle_dropout=self.bundle_dropout,
            agg_dropout=self.agg_dropout,
            iui_threshold=self.iui_threshold,
            ibi_threshold=self.ibi_threshold,
            hyper_threshold=self.hyper_threshold,
            num_heads=self.num_heads,
            attn_dropout=self.attn_dropout,
            attn_weight=self.attn_weight,
            residual_weight=self.residual_weight,
            coherent_weights=self.coherent_weights,
            cohesive_weights=self.cohesive_weights,
            c_lambda=self.c_lambda,
            c_temp=self.c_temp,
            contrast_weights=self.contrast_weights,
            rng=self.rng,
            device=device,
        )
