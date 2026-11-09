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


class BundleGT(BundleTorchRecommender):
    """BundleGT: Strategy-aware Bundle Recommender System (hierarchical graph transformer).

    A LightGCN over the user-item graph gives collaborative user and item
    vectors. Each of ``num_trans_layers`` transformer layers then (1) passes
    one normalized message over the user-bundle and bundle-item graphs,
    (2) encodes each bundle by masked self-attention over its first
    ``num_token`` items and (3) encodes each user by masked self-attention over
    their first ``num_user_token`` training bundles, mixing the attention pool
    and the graph message with weight ``alpha``. The score is the dot product of
    the final user and bundle vectors; training uses BPR plus L2 on all
    embedding tables.

    Parameters
    ----------
    name: str, default: 'BundleGT'

    emb_size: int, default: 64

    num_ui_layers: int, default: 4
        LightGCN layers of the user-item branch.

    num_trans_layers: int, default: 3
        Transformer layers.

    num_token: int, default: 70
        Items per bundle kept for bundle attention.

    num_user_token: int, default: 70
        Bundles per user kept for user attention (hard-coded per dataset in the
        reference code: 70 Youshu, 30 NetEase, 5 iFashion).

    alpha: float, default: 0.5
        Weight of the attention pool against the graph message.

    gcn_norm: bool, default: False
        Weight user-item layers by 1/(l+1) and L2-normalize them.

    num_epochs: int, default: 100

    learning_rate: float, default: 1e-3

    batch_size: int, default: 2048

    lambda_reg: float, default: 1e-5
        L2 weight on the user, bundle and item tables (no Adam weight decay).

    official_variant: bool, default: False
        False trains the attention projections, as the paper describes. True
        reproduces the released code, where they are never registered with the
        optimizer and stay at their random initialization.

    eval_every, patience, eval_k, device, trainable, verbose, seed:
        See :class:`cornac.models.bundle_utils.BundleTorchRecommender`.

    Notes
    -----
    Clean-room reimplementation from the paper; settings checked against
    https://github.com/Xiaohao-Liu/BundleGT (commit d7779ec) and its issue #1
    (NetEase and iFashion settings). Memory grows with
    ``num_bundles * num_token`` and ``num_users * num_user_token``.

    References
    ----------
    * Wei, Y., Liu, X., Ma, Y., Wang, X., Nie, L., & Chua, T. S. (2023).
      Strategy-aware Bundle Recommender System. SIGIR 2023.
    """

    def __init__(
        self,
        name="BundleGT",
        emb_size=64,
        num_ui_layers=4,
        num_trans_layers=3,
        num_token=70,
        num_user_token=70,
        alpha=0.5,
        gcn_norm=False,
        num_epochs=100,
        learning_rate=1e-3,
        batch_size=2048,
        lambda_reg=1e-5,
        official_variant=False,
        eval_every=10,
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
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.num_ui_layers = num_ui_layers
        self.num_trans_layers = num_trans_layers
        self.num_token = num_token
        self.num_user_token = num_user_token
        self.alpha = alpha
        self.gcn_norm = gcn_norm
        self.official_variant = official_variant

    def _build_module(self, train_set, device):
        from .bundlegt import BundleGTModule

        return BundleGTModule(
            ui=user_item_matrix(train_set),
            ub=binary(train_set.csr_matrix),
            bi=binary(train_set.bundle_item_matrix),
            emb_size=self.emb_size,
            num_ui_layers=self.num_ui_layers,
            num_trans_layers=self.num_trans_layers,
            num_token=self.num_token,
            num_user_token=self.num_user_token,
            alpha=self.alpha,
            gcn_norm=self.gcn_norm,
            lambda_reg=self.lambda_reg,
            official_variant=self.official_variant,
            device=device,
        )

    def _optimizer(self, module):
        import torch

        params = [p for p in module.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.learning_rate)
