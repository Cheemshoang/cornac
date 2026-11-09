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


class CrossCBR(BundleTorchRecommender):
    """CrossCBR: Cross-view Contrastive Learning for Bundle Recommendation.

    Two views are learned with LightGCN. The item view propagates over the
    user-item graph and represents a bundle by the mean of its items; the
    bundle view propagates over the user-bundle graph. The score is the sum of
    the two views' dot products. Training minimizes BPR plus ``c_lambda`` times
    an InfoNCE loss aligning each user's (and positive bundle's) two views, with
    edge dropout regenerated every epoch as augmentation.

    Parameters
    ----------
    name: str, default: 'CrossCBR'

    emb_size: int, default: 64
        Embedding size of each view.

    num_layers: int, default: 1
        LightGCN layers. Propagated layers are L2-normalized and summed with the
        raw layer-0 embedding.

    num_epochs: int, default: 100

    learning_rate: float, default: 1e-3

    batch_size: int, default: 2048

    lambda_reg: float, default: 1e-4
        Adam weight decay.

    item_dropout, bundle_dropout, agg_dropout: float, default: 0.2
        Edge dropout ratios of the user-item graph, the user-bundle graph and
        the bundle-item aggregation matrix (training only).

    c_lambda: float, default: 0.04
        Weight of the contrastive loss.

    c_temp: float, default: 0.25
        InfoNCE temperature.

    use_item_view, use_bundle_view: bool, default: True
        Ablation switches. Disabling a view also disables the contrastive loss.

    eval_every, patience, eval_k, device, trainable, verbose, seed:
        See :class:`cornac.models.bundle_utils.BundleTorchRecommender`.

    Notes
    -----
    Clean-room reimplementation from the paper. Hyperparameters and protocol
    were checked against https://github.com/mysbupt/CrossCBR (commit bbe80df),
    whose ``config.yaml`` gives the per-dataset settings used in
    ``examples/bundle_crosscbr_repro.py``. Without user-item feedback the item
    view uses the items of each user's training bundles.

    References
    ----------
    * Ma, Y., He, Y., Zhang, A., Wang, X., & Chua, T. S. (2022). CrossCBR:
      Cross-view Contrastive Learning for Bundle Recommendation. KDD 2022.
    """

    def __init__(
        self,
        name="CrossCBR",
        emb_size=64,
        num_layers=1,
        num_epochs=100,
        learning_rate=1e-3,
        batch_size=2048,
        lambda_reg=1e-4,
        item_dropout=0.2,
        bundle_dropout=0.2,
        agg_dropout=0.2,
        c_lambda=0.04,
        c_temp=0.25,
        use_item_view=True,
        use_bundle_view=True,
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
        if not (use_item_view or use_bundle_view):
            raise ValueError("at least one of use_item_view and use_bundle_view")
        self.num_layers = num_layers
        self.item_dropout = item_dropout
        self.bundle_dropout = bundle_dropout
        self.agg_dropout = agg_dropout
        self.c_lambda = c_lambda
        self.c_temp = c_temp
        self.use_item_view = use_item_view
        self.use_bundle_view = use_bundle_view

    def _build_module(self, train_set, device):
        from .crosscbr import CrossCBRModule

        return CrossCBRModule(
            ui=user_item_matrix(train_set),
            ub=binary(train_set.csr_matrix),
            bi=binary(train_set.bundle_item_matrix),
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            item_dropout=self.item_dropout,
            bundle_dropout=self.bundle_dropout,
            agg_dropout=self.agg_dropout,
            c_lambda=self.c_lambda,
            c_temp=self.c_temp,
            use_item_view=self.use_item_view,
            use_bundle_view=self.use_bundle_view,
            rng=self.rng,
            device=device,
        )
