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

VIEWS = ("UB", "UI", "BI")


def _per_view(values, name):
    values = tuple(values)
    if len(values) != 3:
        raise ValueError("{} needs one value per view (UB, UI, BI)".format(name))
    return dict(zip(VIEWS, values))


class MultiCBR(BundleTorchRecommender):
    """MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation.

    Three LightGCN views are learned: user-bundle (UB), user-item (UI, bundles
    as the mean of their items) and bundle-item (BI, users as the mean of their
    items). Each view combines its layers with fixed ``*_layer_weights`` and
    the views are fused early with fixed ``modal_weights``; the score is the dot
    product of the fused user and bundle vectors. Training uses BPR plus
    ``c_lambda`` times InfoNCE on the fused users and positive bundles, with
    noise augmentation (``view_eps``) after every propagation and aggregation.

    Parameters
    ----------
    name: str, default: 'MultiCBR'

    emb_size: int, default: 64

    num_layers: int, default: 2

    num_epochs: int, default: 100

    learning_rate: float, default: 1e-3

    batch_size: int, default: 2048

    lambda_reg: float, default: 1e-5
        Adam weight decay.

    view_eps: tuple of 3 floats, default: (0.2, 0.1, 0.1)
        Noise magnitude of the UB, UI and BI views (training only).

    modal_weights: tuple of 3 floats, default: (0.35, 0.15, 0.5)
        Fusion weights of the UB, UI and BI views. A zero weight ablates a view.

    ub_layer_weights, ui_layer_weights, bi_layer_weights: tuple of floats
        Weights of layers 0..num_layers in each view (defaults: the official
        Youshu settings).

    c_lambda: float, default: 0.04
        Weight of the contrastive loss.

    c_temp: float, default: 0.23
        InfoNCE temperature.

    official_variant: bool, default: False
        False follows the paper: the contrastive loss aligns two independently
        perturbed forward passes. True follows the released code, which
        contrasts each embedding of a single pass with itself (a uniformity term).

    eval_every, patience, eval_k, device, trainable, verbose, seed:
        See :class:`cornac.models.bundle_utils.BundleTorchRecommender`.

    Notes
    -----
    Clean-room reimplementation from the paper; settings checked against
    https://github.com/HappyPointer/MultiCBR (commit 744cec8).

    References
    ----------
    * Ma, Y., He, Y., Wang, X., Wei, Y., Du, X., Fu, Y., & Chua, T. S. (2024).
      MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation.
      ACM Transactions on Information Systems.
    """

    def __init__(
        self,
        name="MultiCBR",
        emb_size=64,
        num_layers=2,
        num_epochs=100,
        learning_rate=1e-3,
        batch_size=2048,
        lambda_reg=1e-5,
        view_eps=(0.2, 0.1, 0.1),
        modal_weights=(0.35, 0.15, 0.5),
        ub_layer_weights=(0.7, 0.2, 0.1),
        ui_layer_weights=(0.6, 0.2, 0.2),
        bi_layer_weights=(0.6, 0.1, 0.3),
        c_lambda=0.04,
        c_temp=0.23,
        official_variant=False,
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
        self.view_eps = tuple(view_eps)
        self.modal_weights = tuple(modal_weights)
        self.ub_layer_weights = tuple(ub_layer_weights)
        self.ui_layer_weights = tuple(ui_layer_weights)
        self.bi_layer_weights = tuple(bi_layer_weights)
        self.c_lambda = c_lambda
        self.c_temp = c_temp
        self.official_variant = official_variant
        _per_view(self.view_eps, "view_eps")
        _per_view(self.modal_weights, "modal_weights")
        for weights in (self.ub_layer_weights, self.ui_layer_weights, self.bi_layer_weights):
            if len(weights) != num_layers + 1:
                raise ValueError("layer weights need num_layers + 1 values")

    def _build_module(self, train_set, device):
        from .multicbr import MultiCBRModule

        return MultiCBRModule(
            ui=user_item_matrix(train_set),
            ub=binary(train_set.csr_matrix),
            bi=binary(train_set.bundle_item_matrix),
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            layer_weights={"UB": self.ub_layer_weights, "UI": self.ui_layer_weights,
                           "BI": self.bi_layer_weights},
            view_eps=_per_view(self.view_eps, "view_eps"),
            modal_weights=_per_view(self.modal_weights, "modal_weights"),
            c_lambda=self.c_lambda,
            c_temp=self.c_temp,
            official_variant=self.official_variant,
            device=device,
        )
