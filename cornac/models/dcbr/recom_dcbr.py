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


class DCBR(BundleTorchRecommender):
    """DCBR: Disentangled Contrastive Bundle Recommendation with Conditional Diffusion.

    Users and bundles are scored from two LightGCN views fused by ``omega``:
    the user-item view (bundles as the mean of their items) and the
    bundle-item view (users as the mean of their items), each with fixed layer
    weights and noise augmentation. A third, user-bundle view is learned on a
    graph rebuilt every epoch by a diffusion model: an MLP denoiser trained on
    the training user-bundle rows (x_0 prediction, plus ``blcc_weight`` times a
    bundle-level consistency term), whose denoised rows keep the top
    ``rebuild_k`` bundles of each user. That view enters only the inter- and
    intra-view InfoNCE losses; training uses BPR plus L2 on all embeddings.

    Parameters
    ----------
    name: str, default: 'DCBR'

    emb_size: int, default: 64

    num_layers: int, default: 2

    ub_layer_weights, ui_layer_weights, bi_layer_weights: tuple of floats
        Weights of layers 0..num_layers of each view.

    view_eps: tuple of 3 floats, default: (0.05, 0.0, 0.2)
        Noise magnitude of the UB, UI and BI views (training only).

    omega: float, default: 0.5
        Weight of the UI view in the fused (scored) representation.

    c_lambda: float, default: 0.3
        Weight of the contrastive loss.

    c_temp: float, default: 0.08
        InfoNCE temperature.

    gammas: (float, float), default: (0.3, 0.2)
        Weights of the inter-view and intra-view terms.

    lambda_reg: float, default: 1e-6
        L2 weight on all embedding parameters (no Adam weight decay).

    diff_steps: int, default: 140
        Diffusion steps.

    noise_scale, noise_min, noise_max: float, default: 0.1, 1e-4, 0.02
        Linear variance schedule ``noise_scale * linspace(noise_min, noise_max)``.

    diff_dims: tuple of int, default: (1000,)
        Hidden sizes of the denoiser MLP.

    time_emb_size: int, default: 10

    diff_dropout: float, default: 0.5
        Input dropout of the denoiser.

    diff_lr: float, default: 1e-3
        Learning rate of the denoiser.

    diff_batch_size: int, default: 1024
        Users per diffusion step (rows are densified one batch at a time).

    blcc_weight: float, default: 2.0
        Weight of the bundle-level consistency term.

    rebuild_k: int, default: 1
        Bundles kept per user in the rebuilt user-bundle graph.

    num_epochs, learning_rate, batch_size, eval_every, patience, eval_k, device,
    trainable, verbose, seed:
        See :class:`cornac.models.bundle_utils.BundleTorchRecommender`.

    Notes
    -----
    Clean-room reimplementation from the paper; settings checked against
    https://github.com/recomall/DCBR (commit 502eadf). Defaults are its
    iFashion configuration. The reference hard-codes CUDA; here the device
    follows ``device``. ``noise_min``/``noise_max`` and the denoiser's
    dropout follow DiffRec, on which the reference builds.

    References
    ----------
    * Li, J. (2025). Disentangled Contrastive Bundle Recommendation with
      Conditional Diffusion. AAAI 2025.
    """

    def __init__(
        self,
        name="DCBR",
        emb_size=64,
        num_layers=2,
        ub_layer_weights=(0.5, 0.5, 0.0),
        ui_layer_weights=(0.3, 0.4, 0.3),
        bi_layer_weights=(0.5, 0.5, 0.0),
        view_eps=(0.05, 0.0, 0.2),
        omega=0.5,
        c_lambda=0.3,
        c_temp=0.08,
        gammas=(0.3, 0.2),
        lambda_reg=1e-6,
        diff_steps=140,
        noise_scale=0.1,
        noise_min=1e-4,
        noise_max=0.02,
        diff_dims=(1000,),
        time_emb_size=10,
        diff_dropout=0.5,
        diff_lr=1e-3,
        diff_batch_size=1024,
        blcc_weight=2.0,
        rebuild_k=1,
        num_epochs=50,
        learning_rate=1e-3,
        batch_size=2048,
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
        self.ub_layer_weights = tuple(ub_layer_weights)
        self.ui_layer_weights = tuple(ui_layer_weights)
        self.bi_layer_weights = tuple(bi_layer_weights)
        self.view_eps = tuple(view_eps)
        self.omega = omega
        self.c_lambda = c_lambda
        self.c_temp = c_temp
        self.gammas = tuple(gammas)
        self.diff_steps = diff_steps
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.diff_dims = tuple(diff_dims)
        self.time_emb_size = time_emb_size
        self.diff_dropout = diff_dropout
        self.diff_lr = diff_lr
        self.diff_batch_size = diff_batch_size
        self.blcc_weight = blcc_weight
        self.rebuild_k = rebuild_k
        for weights in (self.ub_layer_weights, self.ui_layer_weights, self.bi_layer_weights):
            if len(weights) != num_layers + 1:
                raise ValueError("layer weights need num_layers + 1 values")
        if len(self.view_eps) != 3 or len(self.gammas) != 2:
            raise ValueError("view_eps needs 3 values and gammas 2 values")
        if not 0 <= omega <= 1:
            raise ValueError("omega must be in [0, 1]")
        if diff_steps < 1 or rebuild_k < 1:
            raise ValueError("diff_steps and rebuild_k must be positive")

    def _build_module(self, train_set, device):
        from .dcbr import DCBRModule

        return DCBRModule(
            ui=user_item_matrix(train_set),
            ub=binary(train_set.csr_matrix),
            bi=binary(train_set.bundle_item_matrix),
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            layer_weights=dict(zip(VIEWS, (self.ub_layer_weights, self.ui_layer_weights,
                                           self.bi_layer_weights))),
            view_eps=dict(zip(VIEWS, self.view_eps)),
            omega=self.omega,
            c_lambda=self.c_lambda,
            c_temp=self.c_temp,
            gammas=self.gammas,
            lambda_reg=self.lambda_reg,
            diff_steps=self.diff_steps,
            noise_scale=self.noise_scale,
            noise_min=self.noise_min,
            noise_max=self.noise_max,
            diff_dims=self.diff_dims,
            time_emb_size=self.time_emb_size,
            diff_dropout=self.diff_dropout,
            diff_lr=self.diff_lr,
            diff_batch_size=self.diff_batch_size,
            blcc_weight=self.blcc_weight,
            rebuild_k=self.rebuild_k,
            rng=self.rng,
            device=device,
        )

    def _optimizer(self, module):
        import torch

        return torch.optim.Adam(module.embedding_parameters(), lr=self.learning_rate)
