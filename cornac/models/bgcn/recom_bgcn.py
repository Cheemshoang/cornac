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

import numpy as np

from ..bundle_utils import BundleTorchRecommender
from ..bundle_utils.graph import binary, sample_negatives, user_item_matrix


class BGCN(BundleTorchRecommender):
    """BGCN: Bundle Recommendation with Graph Convolutional Networks.

    A heterogeneous user-item-bundle graph is split into two levels. The item
    level convolves the user-item graph (with self-loops) and represents a
    bundle by the mean of its items; the bundle level convolves the user-bundle
    graph extended with bundle-bundle edges weighted by item overlap. Each level
    uses graph convolutions with a linear transform and LeakyReLU, concatenating
    the L2-normalized layer outputs. The score sums the two levels' dot products
    and the model is trained with BPR plus L2 on the batch embeddings.

    An optional hard-negative stage (``hard_neg_epochs``) continues training
    with negatives drawn, with probability ``hard_neg_ratio``, half from the
    user's highest-scored unobserved bundles and half from bundles overlapping
    the positive bundle.

    Parameters
    ----------
    name: str, default: 'BGCN'

    emb_size: int, default: 64

    num_layers: int, default: 2

    num_epochs: int, default: 100
        Epochs of the uniform-negative stage.

    learning_rate: float, default: 3e-4

    batch_size: int, default: 2048

    lambda_reg: float, default: 1e-7
        L2 weight on the batch embeddings (no Adam weight decay).

    message_dropout: float, default: 0.3
        Dropout on propagated messages.

    node_dropout: float, default: 0.0
        Edge dropout of the user-item and user-bundle graphs (training only).

    hard_neg_epochs: int, default: 0
        Extra epochs of the hard-negative stage (0 disables it).

    hard_neg_ratio: float, default: 0.8
        Probability that a negative of the hard stage is a hard one.

    hard_neg_window: int, default: 50
        Number of highest-scored unobserved bundles per user used as hard negatives.

    use_item_level, use_bundle_level: bool, default: True
        Ablation switches.

    eval_every, patience, eval_k, device, trainable, verbose, seed:
        See :class:`cornac.models.bundle_utils.BundleTorchRecommender`.

    Notes
    -----
    Clean-room reimplementation from the paper; settings checked against
    https://github.com/cjx0525/BGCN (commit a81cbd8), which only ships a NetEase
    configuration. ``num_epochs`` counts the uniform stage; the total is
    ``num_epochs + hard_neg_epochs``.

    References
    ----------
    * Chang, J., Gao, C., He, X., Jin, D., & Li, Y. (2020). Bundle
      Recommendation with Graph Convolutional Networks. SIGIR 2020.
    """

    def __init__(
        self,
        name="BGCN",
        emb_size=64,
        num_layers=2,
        num_epochs=100,
        learning_rate=3e-4,
        batch_size=2048,
        lambda_reg=1e-7,
        message_dropout=0.3,
        node_dropout=0.0,
        hard_neg_epochs=0,
        hard_neg_ratio=0.8,
        hard_neg_window=50,
        use_item_level=True,
        use_bundle_level=True,
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
        if not (use_item_level or use_bundle_level):
            raise ValueError("at least one of use_item_level and use_bundle_level")
        if not 0 <= hard_neg_ratio <= 1:
            raise ValueError("hard_neg_ratio must be in [0, 1]")
        self.num_layers = num_layers
        self.message_dropout = message_dropout
        self.node_dropout = node_dropout
        self.hard_neg_epochs = hard_neg_epochs
        self.hard_neg_ratio = hard_neg_ratio
        self.hard_neg_window = hard_neg_window
        self.use_item_level = use_item_level
        self.use_bundle_level = use_bundle_level

    def fit(self, train_set, val_set=None):
        uniform_epochs = self.num_epochs
        self._hard_candidates = None
        self.num_epochs = uniform_epochs + self.hard_neg_epochs
        try:
            return super().fit(train_set, val_set)
        finally:
            self.num_epochs = uniform_epochs
            self._hard_candidates = None

    def _build_module(self, train_set, device):
        from .bgcn import BGCNModule

        self._uniform_epochs = self.num_epochs - self.hard_neg_epochs
        return BGCNModule(
            ui=user_item_matrix(train_set),
            ub=binary(train_set.csr_matrix),
            bi=binary(train_set.bundle_item_matrix),
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            message_dropout=self.message_dropout,
            node_dropout=self.node_dropout,
            lambda_reg=self.lambda_reg,
            use_item_level=self.use_item_level,
            use_bundle_level=self.use_bundle_level,
            rng=self.rng,
            device=device,
        )

    def _optimizer(self, module):
        import torch

        return torch.optim.Adam(module.parameters(), lr=self.learning_rate)

    def _hard_negative_candidates(self, positives):
        """Per-user top-``hard_neg_window`` unobserved bundles and bundle overlap lists."""
        import torch

        module = self._module
        module.eval()
        with torch.no_grad():
            users, bundles = module.embeddings()
        scores = (users @ bundles.T).cpu().numpy()
        scores[positives.nonzero()] = -np.inf
        window = min(self.hard_neg_window, scores.shape[1])
        top = np.argpartition(-scores, window - 1, axis=1)[:, :window]
        overlap = module.overlap.tolil()
        overlap.setdiag(0)
        neighbors = [np.asarray(row, dtype=np.int64) for row in overlap.rows]
        module.train()
        return top, neighbors

    def _sample_negatives(self, positives, users, pos_bundles, epoch):
        negatives = sample_negatives(positives, users, self.rng)
        if epoch < self._uniform_epochs or self.hard_neg_ratio == 0:
            return negatives
        if self._hard_candidates is None:
            self._hard_candidates = self._hard_negative_candidates(positives)
        top, neighbors = self._hard_candidates
        draws = self.rng.random_sample(len(users))
        half = self.hard_neg_ratio / 2
        for k, (user, pos, r) in enumerate(zip(users, pos_bundles, draws)):
            if r < half:
                negatives[k] = top[user, self.rng.randint(top.shape[1])]
            elif r < self.hard_neg_ratio and len(neighbors[pos]) > 0:
                candidate = neighbors[pos][self.rng.randint(len(neighbors[pos]))]
                if positives[user, candidate] == 0:
                    negatives[k] = candidate
        return negatives
