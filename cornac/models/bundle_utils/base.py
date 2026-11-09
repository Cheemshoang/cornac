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
"""Training skeleton shared by PyTorch bundle recommenders."""

import numpy as np
from tqdm.auto import trange

from ..recommender import ANNMixin, MEASURE_DOT, BundleRecommender
from ...exception import ScoreException
from ...utils import get_rng
from .graph import binary, sample_negatives


class BundleTorchRecommender(BundleRecommender, ANNMixin):
    """Base class for PyTorch bundle recommenders trained with BPR.

    Subclasses implement :meth:`_build_module`, returning a ``torch.nn.Module``
    with these methods:

    - ``loss(users, pos_bundles, neg_bundles)`` -> ``(loss, {name: float})``
    - ``embeddings()`` -> ``(user_matrix, bundle_matrix)`` whose dot product is
      the score (multi-view models concatenate their views)
    - optionally ``epoch_start(epoch)``, called before each training epoch
      (e.g. to resample edge dropout or train an auxiliary module)

    The base class owns seeding, negative sampling, the epoch loop and
    validation-only checkpoint selection. Final embeddings are copied to NumPy,
    so prediction, pickling and :meth:`clone` never need PyTorch.

    Parameters
    ----------
    name: str
        Model name.

    emb_size: int, default: 64
        Embedding size.

    num_epochs: int, default: 100
        Maximum number of training epochs.

    learning_rate: float, default: 1e-3
        Adam learning rate.

    batch_size: int, default: 2048
        Number of (user, positive bundle, negative bundle) triples per step.

    lambda_reg: float, default: 1e-4
        L2 regularization (Adam weight decay unless the model says otherwise).

    eval_every: int, default: 5
        Evaluate on ``val_set`` every ``eval_every`` epochs (and after the last).

    patience: int, optional, default: None
        Stop after this many evaluations without improvement. None trains
        for ``num_epochs``.

    eval_k: int, default: 20
        Cutoff of the Recall and NDCG used for checkpoint selection. A
        checkpoint is kept only when both improve on the validation set, as
        in the reference implementations. The test set is never used.

    device: str, optional, default: None
        Torch device; None picks CUDA when available, else CPU.

    trainable: bool, default: True
        When False, ``fit`` only stores the training data.

    verbose: bool, default: False
        Show progress bars and validation scores.

    seed: int, optional, default: None
        Random seed for initialization, negative sampling and augmentation.
    """

    def __init__(
        self,
        name,
        emb_size=64,
        num_epochs=100,
        learning_rate=1e-3,
        batch_size=2048,
        lambda_reg=1e-4,
        eval_every=5,
        patience=None,
        eval_k=20,
        device=None,
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.emb_size = emb_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.eval_every = eval_every
        self.patience = patience
        self.eval_k = eval_k
        self.device = device
        self.seed = seed

    # ------------------------------------------------------------------ hooks
    def _build_module(self, train_set, device):
        """Return the torch module trained by :meth:`fit`."""
        raise NotImplementedError()

    def _sample_negatives(self, positives, users, pos_bundles, epoch):
        """Negative bundles of a batch; uniform over unobserved bundles by default."""
        return sample_negatives(positives, users, self.rng)

    def _optimizer(self, module):
        """Optimizer of the module's parameters."""
        import torch

        return torch.optim.Adam(
            module.parameters(), lr=self.learning_rate, weight_decay=self.lambda_reg
        )

    # ------------------------------------------------------------------ training
    def _torch_device(self):
        import torch

        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _store_embeddings(self, module):
        import torch

        module.eval()
        with torch.no_grad():
            users, bundles = module.embeddings()
        self.U = users.detach().cpu().numpy().astype(np.float32)
        self.V = bundles.detach().cpu().numpy().astype(np.float32)

    def _validate(self, train_set, val_set):
        from ...eval_methods import ranking_eval
        from ...metrics import NDCG, Recall

        recall, ndcg = ranking_eval(
            model=self,
            metrics=[Recall(k=self.eval_k), NDCG(k=self.eval_k)],
            train_set=train_set,
            test_set=val_set,
        )[0]
        return recall, ndcg

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.BundleDataset`, required
            Training user-bundle feedback with bundle-item membership.

        val_set: :obj:`cornac.data.BundleDataset`, optional, default: None
            Validation feedback, used only for checkpoint selection.

        Returns
        -------
        self : object
        """
        BundleRecommender.fit(self, train_set, val_set)
        if not self.trainable:
            return self

        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        self.rng = get_rng(self.seed)
        device = self._torch_device()

        module = self._build_module(train_set, device).to(device)
        optimizer = self._optimizer(module)
        self._module = module  # available to hooks during training only
        try:
            return self._train(module, optimizer, train_set, val_set, device)
        finally:
            del self._module

    def _train(self, module, optimizer, train_set, val_set, device):
        import torch

        positives = binary(train_set.csr_matrix)
        pair_users, pair_bundles = positives.nonzero()
        num_pairs = len(pair_users)

        self.best_epoch = None
        self.val_history = []
        best = (-np.inf, -np.inf)
        best_embeddings = None
        stale = 0

        progress = trange(self.num_epochs, desc="Training", disable=not self.verbose)
        for epoch in progress:
            module.train()
            if hasattr(module, "epoch_start"):
                module.epoch_start(epoch)
            order = self.rng.permutation(num_pairs)
            total = 0.0
            for start in range(0, num_pairs, self.batch_size):
                batch = order[start : start + self.batch_size]
                users = pair_users[batch]
                negatives = self._sample_negatives(
                    positives, users, pair_bundles[batch], epoch
                )
                loss, _ = module.loss(
                    torch.as_tensor(users, dtype=torch.long, device=device),
                    torch.as_tensor(pair_bundles[batch], dtype=torch.long, device=device),
                    torch.as_tensor(negatives, dtype=torch.long, device=device),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            progress.set_postfix(loss=total / max(num_pairs, 1))

            last = epoch + 1 == self.num_epochs
            if val_set is None or not ((epoch + 1) % self.eval_every == 0 or last):
                continue
            self._store_embeddings(module)
            recall, ndcg = self._validate(train_set, val_set)
            self.val_history.append((epoch + 1, recall, ndcg))
            if self.verbose:
                print(
                    "epoch {}: val Recall@{k}={:.4f} NDCG@{k}={:.4f}".format(
                        epoch + 1, recall, ndcg, k=self.eval_k
                    )
                )
            if recall > best[0] and ndcg > best[1]:
                best = (recall, ndcg)
                best_embeddings = (self.U, self.V)
                self.best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
                if self.patience is not None and stale >= self.patience:
                    break

        if best_embeddings is not None:
            self.U, self.V = best_embeddings
        else:
            self._store_embeddings(module)
        return self

    # ------------------------------------------------------------------ prediction
    def score(self, user_idx, item_idx=None):
        """Predict the scores of a user for a bundle or for all bundles.

        Parameters
        ----------
        user_idx: int, required
            The index of the user.

        item_idx: int, optional, default: None
            The index of the bundle. If None, scores of all known bundles are returned.

        Returns
        -------
        res : A scalar or a Numpy array
        """
        if not self.knows_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)
        if item_idx is None:
            return self.V.dot(self.U[user_idx])
        if not self.knows_item(item_idx):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d, item_id=%d)"
                % (user_idx, item_idx)
            )
        return self.V[item_idx].dot(self.U[user_idx])

    def get_vector_measure(self):
        """Score is the dot product of user and bundle vectors."""
        return MEASURE_DOT

    def get_user_vectors(self):
        """User vectors (concatenated over views for multi-view models)."""
        return self.U

    def get_item_vectors(self):
        """Bundle vectors (concatenated over views for multi-view models)."""
        return self.V
