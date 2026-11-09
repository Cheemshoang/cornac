# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Checks of the shared bundle graph helpers and the PyTorch training skeleton."""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

from cornac.eval_methods import BundleEvaluation
from cornac.models.bundle_utils import (
    BundleTorchRecommender,
    bi_norm,
    bipartite,
    edge_dropout,
    row_norm,
    user_item_matrix,
)
from cornac.models.bundle_utils.graph import sample_negatives


def toy_bundle_data(seed=0, users=12, bundles=10, items=15):
    """Small random catalog and user-bundle splits for bundle model tests."""
    rng = np.random.RandomState(seed)
    catalog = []
    for b in range(bundles):
        for i in rng.choice(items, size=3, replace=False):
            catalog.append(("b%d" % b, "i%d" % i))
    pairs = sorted({(u, b) for u in range(users)
                    for b in rng.choice(bundles, size=4, replace=False)})
    rows = [("u%d" % u, "b%d" % b, 1.0) for u, b in pairs]
    rng.shuffle(rows)
    n = len(rows)
    train, val, test = rows[: n * 6 // 10], rows[n * 6 // 10: n * 8 // 10], rows[n * 8 // 10:]
    user_items = [("u%d" % u, "i%d" % rng.randint(items), 1.0) for u in range(users)
                  for _ in range(3)]
    return train, val, test, catalog, user_items


def toy_evaluation(seed=0, with_user_items=True):
    train, val, test, catalog, user_items = toy_bundle_data(seed)
    return BundleEvaluation.from_splits(
        train_data=train, val_data=val, test_data=test,
        bundle_item_data=catalog,
        user_item_data=user_items if with_user_items else None,
        test_mask="train", exclude_unknowns=True, seed=seed,
    )


# ---------------------------------------------------------------- graph helpers
def test_bi_norm_matches_formula():
    a = sp.csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    expected = np.diag(1 / np.sqrt([2, 2])) @ a.toarray() @ np.diag(1 / np.sqrt([1, 2, 1]))
    np.testing.assert_allclose(bi_norm(a).toarray(), expected, rtol=1e-6)


def test_row_norm_is_mean_aggregation():
    a = sp.csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(
        row_norm(a).toarray(), [[0.5, 0.5, 0], [0, 0, 1], [0, 0, 0]], atol=1e-6
    )


def test_bipartite_blocks():
    a = sp.csr_matrix(np.array([[1.0, 0.0, 2.0]]))
    full = bipartite(a, diag_rows=sp.identity(1)).toarray()
    assert full.shape == (4, 4)
    np.testing.assert_array_equal(full[0, 1:], [1, 0, 2])
    np.testing.assert_array_equal(full[1:, 0], [1, 0, 2])
    assert full[0, 0] == 1


def test_edge_dropout_keeps_subset():
    a = sp.random(50, 40, density=0.3, format="csr", random_state=1)
    a.data[:] = 1.0
    dropped = edge_dropout(a, 0.5, np.random.RandomState(0))
    assert 0 < dropped.nnz < a.nnz
    assert (dropped.multiply(a) != dropped).nnz == 0  # only existing edges survive
    assert edge_dropout(a, 0.0, np.random.RandomState(0)) is a


def test_sample_negatives_avoids_positives():
    positives = sp.csr_matrix(np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=float))
    users = np.array([0, 1] * 50)
    negatives = sample_negatives(positives, users, np.random.RandomState(0))
    assert set(negatives[users == 0]) == {3}
    assert set(negatives[users == 1]) == {0}


def test_user_item_fallback_uses_training_bundles():
    with_ui = toy_evaluation(with_user_items=True).train_set
    np.testing.assert_array_equal(
        user_item_matrix(with_ui).toarray(), (with_ui.user_item_matrix.toarray() != 0)
    )
    no_ui = toy_evaluation(with_user_items=False).train_set
    derived = (no_ui.csr_matrix @ no_ui.bundle_item_matrix).toarray() != 0
    np.testing.assert_array_equal(user_item_matrix(no_ui).toarray(), derived)


# ---------------------------------------------------------------- torch helpers
def test_info_nce_and_segment_softmax():
    torch = pytest.importorskip("torch")
    from cornac.models.bundle_utils.torch_utils import (
        bpr_loss, info_nce, lightgcn_propagate, segment_softmax, sp_to_torch,
    )

    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    b = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
    an = a.numpy() / np.linalg.norm(a.numpy(), axis=1, keepdims=True)
    bn = b.numpy() / np.linalg.norm(b.numpy(), axis=1, keepdims=True)
    logits = an @ bn.T / 0.5
    expected = np.mean(np.log(np.exp(logits).sum(1)) - np.diag(logits))
    assert abs(info_nce(a, b, 0.5).item() - expected) < 1e-5

    scores = torch.tensor([[1.0], [2.0], [3.0]])
    out = segment_softmax(scores, torch.tensor([0, 0, 1]), 2).squeeze(1).numpy()
    np.testing.assert_allclose(out, [np.exp(1) / (np.exp(1) + np.exp(2)),
                                     np.exp(2) / (np.exp(1) + np.exp(2)), 1.0], rtol=1e-5)

    pos, neg = torch.tensor([2.0]), torch.tensor([0.0])
    assert abs(bpr_loss(pos, neg).item() - np.log1p(np.exp(-2.0))) < 1e-6

    graph = sp_to_torch(sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]])))
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = lightgcn_propagate(graph, x, 1, layer_weights=[0.5, 0.5])
    np.testing.assert_allclose(out.numpy(), [[0.5, 0.5], [0.5, 0.5]])


# ---------------------------------------------------------------- training skeleton
class ToyMF(BundleTorchRecommender):
    """Plain MF bundle model used to test the base class."""

    def __init__(self, name="ToyMF", emb_size=8, num_epochs=6, learning_rate=0.05,
                 batch_size=16, lambda_reg=0.0, eval_every=2, patience=None,
                 eval_k=5, device="cpu", trainable=True, verbose=False, seed=1):
        super().__init__(name=name, emb_size=emb_size, num_epochs=num_epochs,
                         learning_rate=learning_rate, batch_size=batch_size,
                         lambda_reg=lambda_reg, eval_every=eval_every,
                         patience=patience, eval_k=eval_k, device=device,
                         trainable=trainable, verbose=verbose, seed=seed)

    def _build_module(self, train_set, device):
        import torch
        from cornac.models.bundle_utils.torch_utils import bpr_loss

        emb_size = self.emb_size

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.u = torch.nn.Embedding(train_set.num_users, emb_size)
                self.b = torch.nn.Embedding(train_set.num_items, emb_size)

            def loss(self, users, pos, neg):
                u = self.u(users)
                value = bpr_loss((u * self.b(pos)).sum(1), (u * self.b(neg)).sum(1))
                return value, {"bpr": value.item()}

            def embeddings(self):
                return self.u.weight, self.b.weight

        return Module()


@pytest.fixture(scope="module")
def evaluation():
    return toy_evaluation()


def test_fit_returns_self_and_selects_on_validation(evaluation):
    pytest.importorskip("torch")
    model = ToyMF()
    assert model.fit(evaluation.train_set, evaluation.val_set) is model
    assert model.U.shape == (evaluation.train_set.num_users, 8)
    assert model.V.shape == (evaluation.train_set.num_items, 8)
    assert [h[0] for h in model.val_history] == [2, 4, 6]
    assert model.best_epoch in (2, 4, 6)
    scores = model.score(0)
    assert scores.shape == (evaluation.train_set.num_items,)
    assert abs(model.score(0, 1) - scores[1]) < 1e-5


def test_seed_determinism_save_load_clone(evaluation):
    pytest.importorskip("torch")
    a = ToyMF().fit(evaluation.train_set, evaluation.val_set)
    b = ToyMF().fit(evaluation.train_set, evaluation.val_set)
    np.testing.assert_allclose(a.U, b.U)
    with tempfile.TemporaryDirectory() as path:
        loaded = ToyMF.load(a.save(path))
        np.testing.assert_allclose(loaded.score(0), a.score(0))
    clone = a.clone({"emb_size": 4})
    assert clone.emb_size == 4 and clone.seed == a.seed


def test_unknown_user_raises(evaluation):
    pytest.importorskip("torch")
    from cornac.exception import ScoreException

    model = ToyMF(num_epochs=1).fit(evaluation.train_set)
    with pytest.raises(ScoreException):
        model.score(10 ** 6)


def test_untrainable_fit_skips_torch(evaluation):
    model = ToyMF(trainable=False)
    assert model.fit(evaluation.train_set) is model
    assert not hasattr(model, "U")


def test_bundle_utils_import_does_not_load_torch():
    script = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] == 'torch':
            raise ImportError('blocked')
sys.meta_path.insert(0, Block())
import cornac.models.bundle_utils
from cornac.models import BundleRecommender
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


@pytest.mark.parametrize("name", ["LightGCN", "NGCF"])
def test_graph_baselines_fit_returns_self(name):
    pytest.importorskip("torch")
    pytest.importorskip("dgl")
    import cornac
    from cornac.data import Dataset

    data = Dataset.from_uir([("u1", "i1", 1.0), ("u1", "i2", 1.0), ("u2", "i2", 1.0)])
    model = getattr(cornac.models, name)(emb_size=4, num_epochs=1, batch_size=2)
    assert model.fit(data) is model
