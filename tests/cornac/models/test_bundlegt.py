# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Small CPU checks of BundleGT."""

import numpy as np
import pytest
import scipy.sparse as sp

from cornac.models import BundleGT

from ._bundle_checks import check_bundle_model, toy_evaluation


def tiny(**kwargs):
    params = dict(emb_size=8, num_ui_layers=2, num_trans_layers=2, num_token=3,
                  num_user_token=4, num_epochs=4, batch_size=16, learning_rate=0.05,
                  eval_every=2, eval_k=5, device="cpu", seed=3)
    params.update(kwargs)
    return BundleGT(**params)


@pytest.fixture(scope="module")
def evaluation():
    pytest.importorskip("torch")
    return toy_evaluation()


def test_contract(evaluation):
    model = check_bundle_model(tiny, evaluation)
    assert model.U.shape[1] == 8


def test_token_sequences_pad_and_truncate():
    pytest.importorskip("torch")
    from cornac.models.bundlegt.bundlegt import token_sequences

    m = sp.csr_matrix(np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=float))
    tokens, mask = token_sequences(m, 2)
    np.testing.assert_array_equal(tokens, [[0, 1], [1, 3], [3, 3]])
    np.testing.assert_array_equal(mask, [[True, True], [True, False], [False, False]])


def test_set_attention_masks_padding():
    torch = pytest.importorskip("torch")
    from cornac.models.bundlegt.bundlegt import SetAttention

    layer = SetAttention(4)
    tokens = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    out = layer(tokens, mask)
    changed = tokens.clone()
    changed[0, 2] = 100.0  # padding must not matter
    np.testing.assert_allclose(layer(changed, mask).detach().numpy(), out.detach().numpy(),
                               rtol=1e-5)
    assert torch.all(out[1] == 0)  # empty set pools to zero


@pytest.mark.parametrize("official", [False, True])
def test_attention_training_switch(evaluation, official):
    import torch

    model = tiny(official_variant=official)
    model.rng = np.random.RandomState(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    before = module.bundle_attn[0].query.weight.detach().clone()
    optimizer = model._optimizer(module)
    users = torch.arange(6)
    loss, _ = module.loss(users, users % 4, (users + 1) % 4)
    loss.backward()
    optimizer.step()
    moved = not torch.equal(before, module.bundle_attn[0].query.weight)
    assert moved is (not official)


def test_paper_and_official_residuals_differ(evaluation):
    import torch

    def forward(official):
        model = tiny(official_variant=official)
        model.rng = np.random.RandomState(0)
        torch.manual_seed(0)
        module = model._build_module(evaluation.train_set, torch.device("cpu"))
        with torch.no_grad():
            return module.forward_all()

    paper_u, paper_b = forward(False)
    code_u, code_b = forward(True)
    # Same initialization, different residual sources (Eq. 9 and 14 vs released code).
    assert not torch.allclose(paper_b, code_b)
    assert not torch.allclose(paper_u, code_u)


def test_paper_bundle_residual_aggregates_lightgcn_items(evaluation):
    import torch

    model = tiny(num_trans_layers=1, alpha=0.0)
    model.rng = np.random.RandomState(0)
    torch.manual_seed(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    with torch.no_grad():
        _, b = module.forward_all()
        _, i_cf = module._ui_branch()
        nb = module.num_bundles
        b_u2b = torch.sparse.mm(module.ub_graph,
                                torch.cat([module.users, module.bundles]))[module.num_users:]
        bi = torch.sparse.mm(module.bi_graph, torch.cat([torch.zeros_like(module.bundles), i_cf]))
    # alpha=0: b = sum_i norm * i^cf + UB message (Eq. 9-10).
    np.testing.assert_allclose(b.numpy(), (bi[:nb] + b_u2b).numpy(), rtol=1e-4, atol=1e-6)
