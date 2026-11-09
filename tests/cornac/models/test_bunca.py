# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Small CPU checks of BunCa."""

import numpy as np
import pytest
import scipy.sparse as sp

from cornac.models import BunCa

from ._bundle_checks import check_bundle_model, toy_evaluation


def tiny(**kwargs):
    params = dict(emb_size=8, num_layers=1, num_epochs=4, batch_size=16,
                  learning_rate=0.05, iui_threshold=1, ibi_threshold=1,
                  hyper_threshold=1, eval_every=2, eval_k=5, device="cpu", seed=3)
    params.update(kwargs)
    return BunCa(**params)


@pytest.fixture(scope="module")
def evaluation():
    pytest.importorskip("torch")
    return toy_evaluation()


def test_contract(evaluation):
    model = check_bundle_model(tiny, evaluation)
    assert model.U.shape[1] == 16  # coherent and cohesive views


def test_cooccurrence_and_shared_neighbors():
    pytest.importorskip("torch")
    from cornac.models.bunca.bunca import cooccurrence, shared_neighbors

    m = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float))
    np.testing.assert_array_equal(
        cooccurrence(m, 2).toarray(), [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(
        shared_neighbors(m, 1).toarray(), [[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def test_attention_matches_manual_gatv2():
    torch = pytest.importorskip("torch")
    from cornac.models.bunca.bunca import AsymmetricAttention

    torch.manual_seed(0)
    layer = AsymmetricAttention(emb_size=3, num_heads=1, dropout=0.0).eval()
    x = torch.randn(3, 3)
    src, dst = torch.tensor([1, 2]), torch.tensor([0, 0])  # 1->0, 2->0
    out = layer(x, src, dst)
    with torch.no_grad():
        xs, xt = layer.source(x), layer.target(x)
        e = [(torch.nn.functional.leaky_relu(xt[0] + xs[j], 0.2) * layer.att[0]).sum()
             for j in (1, 2)]
        a = torch.softmax(torch.stack(e), 0)
        expected0 = (x[0] + a[0] * xs[1] + a[1] * xs[2]) / 2
    np.testing.assert_allclose(out[0].detach().numpy(), expected0.numpy(), rtol=1e-5)
    np.testing.assert_allclose(out[1].detach().numpy(), (x[1] / 2).numpy(), rtol=1e-5)


def test_released_config_switches(evaluation):
    # Released Youshu setting: no residual, cross-view terms only, no UU/BB edges.
    model = tiny(residual_weight=0.0, contrast_weights=(0.5, 0.5, 0, 0),
                 hyper_threshold=10000)
    assert model.fit(evaluation.train_set, evaluation.val_set) is model
    with pytest.raises(ValueError):
        BunCa(contrast_weights=(1, 1))
