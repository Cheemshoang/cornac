# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Small CPU checks of CrossCBR."""

import numpy as np
import pytest

from cornac.models import CrossCBR

from ._bundle_checks import check_bundle_model, toy_evaluation


def tiny(**kwargs):
    params = dict(emb_size=8, num_layers=1, num_epochs=4, batch_size=16,
                  learning_rate=0.05, eval_every=2, eval_k=5, device="cpu", seed=3)
    params.update(kwargs)
    return CrossCBR(**params)


@pytest.fixture(scope="module")
def evaluation():
    pytest.importorskip("torch")
    return toy_evaluation()


def test_contract(evaluation):
    model = check_bundle_model(tiny, evaluation)
    # Two views are concatenated, so the score is the sum of view dot products.
    assert model.U.shape[1] == 16


def test_single_view_ablation(evaluation):
    item_only = tiny(use_bundle_view=False).fit(evaluation.train_set)
    bundle_only = tiny(use_item_view=False).fit(evaluation.train_set)
    assert item_only.U.shape[1] == 8 and bundle_only.U.shape[1] == 8
    with pytest.raises(ValueError):
        CrossCBR(use_item_view=False, use_bundle_view=False)


def test_item_view_bundle_is_mean_of_items(evaluation):
    import torch
    from cornac.models.bundle_utils.graph import binary, user_item_matrix
    from cornac.models.crosscbr.crosscbr import CrossCBRModule

    train_set = evaluation.train_set
    module = CrossCBRModule(
        ui=user_item_matrix(train_set), ub=binary(train_set.csr_matrix),
        bi=binary(train_set.bundle_item_matrix), emb_size=4, num_layers=1,
        item_dropout=0, bundle_dropout=0, agg_dropout=0, c_lambda=0.1, c_temp=0.2,
        use_item_view=True, use_bundle_view=True,
        rng=np.random.RandomState(0), device="cpu",
    )
    with torch.no_grad():
        views = module.propagate(module.clean)
        nu = module.num_users
        from cornac.models.bundle_utils.torch_utils import lightgcn_propagate
        feats = lightgcn_propagate(module.clean[0], torch.cat([module.users, module.items]),
                                   1, normalize=True)
        items = feats[nu:].numpy()
    bi = binary(train_set.bundle_item_matrix).toarray()
    expected = (bi / bi.sum(1, keepdims=True)) @ items
    np.testing.assert_allclose(views["item"][1].numpy(), expected, rtol=1e-4, atol=1e-6)


def test_contrastive_loss_is_used(evaluation):
    import torch

    model = tiny()
    model.rng = np.random.RandomState(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    users = torch.tensor([0, 1, 2])
    pos = torch.tensor([0, 1, 2])
    neg = torch.tensor([3, 4, 5])
    _, parts = module.loss(users, pos, neg)
    assert set(parts) == {"bpr", "contrastive"}
