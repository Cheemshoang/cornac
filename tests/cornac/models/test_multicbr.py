# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Small CPU checks of MultiCBR."""

import numpy as np
import pytest

from cornac.models import MultiCBR

from ._bundle_checks import check_bundle_model, toy_evaluation


def tiny(**kwargs):
    params = dict(emb_size=8, num_layers=1, ub_layer_weights=(0.5, 0.5),
                  ui_layer_weights=(0.5, 0.5), bi_layer_weights=(0.5, 0.5),
                  num_epochs=4, batch_size=16, learning_rate=0.05, eval_every=2,
                  eval_k=5, device="cpu", seed=3)
    params.update(kwargs)
    return MultiCBR(**params)


@pytest.fixture(scope="module")
def evaluation():
    pytest.importorskip("torch")
    return toy_evaluation()


def test_contract(evaluation):
    model = check_bundle_model(tiny, evaluation)
    assert model.U.shape[1] == 8  # early fusion keeps the embedding size


@pytest.mark.parametrize("official", [False, True])
def test_variants_train(evaluation, official):
    model = tiny(official_variant=official).fit(evaluation.train_set, evaluation.val_set)
    assert np.isfinite(model.U).all()


def test_paper_variant_uses_two_perturbed_passes(evaluation):
    import torch

    def contrastive(official):
        model = tiny(official_variant=official, view_eps=(0.5, 0.5, 0.5))
        model.rng = np.random.RandomState(0)
        torch.manual_seed(0)
        module = model._build_module(evaluation.train_set, torch.device("cpu"))
        users = torch.arange(6)
        _, parts = module.loss(users, users % 4, (users + 1) % 4)
        return parts["contrastive"]

    # Self-contrast is a pure uniformity term; two noisy passes differ from it.
    assert contrastive(True) != pytest.approx(contrastive(False))


def test_noise_only_in_training(evaluation):
    import torch

    model = tiny(view_eps=(0.5, 0.5, 0.5))
    model.rng = np.random.RandomState(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    with torch.no_grad():
        a, _ = module.embeddings()
        b, _ = module.embeddings()
        c, _ = module.fused(noise=True)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_argument_validation():
    with pytest.raises(ValueError):
        MultiCBR(modal_weights=(1, 1))
    with pytest.raises(ValueError):
        MultiCBR(num_layers=1)  # default layer weights have 3 values
