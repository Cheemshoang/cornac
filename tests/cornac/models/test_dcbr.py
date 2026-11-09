# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Small CPU checks of DCBR."""

import numpy as np
import pytest

from cornac.models import DCBR

from ._bundle_checks import check_bundle_model, toy_evaluation


def tiny(**kwargs):
    params = dict(emb_size=8, num_layers=1, ub_layer_weights=(0.5, 0.5),
                  ui_layer_weights=(0.5, 0.5), bi_layer_weights=(0.5, 0.5),
                  diff_steps=3, diff_dims=(16,), diff_batch_size=5, num_epochs=4,
                  batch_size=16, learning_rate=0.05, eval_every=2, eval_k=5,
                  device="cpu", seed=3)
    params.update(kwargs)
    return DCBR(**params)


@pytest.fixture(scope="module")
def evaluation():
    pytest.importorskip("torch")
    return toy_evaluation()


def test_contract(evaluation):
    model = check_bundle_model(tiny, evaluation)
    assert model.U.shape[1] == 8  # UI and BI views fused


def test_diffusion_rebuilds_top_k_graph(evaluation):
    import torch

    model = tiny(rebuild_k=2)
    model.rng = np.random.RandomState(0)
    torch.manual_seed(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    before = module.ub_graph.to_dense().clone()
    module.epoch_start(0)
    after = module.ub_graph.to_dense()
    nu = module.num_users
    links = (after[:nu, nu:] > 0).sum(dim=1)
    assert torch.all(links == 2)  # exactly rebuild_k bundles per user
    assert not torch.equal(before, after)
    assert torch.all(torch.diagonal(after) > 0)  # self-loops


def test_diffusion_schedule_and_denoise_shapes():
    torch = pytest.importorskip("torch")
    from cornac.models.dcbr.dcbr import Diffusion

    diffusion = Diffusion(num_bundles=6, steps=4, noise_scale=0.1, noise_min=1e-4,
                          noise_max=0.02, hidden_dims=(8,), time_emb_size=5, dropout=0.0)
    assert torch.all(diffusion.sqrt_alpha_bar[1:] <= diffusion.sqrt_alpha_bar[:-1])
    x0 = torch.zeros(3, 6)
    x0[0, 1] = 1.0
    loss, x0_hat = diffusion.training_loss(x0)
    assert x0_hat.shape == (3, 6) and torch.isfinite(loss)
    assert diffusion.denoise(x0).shape == (3, 6)


def test_diffusion_not_in_main_optimizer(evaluation):
    import torch

    model = tiny()
    model.rng = np.random.RandomState(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    main = {id(p) for group in model._optimizer(module).param_groups for p in group["params"]}
    assert not main & {id(p) for p in module.diffusion.parameters()}


def test_argument_validation():
    with pytest.raises(ValueError):
        DCBR(num_layers=1)
    with pytest.raises(ValueError):
        DCBR(rebuild_k=0)
