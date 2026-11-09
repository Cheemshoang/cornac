# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Small CPU checks of BGCN."""

import numpy as np
import pytest
import scipy.sparse as sp

from cornac.models import BGCN

from ._bundle_checks import check_bundle_model, toy_evaluation


def tiny(**kwargs):
    params = dict(emb_size=8, num_layers=2, num_epochs=4, batch_size=16,
                  learning_rate=0.01, message_dropout=0.1, eval_every=2, eval_k=5,
                  device="cpu", seed=3)
    params.update(kwargs)
    return BGCN(**params)


@pytest.fixture(scope="module")
def evaluation():
    pytest.importorskip("torch")
    return toy_evaluation()


def test_contract(evaluation):
    model = check_bundle_model(tiny, evaluation)
    # Two levels, each concatenating layers 0..2 of size 8.
    assert model.U.shape[1] == 2 * 3 * 8


def test_bundle_overlap_is_cosine():
    pytest.importorskip("torch")
    from cornac.models.bgcn.bgcn import bundle_overlap

    bi = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=float))
    overlap = bundle_overlap(bi).toarray()
    np.testing.assert_allclose(np.diag(overlap), 1.0)
    np.testing.assert_allclose(overlap[0, 1], 0.5)
    assert overlap[0, 2] == 0


def test_hard_negative_stage_runs(evaluation):
    model = tiny(num_epochs=2, hard_neg_epochs=2, hard_neg_ratio=1.0, hard_neg_window=3)
    model.fit(evaluation.train_set, evaluation.val_set)
    assert model.num_epochs == 2  # restored after training
    assert [h[0] for h in model.val_history][-1] == 4


def test_level_ablation(evaluation):
    assert tiny(use_bundle_level=False).fit(evaluation.train_set).U.shape[1] == 24
    with pytest.raises(ValueError):
        BGCN(use_item_level=False, use_bundle_level=False)
