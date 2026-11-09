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
"""Example for MultiCBR on the Youshu bundle dataset (CrossCBR fixed splits)."""

import cornac
from cornac.datasets import bundle
from cornac.eval_methods import BundleEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import BundlePop, MultiCBR

bundle_eval = BundleEvaluation.from_splits(
    train_data=bundle.load_feedback("Youshu", split="train"),
    val_data=bundle.load_feedback("Youshu", split="validation"),
    test_data=bundle.load_feedback("Youshu", split="test"),
    bundle_item_data=bundle.load_bundle_item("Youshu"),
    user_item_data=bundle.load_user_item("Youshu"),
    test_mask="train",
    verbose=True,
    **bundle.load_ids("Youshu"),
)

# Official Youshu settings of the reference implementation. By default the
# contrastive loss follows the paper (two perturbed passes); official_variant=True
# reproduces the released code's self-contrast.
multicbr = MultiCBR(
    emb_size=64, num_layers=2, num_epochs=100, learning_rate=1e-3, batch_size=2048,
    lambda_reg=1e-5, view_eps=(0.2, 0.1, 0.1), modal_weights=(0.35, 0.15, 0.5),
    ub_layer_weights=(0.7, 0.2, 0.1), ui_layer_weights=(0.6, 0.2, 0.2),
    bi_layer_weights=(0.6, 0.1, 0.3), c_lambda=0.04, c_temp=0.23, eval_every=5,
    seed=2026, verbose=True,
)

cornac.Experiment(
    eval_method=bundle_eval,
    models=[BundlePop(popularity="user"), multicbr],
    metrics=[Recall(k=20), NDCG(k=20)],
).run()
