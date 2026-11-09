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
"""Example for BGCN on the NetEase bundle dataset (CrossCBR fixed splits)."""

import cornac
from cornac.datasets import bundle
from cornac.eval_methods import BundleEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import BGCN, BundlePop

bundle_eval = BundleEvaluation.from_splits(
    train_data=bundle.load_feedback("NetEase", split="train"),
    val_data=bundle.load_feedback("NetEase", split="validation"),
    test_data=bundle.load_feedback("NetEase", split="test"),
    bundle_item_data=bundle.load_bundle_item("NetEase"),
    user_item_data=bundle.load_user_item("NetEase"),
    test_mask="train",
    verbose=True,
    **bundle.load_ids("NetEase"),
)

# Official NetEase settings of the reference implementation (the only dataset it
# configures). Training stops after 10 validations (50 epochs) without
# improvement. Set hard_neg_epochs > 0 to add the hard-negative stage.
bgcn = BGCN(
    emb_size=64, num_layers=2, num_epochs=1000, learning_rate=3e-4, batch_size=2048,
    lambda_reg=1e-7, message_dropout=0.3, node_dropout=0.0, eval_every=5, patience=10,
    seed=2026, verbose=True,
)

cornac.Experiment(
    eval_method=bundle_eval,
    models=[BundlePop(popularity="user"), bgcn],
    metrics=[Recall(k=20), NDCG(k=20)],
).run()
