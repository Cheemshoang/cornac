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
"""Example for DCBR on the iFashion bundle dataset (CrossCBR fixed splits).

iFashion is the CrossCBR dataset DCBR reports on; it is large, so a GPU is
recommended (the diffusion model rebuilds the user-bundle graph every epoch).
"""

import cornac
from cornac.datasets import bundle
from cornac.eval_methods import BundleEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import BundlePop, DCBR

bundle_eval = BundleEvaluation.from_splits(
    train_data=bundle.load_feedback("iFashion", split="train"),
    val_data=bundle.load_feedback("iFashion", split="validation"),
    test_data=bundle.load_feedback("iFashion", split="test"),
    bundle_item_data=bundle.load_bundle_item("iFashion"),
    user_item_data=bundle.load_user_item("iFashion"),
    test_mask="train",
    verbose=True,
    **bundle.load_ids("iFashion"),
)

# Official iFashion settings of the reference implementation (the class defaults).
dcbr = DCBR(num_epochs=50, eval_every=5, seed=2026, verbose=True)

cornac.Experiment(
    eval_method=bundle_eval,
    models=[BundlePop(popularity="user"), dcbr],
    metrics=[Recall(k=20), NDCG(k=20)],
).run()
