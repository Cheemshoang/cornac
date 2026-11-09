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
"""Shared building blocks for graph-based bundle recommenders.

Graph helpers work on SciPy matrices; PyTorch is imported only inside the
functions and methods that need it, so importing this package stays light.
"""

from .base import BundleTorchRecommender
from .graph import bi_norm, bipartite, edge_dropout, row_norm, user_item_matrix
