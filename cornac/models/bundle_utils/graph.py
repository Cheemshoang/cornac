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
"""SciPy graph helpers shared by bundle recommenders (no PyTorch needed)."""

import numpy as np
import scipy.sparse as sp


def binary(matrix):
    """Binary float64 CSR copy of a sparse matrix (None stays None)."""
    if matrix is None:
        return None
    out = (sp.csr_matrix(matrix) != 0).astype(np.float64).tocsr()
    out.eliminate_zeros()
    return out


def user_item_matrix(train_set):
    """Training user x constituent item matrix of a :obj:`BundleDataset`.

    Uses the supplied ``user_item_matrix`` when available, otherwise derives
    it from the user's training bundles (UB @ BI, binarized). Only training
    observations are used, so no validation or test signal can leak.
    """
    if train_set.user_item_matrix is not None:
        return binary(train_set.user_item_matrix)
    return binary(binary(train_set.csr_matrix) @ binary(train_set.bundle_item_matrix))


def bipartite(matrix, diag_rows=None, diag_cols=None):
    """Symmetric block matrix [[R, A], [A^T, C]] of an m x n matrix A.

    ``diag_rows`` (m x m) and ``diag_cols`` (n x n) fill the diagonal blocks,
    zero when None.
    """
    m, n = matrix.shape
    top_left = sp.csr_matrix((m, m)) if diag_rows is None else diag_rows
    bottom_right = sp.csr_matrix((n, n)) if diag_cols is None else diag_cols
    return sp.bmat([[top_left, matrix], [matrix.T, bottom_right]], format="csr")


def bi_norm(matrix, eps=1e-8):
    """Symmetric normalization D_r^-1/2 A D_c^-1/2 (rows/columns by degree)."""
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    rows = np.asarray(matrix.sum(axis=1)).ravel()
    cols = np.asarray(matrix.sum(axis=0)).ravel()
    left = sp.diags(1.0 / (np.sqrt(rows) + eps))
    right = sp.diags(1.0 / (np.sqrt(cols) + eps))
    return (left @ matrix @ right).tocsr()


def row_norm(matrix, eps=1e-8):
    """Row normalization D_r^-1 A, i.e. mean aggregation over each row's neighbors."""
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    rows = np.asarray(matrix.sum(axis=1)).ravel()
    return (sp.diags(1.0 / (rows + eps)) @ matrix).tocsr()


def edge_dropout(matrix, ratio, rng):
    """Drop each stored edge independently with probability ``ratio``.

    Kept edges keep their value (no rescaling); callers re-normalize the graph
    afterwards. ``ratio`` 0 returns the matrix unchanged.
    """
    if ratio <= 0:
        return matrix
    coo = sp.coo_matrix(matrix)
    keep = rng.random_sample(coo.nnz) >= ratio
    return sp.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=coo.shape
    )


def sample_negatives(positives, users, rng, max_tries=100):
    """Uniform negatives: one column of ``positives`` not observed for each user.

    Vectorized rejection sampling against the CSR rows of ``positives``.
    """
    num_cols = positives.shape[1]
    keys = np.asarray(positives.indices, dtype=np.int64) + np.repeat(
        np.arange(positives.shape[0], dtype=np.int64) * num_cols,
        np.diff(positives.indptr),
    )
    keys.sort()
    users = np.asarray(users, dtype=np.int64)
    negatives = rng.randint(num_cols, size=len(users))
    for _ in range(max_tries):
        hit = np.isin(users * num_cols + negatives, keys, assume_unique=False)
        if not hit.any():
            break
        negatives[hit] = rng.randint(num_cols, size=int(hit.sum()))
    return negatives
