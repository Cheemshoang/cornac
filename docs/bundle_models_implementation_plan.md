# Implementation Plan: Bundle Recommendation Models

Adds BGCN, CrossCBR, MultiCBR, BunCa, BundleGT and DCBR to Cornac on the `BunRec` branch.

## Context
`BunRec` has the bundle infrastructure (`BundleDataset`, `BundleEvaluation`, `BundleRecommender`,
CrossCBR loaders) but only one bundle model, `BundlePop`. None of the six standard models exist.
The goal is Cornac-style, PyTorch ports of all six, each passing every gate in
[`new_model_reproducibility_checklist.md`](new_model_reproducibility_checklist.md).

Decisions:
- **Defaults follow the paper equations, with known bugs fixed.** Each model with a known
  code/paper mismatch gets an `official_variant=False` flag that reproduces the released code.
  This lets Rule 2 explain any gap to the published numbers.
- **DCBR is checked on iFashion only.** No MealRec+ loader. Youshu and NetEase are reported as new results.
- **Delivery is phased, one model per commit.** A shared layer comes first. Each model must clear
  the pre-merge gate before the next one starts.

Constraints:
- **None of the six official repos has a license.** All code is a clean-room reimplementation from
  the paper equations. Repos are only used to read hyperparameters and protocol details. Each model
  docstring cites the paper and names the repo and commit used as reference.
- **All six repos use the CrossCBR evaluation protocol:**
  - CrossCBR fixed splits
  - full-catalog ranking with only training positives masked → `BundleEvaluation.from_splits(..., test_mask="train")`
  - Recall@20 and NDCG@20
  - checkpoint selected on validation
  - Our loaders and evaluation already match this.

## Phase 0: Shared layer and fixes (commit 1)
- **New `cornac/models/bundle_utils/`.** Torch is imported only inside functions.
  `torch_utils.py` holds:
  - `sp_to_torch(csr)`
  - `bi_norm(A)`: D_r^-½ A D_c^-½ normalization
  - `row_norm(A)`: mean aggregation, e.g. a bundle as the mean of its items
  - `edge_dropout(A, p, rng)`
  - `lightgcn_propagate(G, emb, layers, layer_weights=None, normalize=False)`
  - `info_nce(a, b, temp)`: cosine similarity, in-batch negatives
  - `segment_softmax`: plain-torch GATv2 helper for BunCa, so no torch-geometric
- **`BundleTorchRecommender(BundleRecommender, ANNMixin)` in `bundle_utils/base.py`.** It covers the
  shared parts:
  - the `fit` skeleton: `BundleRecommender.fit` → lazy import torch → seeding
    (`torch.manual_seed`, `cuda.manual_seed_all`, `get_rng`) → epoch loop over
    `train_set.uij_iter` → `return self`
  - validation-only checkpoint selection every `eval_every` epochs. A checkpoint is kept when
    Recall@20 *and* NDCG@20 on `val_set` both improve, as in the original repos, using
    `ranking_eval` as `LightGCN.monitor_value` does. Patience is optional.
  - final user and bundle embeddings copied to numpy, so `score()` is pure numpy and pickling
    and `clone()` stay safe
  - a `device` init argument
  - `user_item_matrix=None` fallback: derive it from UB·BI, binarized, as BundlePop does with `_binary`
- **Fix `fit()` to `return self`** in `cornac/models/lightgcn/recom_lightgcn.py` and
  `cornac/models/ngcf/recom_ngcf.py`. Today `fit()` returns `None`, so `RandomSearch`
  (`cornac/hyperopt.py:147`) can't tune them, and they are needed as tuned baselines.
- **Tests:** `tests/cornac/models/test_bundle_utils.py`
  - check `bi_norm`, `info_nce` and `segment_softmax` against hand-computed numpy values
  - check that the lazy import works (subprocess test that blocks torch, pattern from `tests/cornac/models/test_diger.py`)
  - check LightGCN/NGCF `fit` returns self

## Phases 1–6: One model per commit, in this order

Every model uses the same layout, following `cornac/models/lightgcn/`:
`cornac/models/<m>/{__init__.py, recom_<m>.py, <m>.py (nn.Module), requirements.txt (torch>=1.12)}`.
Each model is also registered in four places:
- `cornac/models/__init__.py`, alphabetical, no try/except
- a `README.md` model-table row
- `docs/source/api_ref/models.rst`
- `examples/README.md` under "Bundle Algorithms"

| # | Model | Key components to implement | Default (paper) vs `official_variant=True` |
|---|---|---|---|
| 1 | **CrossCBR** (KDD'22) | LightGCN on UI and UB; bundle = BI-mean; edge dropout on 3 graphs; cross-view InfoNCE; BPR; score = sum of the two view dot products | Same in both. No known mismatch. Do this one first as the template |
| 2 | **BGCN** (SIGIR'20) | Item graph `[[I,UI],[UIᵀ,I]]` and bundle graph with BB overlap; Linear+LeakyReLU layers concatenated with L2 normalization; message dropout; optional hard-negative stage (`hard_neg_ratio`) | Same in both |
| 3 | **MultiCBR** (TOIS'24) | Three LightGCN views (UB/UI/BI) with fixed layer weights; noise augmentation; `modal_weight` fusion | Paper: InfoNCE between two augmented forward passes. Official: the code's self-contrast |
| 4 | **BunCa** (ECML-PKDD'24) | Item-item co-occurrence (thresholds `iui`/`ibi`) from **training** UI/BI only; GATv2 multi-head layer in plain torch; coherent view (UP+BC) and cohesive view (UU/BB); cross and native contrast | Paper: residual on and all contrastive terms on. Official: the released `run.sh` weights |
| 5 | **BundleGT** (SIGIR'23) | LightGCN on UI; transformer layers with masked attention over bundle→items (`num_token`) and user→bundles (`num_user_token`, as a parameter instead of hard-coded) | Paper: attention weights trained; bundle residual aggregates i^cf (Eq. 9); user residual aggregates b^(l) (Eq. 14); L2 on all parameters (Eq. 18). Official: attention untrained, both residuals from the previous layer, L2 on tables only |
| 6 | **DCBR** (AAAI'25) | MultiCBR UI and BI views; DiffRec-style MLP diffusion over UB rows (ELBO + λ0·BLCC) with its own optimizer; UB rebuilt with top-`rebuild_k` per user; inter- and intra-view InfoNCE; score from UI+BI only | Same in both. Device follows the `device` argument (the official code hard-codes `.cuda()`) |

Per-dataset default hyperparameters (Youshu, NetEase, iFashion) come from each repo's config, as
listed below. They go into the example scripts, not into class defaults.

## Meeting the checklist (every model)
- **Rule 1 (data):**
  - CrossCBR fixed splits via `bundle.load_*` and `BundleEvaluation.from_splits(test_mask="train")`.
  - Data-signal ablation flags (`use_item_view` / `use_bundle_view` or the equivalent) show that
    each view helps.
  - BunCa co-occurrence and DCBR diffusion use the training matrices only.
- **Rule 2 (implementation):**
  - Unit tests in `tests/cornac/models/test_<m>.py`, written with `pytest.importorskip("torch")`
    on tiny toy data based on `BUNDLE_CATALOG` in `tests/cornac/models/test_recommender.py`:
    - `fit` returns self, `score` shape, ScoreException on unknown users
    - save/load round-trip, `clone`
    - determinism with a fixed seed
    - the `official_variant` switch
  - Component audit table (architecture, loss, regularization, sampling, optimizer, inference),
    plus a reproduction against the paper numbers, in `docs/bundle_models_reproduction.md`.
    Targets (R@20 / N@20):
    - CrossCBR: Youshu 0.2813/0.1668, NetEase 0.0842/0.0457, iFashion 0.1173/0.0895
    - MultiCBR: .2842/.1693, .0907/.0494, .1500/.1203
    - BunCa: .2978/.1762, .0899/.0479, .1431/.1123
    - BGCN: Youshu .2347/.1345, NetEase .0491/.0258
    - DCBR: iFashion .1633/.1370
    - BundleGT: .2905/.1737, .0903/.0478, .1263/.0969 (paper Table 2)
    - Run both variants and explain any gap.
- **Rule 3 (tuning):** `RandomSearch` with the same `n_trails` and the same validation NDCG@20 for
  every trainable model: the new models plus the BPR, LightGCN, EASE and ItemKNN baselines.
  Search spaces are centred on the official configs and published.
- **Rule 4 (evaluation):** validation-only checkpoint selection, full catalog, 5 seeds, mean ± std.
- **Rule 5 (pipeline):**
  - One shared script, `examples/bundle_repro.py --model <M> --dataset <D> [--official] [--tune N --baselines]`,
    built from the checklist template. It holds every model's official configs, paper targets and
    search spaces, and saves the config, environment, per-seed results and logs.
  - One quick-start per model, `examples/bundle_<m>_youshu.py`.
  - Results go into `docs/bundle_models_reproduction.md` using the reporting template.

## Per-model reference sheet

| Model | Paper | Reference repo (no license) | Graphs | Scored views | Contrastive | Main loss |
|---|---|---|---|---|---|---|
| BGCN | Chang et al., SIGIR'20, [arXiv:2005.03475](https://arxiv.org/abs/2005.03475) | [cjx0525/BGCN](https://github.com/cjx0525/BGCN) @a81cbd8 | UI(+I), UB+BB | item + bundle | none | BPR, plus an optional hard-negative stage |
| CrossCBR | Ma et al., KDD'22, [arXiv:2206.00242](https://arxiv.org/abs/2206.00242) | [mysbupt/CrossCBR](https://github.com/mysbupt/CrossCBR) @bbe80df | UI, UB, BI-mean | item + bundle | cross-view InfoNCE | BPR |
| MultiCBR | Ma et al., TOIS'24, [arXiv:2311.16751](https://arxiv.org/abs/2311.16751) | [HappyPointer/MultiCBR](https://github.com/HappyPointer/MultiCBR) @744cec8 | UB, UI, BI | fused UB/UI/BI | InfoNCE (paper: two forward passes) | BPR |
| BunCa | Nguyen et al., ECML-PKDD'24, [arXiv:2408.08906](https://arxiv.org/abs/2408.08906) | [bt-nghia/BunCa](https://github.com/bt-nghia/BunCa) @dcff093 | UI, BI, UB, II co-occ., UU/BB | coherent + cohesive | cross + native | BPR |
| BundleGT | Wei et al., SIGIR'23, [DOI 10.1145/3539618.3591771](https://doi.org/10.1145/3539618.3591771) | [Xiaohao-Liu/BundleGT](https://github.com/Xiaohao-Liu/BundleGT) @d7779ec | UI, UB, BI + attention | transformer output | none | BPR (softplus) + L2 |
| DCBR | Li, AAAI'25, [DOI 10.1609/aaai.v39i11.33314](https://doi.org/10.1609/aaai.v39i11.33314) | [recomall/DCBR](https://github.com/recomall/DCBR) @502eadf | UI, BI, diffused UB | UI + BI | inter + intra-view | BPR + diffusion ELBO/BLCC |

### Official per-dataset configs (they go into the example scripts)

All six use embedding size 64, lr 1e-3 and batch 2048, except where a row says otherwise.

- **CrossCBR**, 1 layer:
  - Youshu: dropout .2/.2/.2, l2 1e-4, c_λ .04, τ .25, 100 epochs
  - NetEase: .1/.3/.1, 1e-4, .1, .25, 50 epochs
  - iFashion: .2/.2/0, 4e-5, .25, .2, 50 epochs
- **BGCN**, 2 layers:
  - NetEase: lr 3e-4, message dropout .3, decay 1e-7, early stop after 50 epochs
  - Youshu, iFashion: no official config, so tune them
- **MultiCBR**, 2 layers, l2 1e-5:
  - Youshu: eps .2/.1/.1, modal [.35,.15,.5], c_λ .04, τ .23, 100 epochs;
    layer weights UB [.7,.2,.1], UI [.6,.2,.2], BI [.6,.1,.3]
  - NetEase: eps 0/0/.2, modal [.5,.2,.3], c_λ .05, τ .2, 50 epochs;
    layer weights UB [.35,.15,.5], UI [.25,.65,.1], BI [.4,.4,.2]
  - iFashion: eps .05/0/.15, modal [.1,.2,.7], c_λ .1, τ .2, 50 epochs;
    layer weights UB [.5,.5,0], UI [.3,.2,.5], BI [.5,.5,0]
- **BunCa**, 1 layer:
  - Youshu: nhead 2, iui=ibi=4, contrast [.5,.5,0,0]
  - NetEase: nhead 2, iui 9, ibi 10, contrast [.3,.3,.2,.2], hyperth 2
  - iFashion: nhead 5, iui 15, ibi 2, batch 12000, hyperth 4
- **BundleGT**, 4 UI layers:
  - Youshu: 3 transformer layers, num_token 70, α .5, l2 1e-5
  - NetEase: 5, 70, .5, 1e-6
  - iFashion: 1, 5, .75, 2e-6, gcn_norm on
- **DCBR**, 2 layers, iFashion: steps 140, noise .1, λ0 2, ω .5, γ .3/.2, τ .08, λ1 .3, λ2 1e-6, 50 epochs

### Implementation tracker

| Phase | Item | Status |
|---|---|---|
| 0 | `bundle_utils` + LightGCN/NGCF `return self` fix + tests | ✅ code + tests (dgl-dependent LightGCN/NGCF checks skip without dgl) |
| 1 | CrossCBR | ✅ code + tests; **Youshu reproduced** (1 seed, 100 epochs, CPU 13 min): R@20 0.2792 / N@20 0.1655 vs paper 0.2813 / 0.1668 (−0.8% / −0.8%); NetEase, iFashion and 5-seed runs pending |
| 2 | BGCN | ✅ code + tests; Youshu smoke run OK (learns at lr 3e-3; official lr 3e-4 converges slowly, as the reference's 1000-epoch budget implies); full reproduction pending |
| 3 | MultiCBR | ✅ code + tests (paper and official variants); Youshu smoke run (5 epochs) OK for both; full reproduction pending |
| 4 | BunCa | ✅ code + tests (paper defaults; released switches via `bundle_repro.py --official`); Youshu smoke run (3 epochs, val R@20 0.233) OK; full reproduction pending |
| 5 | BundleGT | ✅ code + tests (trained vs frozen attention); Youshu smoke run (3 epochs, test R@20 0.223) OK, ~200 s/epoch on CPU so GPU needed; full reproduction pending |
| 6 | DCBR | ✅ code + tests; Youshu smoke run: learns without / with light contrastive loss (val R@20 0.093 / 0.076 at epoch 3) but the iFashion contrastive settings stall it on Youshu, so Youshu/NetEase need `--tune`; iFashion reproduction pending (GPU) |

A phase is ticked only when all 11 rows of the pre-merge gate pass and the user approves.

## Verification
- Run `pytest tests/cornac/models/test_bundle_utils.py tests/cornac/models/test_<m>.py tests/cornac/eval_methods/test_bundle_evaluation.py tests/cornac/models/test_recommender.py`.
  Install torch locally first. CI skips the torch tests.
- Run flake8 `--select=E9,F63,F7,F82`, the same check CI runs.
- Run the quick-start on Youshu (CPU is fine).
- Run the full repro on Kaggle GPU (iFashion and BundleGT on NetEase need large memory). Check
  each model against the targets above. Results must be within about 5% relative, or the gap
  must be explained.
- Fill in the pre-merge gate table for each model before committing.
