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
"""Reproducibility runs of the graph bundle recommenders.

Follows docs/new_model_reproducibility_checklist.md:

- Rule 1: CrossCBR fixed author splits, training-time user-item feedback only.
- Rule 3: ``--tune N`` gives every trainable model (bundle model and baselines)
  the same RandomSearch budget, selected on validation NDCG@20.
- Rule 4: checkpoints and hyperparameters are selected on validation only; the
  full bundle catalog is ranked; several seeds are reported as mean +- std.
- Rule 5: config, environment, per-seed and aggregate results are saved as JSON.

Examples
--------
Reproduce CrossCBR on Youshu with the official configuration and 5 seeds::

    python examples/bundle_repro.py --model CrossCBR --dataset Youshu

Run the released-code variant of a model with a known code/paper mismatch::

    python examples/bundle_repro.py --model MultiCBR --dataset NetEase --official

Tune the model and the baselines with an equal budget of 20 trials::

    python examples/bundle_repro.py --model CrossCBR --dataset Youshu --tune 20 --baselines
"""

import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np

import cornac
from cornac.datasets import bundle
from cornac.eval_methods import BundleEvaluation
from cornac.hyperopt import Continuous, Discrete, RandomSearch
from cornac.metrics import NDCG, Recall

# Official per-dataset settings of each reference repository (see
# docs/bundle_models_implementation_plan.md). Values not listed use the class defaults.
CONFIGS = {
    "CrossCBR": {
        "Youshu": dict(num_layers=1, lambda_reg=1e-4, item_dropout=0.2, bundle_dropout=0.2,
                       agg_dropout=0.2, c_lambda=0.04, c_temp=0.25, num_epochs=100),
        "NetEase": dict(num_layers=1, lambda_reg=1e-4, item_dropout=0.1, bundle_dropout=0.3,
                        agg_dropout=0.1, c_lambda=0.1, c_temp=0.25, num_epochs=50),
        "iFashion": dict(num_layers=1, lambda_reg=4e-5, item_dropout=0.2, bundle_dropout=0.2,
                         agg_dropout=0.0, c_lambda=0.25, c_temp=0.2, num_epochs=50),
    },
    # The BGCN repository only configures NetEase; the same settings are used elsewhere.
    "BGCN": {
        dataset: dict(num_layers=2, learning_rate=3e-4, lambda_reg=1e-7, message_dropout=0.3,
                      node_dropout=0.0, num_epochs=1000, eval_every=5, patience=10)
        for dataset in ("Youshu", "NetEase", "iFashion")
    },
}

# Published test Recall@20 / NDCG@20 of each model (Rule 2 reference check).
PAPER = {
    ("CrossCBR", "Youshu"): (0.2813, 0.1668),
    ("CrossCBR", "NetEase"): (0.0842, 0.0457),
    ("CrossCBR", "iFashion"): (0.1173, 0.0895),
    ("BGCN", "Youshu"): (0.2347, 0.1345),
    ("BGCN", "NetEase"): (0.0491, 0.0258),
}

# RandomSearch spaces (Rule 3), centred on the official settings.
SPACES = {
    "CrossCBR": [
        Continuous("lambda_reg", 1e-6, 1e-3),
        Continuous("c_lambda", 0.01, 0.5),
        Discrete("c_temp", [0.1, 0.15, 0.2, 0.25, 0.3]),
        Discrete("num_layers", [1, 2, 3]),
        Discrete("item_dropout", [0.0, 0.1, 0.2, 0.3]),
        Discrete("bundle_dropout", [0.0, 0.1, 0.2, 0.3]),
    ],
    "BGCN": [
        Continuous("learning_rate", 1e-4, 3e-3),
        Continuous("lambda_reg", 1e-8, 1e-4),
        Discrete("message_dropout", [0.0, 0.1, 0.2, 0.3, 0.5]),
        Discrete("num_layers", [1, 2, 3]),
    ],
}

METRICS = [Recall(k=20), NDCG(k=20), Recall(k=40), NDCG(k=40)]


def environment():
    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    info = {"python": platform.python_version(), "numpy": np.__version__,
            "cornac_commit": run("git", "rev-parse", "HEAD")}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    return info


def load_evaluation(dataset):
    kwargs = dict(
        train_data=bundle.load_feedback(dataset, "train"),
        val_data=bundle.load_feedback(dataset, "validation"),
        test_data=bundle.load_feedback(dataset, "test"),
        bundle_item_data=bundle.load_bundle_item(dataset),
        test_mask="train",  # CrossCBR protocol: only training positives are masked
        verbose=True,
        **bundle.load_ids(dataset),
    )
    if dataset.lower() != "spotify":
        kwargs["user_item_data"] = bundle.load_user_item(dataset)
    return BundleEvaluation.from_splits(**kwargs)


def baselines(seed, eval_method, n_trials):
    """Simple baselines (Rule 2); trainable ones get the same tuning budget (Rule 3)."""
    from cornac.models import BPR, EASE, BundlePop, ItemKNN, MostPop

    models = [MostPop(), BundlePop(name="BundlePop-item", popularity="item"),
              BundlePop(name="BundlePop-user", popularity="user")]
    tunable = [
        (BPR(k=64, max_iter=200, seed=seed),
         [Continuous("learning_rate", 1e-4, 1e-1), Continuous("lambda_reg", 1e-5, 1e-1)]),
        (EASE(verbose=False), [Continuous("lamb", 10, 5000)]),
        (ItemKNN(k=50, verbose=False), [Discrete("k", [10, 20, 50, 100, 200])]),
    ]
    for model, space in tunable:
        if n_trials:
            models.append(RandomSearch(model=model, space=space, metric=NDCG(k=20),
                                       eval_method=eval_method, n_trails=n_trials))
        else:
            models.append(model)
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--dataset", default="Youshu", choices=["Youshu", "NetEase", "iFashion"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--official", action="store_true",
                        help="use official_variant=True (released-code behaviour)")
    parser.add_argument("--tune", type=int, default=0, metavar="N",
                        help="RandomSearch trials per trainable model (0: official config)")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--epochs", type=int, help="override num_epochs (smoke tests)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    model_cls = getattr(cornac.models, args.model)
    config = dict(CONFIGS[args.model].get(args.dataset, {}))
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.official:
        import inspect

        if "official_variant" not in inspect.signature(model_cls.__init__).parameters:
            parser.error("{} has no official_variant switch".format(args.model))
        config["official_variant"] = True
    eval_method = load_evaluation(args.dataset)

    run_dir = os.path.join(args.out, "{}_{}{}".format(
        args.model, args.dataset, "_official" if args.official else ""))
    os.makedirs(run_dir, exist_ok=True)
    report = {"model": args.model, "dataset": args.dataset, "config": config,
              "official_variant": args.official, "tune_trials": args.tune,
              "search_space": [repr(vars(d)) for d in SPACES.get(args.model, [])],
              "seeds": args.seeds, "protocol": "CrossCBR fixed split, test_mask=train, "
              "full-catalog ranking, validation-only selection",
              "environment": environment(), "per_seed": {}}

    for seed in args.seeds:
        model = model_cls(seed=seed, device=args.device, verbose=True, **config)
        if args.tune:
            model = RandomSearch(model=model, space=SPACES[args.model], metric=NDCG(k=20),
                                 eval_method=eval_method, n_trails=args.tune)
        models = [model] + (baselines(seed, eval_method, args.tune) if args.baselines else [])
        start = time.time()
        experiment = cornac.Experiment(eval_method=eval_method, models=models, metrics=METRICS,
                                       save_dir=os.path.join(run_dir, "seed{}".format(seed)))
        experiment.run()
        report["per_seed"][seed] = {
            result.model_name: {k: float(v) for k, v in result.metric_avg_results.items()}
            for result in experiment.result
        }
        report["per_seed"][seed]["wall_time_s"] = time.time() - start
        if args.tune:
            report["per_seed"][seed]["best_params"] = model.best_params

    name = models[0].name
    summary = {}
    for metric in ("Recall@20", "NDCG@20", "Recall@40", "NDCG@40"):
        values = [report["per_seed"][s][name][metric] for s in args.seeds]
        summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    report["summary"] = summary
    target = PAPER.get((args.model, args.dataset))
    if target is not None:
        report["paper"] = {"Recall@20": target[0], "NDCG@20": target[1]}
        report["relative_gap"] = {
            "Recall@20": summary["Recall@20"]["mean"] / target[0] - 1,
            "NDCG@20": summary["NDCG@20"]["mean"] / target[1] - 1,
        }
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ("summary", "paper", "relative_gap")
                      if k in report}, indent=2))


if __name__ == "__main__":
    main()
