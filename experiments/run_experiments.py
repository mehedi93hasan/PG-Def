"""
PG-Def: Main Experiment Runner
================================
Reproduces the experimental evaluation from the manuscript.

Usage:
    python experiments/run_experiments.py --dataset cicids2017 \
        --data_dir data/cicids2017 --mode train_eval

Reference:
    PG-Def manuscript, Section VI: Experimental Evaluation
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.pgdef_pipeline import PGDefPipeline
from src.pipeline.data_loader import DatasetLoader
from src.pipeline.adversarial_eval import AdversarialEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="PG-Def Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["cicids2017", "unswnb15", "edgeiiotset", "csv"],
        default="cicids2017",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/cicids2017",
        help="Path to dataset directory or CSV file",
    )
    parser.add_argument(
        "--mode",
        choices=["train_eval", "eval_only", "adversarial", "cross_domain"],
        default="train_eval",
        help="Experiment mode",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/pgdef",
        help="Directory for saving/loading models",
    )
    parser.add_argument(
        "--no_adaptive",
        action="store_true",
        help="Disable Tier 4 adaptive defense",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for saving results",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="Label",
        help="Label column name (for --dataset csv)",
    )
    return parser.parse_args()


def load_data(args, loader: DatasetLoader):
    """Load dataset based on args."""
    if args.dataset == "cicids2017":
        return loader.load_cicids2017(args.data_dir)
    elif args.dataset == "unswnb15":
        return loader.load_unswnb15(args.data_dir)
    elif args.dataset == "edgeiiotset":
        return loader.load_edgeiiotset(args.data_dir)
    elif args.dataset == "csv":
        return loader.load_csv(args.data_dir, label_col=args.label_col)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def run_train_eval(args):
    """Train and evaluate PG-Def."""
    loader   = DatasetLoader()
    pipeline = PGDefPipeline(
        use_adaptive = not args.no_adaptive,
        model_dir    = args.model_dir,
    )

    X_train, X_test, y_train, y_test = load_data(args, loader)
    print(f"\n[Experiment] Training set: {len(y_train):,} flows")
    print(f"[Experiment] Test set:     {len(y_test):,} flows")
    print(f"[Experiment] Estimated memory: "
          f"{pipeline.memory_footprint_mb():.1f} MB")

    # Train
    t0 = time.perf_counter()
    cv_results = pipeline.train(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"\n[Experiment] Training time: {train_time:.1f}s")

    # Evaluate
    metrics = pipeline.evaluate(X_test, y_test,
                                 dataset_name=args.dataset.upper())

    # Save
    pipeline.save()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                             f"{args.dataset}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "dataset":    args.dataset,
            "cv_results": {k: v for k, v in cv_results.items()
                           if k != "fold_metrics"},
            "test_metrics": metrics,
            "memory_mb":    pipeline.memory_footprint_mb(),
        }, f, indent=2)
    print(f"\n[Experiment] Results saved to {out_path}")
    return metrics


def run_adversarial(args):
    """Evaluate adversarial robustness."""
    loader   = DatasetLoader()
    pipeline = PGDefPipeline(use_adaptive=not args.no_adaptive,
                              model_dir=args.model_dir)

    _, X_test, _, y_test = load_data(args, loader)

    # Load pre-trained model
    pipeline.load()

    evaluator = AdversarialEvaluator(
        classifier=pipeline.ensemble,
        protocol_constraints=True,
    )

    print("\n[Experiment] Running adversarial evaluation...")
    results = evaluator.evaluate_all(X_test, y_test)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                             f"{args.dataset}_adversarial.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Experiment] Adversarial results saved to {out_path}")
    return results


def run_cross_domain(args):
    """
    Cross-domain generalisation: train on source, evaluate on all datasets.
    Requires data directories for all three datasets.
    """
    loader = DatasetLoader()
    datasets = {
        "cicids2017":   os.path.join(os.path.dirname(args.data_dir),
                                      "cicids2017"),
        "unswnb15":     os.path.join(os.path.dirname(args.data_dir),
                                      "unswnb15"),
        "edgeiiotset":  os.path.join(os.path.dirname(args.data_dir),
                                      "edgeiiotset"),
    }

    results_matrix = {}
    print("\n[Experiment] Cross-domain generalisation evaluation")
    print("Train → Test matrix (TPR %):\n")

    for train_ds in datasets:
        results_matrix[train_ds] = {}
        data_path = datasets[train_ds]
        if not os.path.exists(data_path):
            print(f"  Skipping {train_ds} (not found at {data_path})")
            continue

        # Train pipeline on source dataset
        pipeline = PGDefPipeline(use_adaptive=False,
                                  model_dir=f"{args.model_dir}_{train_ds}")
        args_copy = argparse.Namespace(**vars(args))
        args_copy.dataset  = train_ds
        args_copy.data_dir = data_path

        X_tr, _, y_tr, _ = load_data(args_copy, loader)
        pipeline.train(X_tr, y_tr, verbose=False)

        # Evaluate on all datasets
        for test_ds in datasets:
            test_path = datasets[test_ds]
            if not os.path.exists(test_path):
                continue
            args_copy.dataset  = test_ds
            args_copy.data_dir = test_path
            _, X_te, _, y_te = load_data(args_copy, loader)
            metrics = pipeline.evaluate(X_te, y_te, dataset_name=test_ds)
            results_matrix[train_ds][test_ds] = metrics["tpr"]
            print(f"  {train_ds:12s} → {test_ds:12s}: "
                  f"TPR={metrics['tpr']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "cross_domain_matrix.json")
    with open(out_path, "w") as f:
        json.dump(results_matrix, f, indent=2)
    print(f"\n[Experiment] Cross-domain matrix saved to {out_path}")
    return results_matrix


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("PG-Def: Protocol-Grounded Defense Framework")
    print("=" * 60)
    print(f"Dataset : {args.dataset}")
    print(f"Mode    : {args.mode}")
    print(f"Adaptive: {not args.no_adaptive}")
    print("=" * 60)

    if args.mode == "train_eval":
        run_train_eval(args)
    elif args.mode == "adversarial":
        run_adversarial(args)
    elif args.mode == "cross_domain":
        run_cross_domain(args)
    elif args.mode == "eval_only":
        loader   = DatasetLoader()
        pipeline = PGDefPipeline(use_adaptive=not args.no_adaptive,
                                  model_dir=args.model_dir)
        pipeline.load()
        _, X_te, _, y_te = load_data(args, loader)
        pipeline.evaluate(X_te, y_te, dataset_name=args.dataset.upper())
