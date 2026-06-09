"""
PG-Def: Complete Four-Tier Detection Pipeline
===============================================
Integrates all four tiers:

    Tier 1: Flow Aggregation (eBPF -- simulated via PCAP/CSV)
    Tier 2: Welford Streaming Feature Extraction
    Tier 3: Lightweight Ensemble Classification
    Tier 4: Adaptive Defense System

Supports training, evaluation, and real-time detection modes.

Reference:
    PG-Def manuscript, Section V-A: Three-Tier Streaming Pipeline
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, f1_score
)
from imblearn.over_sampling import SMOTE

from src.features.welford_extractor import ProtocolGroundedExtractor
from src.models.ensemble import PGDefEnsemble
from src.defense.adaptive_defense import AdaptiveDefenseSystem


class PGDefPipeline:
    """
    Complete PG-Def four-tier detection pipeline.

    Parameters
    ----------
    use_adaptive : bool
        Whether to enable Tier 4 adaptive defense (default True).
    n_folds : int
        Number of cross-validation folds (default 5).
    smote_k : int
        SMOTE neighbours for minority oversampling (default 5).
    model_dir : str
        Directory for saving/loading trained models.
    """

    def __init__(
        self,
        use_adaptive: bool = True,
        n_folds:      int  = 5,
        smote_k:      int  = 5,
        model_dir:    str  = "models/pgdef",
    ):
        self.use_adaptive = use_adaptive
        self.n_folds      = n_folds
        self.smote_k      = smote_k
        self.model_dir    = model_dir

        self.extractor = ProtocolGroundedExtractor()
        self.ensemble  = PGDefEnsemble()
        self.adaptive  = None  # initialised after ensemble is fitted

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X:       np.ndarray,
        y:       np.ndarray,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the full pipeline using stratified 5-fold cross-validation.

        Minority classes are oversampled with SMOTE on training folds only.
        Test folds retain original distributions (manuscript Section VI-A).

        Parameters
        ----------
        X : (n_samples, 30) protocol-grounded feature matrix
        y : (n_samples,)  binary labels {0=benign, 1=malicious}

        Returns
        -------
        cv_results : dict with per-fold and mean metrics
        """
        if verbose:
            print(f"[PG-Def] Training on {X.shape[0]:,} flows "
                  f"with {X.shape[1]} features")
            print(f"[PG-Def] Cross-validation: {self.n_folds}-fold stratified")

        skf    = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                  random_state=42)
        smote  = SMOTE(k_neighbors=self.smote_k, random_state=42)

        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # SMOTE only on training partition (manuscript Section VI-A)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

            # 20% validation split for weight search and early stopping
            val_size   = max(int(0.2 * len(X_tr_res)), 100)
            X_val      = X_tr_res[-val_size:]
            y_val      = y_tr_res[-val_size:]
            X_tr_final = X_tr_res[:-val_size]
            y_tr_final = y_tr_res[:-val_size]

            self.ensemble.fit(X_tr_final, y_tr_final, X_val, y_val)

            y_pred  = self.ensemble.predict(X_te)
            y_proba = self.ensemble.predict_proba(X_te)[:, 1]

            metrics = self._compute_metrics(y_te, y_pred, y_proba)
            fold_metrics.append(metrics)

            if verbose:
                print(f"  Fold {fold+1}/{self.n_folds}: "
                      f"TPR={metrics['tpr']:.4f}  "
                      f"FPR={metrics['fpr']:.4f}  "
                      f"F1={metrics['f1']:.4f}  "
                      f"AUC={metrics['auc']:.4f}")

        # Final model: train on full dataset
        if verbose:
            print("[PG-Def] Training final model on full dataset...")
        X_full, y_full = smote.fit_resample(X, y)
        val_size       = max(int(0.2 * len(X_full)), 100)
        self.ensemble.fit(
            X_full[:-val_size], y_full[:-val_size],
            X_full[-val_size:], y_full[-val_size:]
        )

        if self.use_adaptive:
            self.adaptive = AdaptiveDefenseSystem(self.ensemble)

        # Aggregate results
        cv_results = {
            "fold_metrics": fold_metrics,
            "mean_tpr":  np.mean([m["tpr"]  for m in fold_metrics]),
            "std_tpr":   np.std( [m["tpr"]  for m in fold_metrics]),
            "mean_fpr":  np.mean([m["fpr"]  for m in fold_metrics]),
            "mean_f1":   np.mean([m["f1"]   for m in fold_metrics]),
            "mean_auc":  np.mean([m["auc"]  for m in fold_metrics]),
        }

        if verbose:
            print(f"\n[PG-Def] CV Results:")
            print(f"  TPR: {cv_results['mean_tpr']:.4f} ± "
                  f"{cv_results['std_tpr']:.4f}")
            print(f"  FPR: {cv_results['mean_fpr']:.4f}")
            print(f"  F1:  {cv_results['mean_f1']:.4f}")
            print(f"  AUC: {cv_results['mean_auc']:.4f}")

        return cv_results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X:           np.ndarray,
        y:           np.ndarray,
        dataset_name: str = "test",
    ) -> Dict:
        """
        Evaluate the pipeline on a held-out test set.

        Parameters
        ----------
        X            : (n_samples, 30) feature matrix
        y            : (n_samples,)  ground truth labels
        dataset_name : label for logging

        Returns
        -------
        metrics dict with TPR, FPR, Precision, F1, AUC-ROC
        """
        t0 = time.perf_counter()

        if self.use_adaptive and self.adaptive is not None:
            y_pred = self.adaptive.classify_batch(X, true_labels=y)
        else:
            y_pred = self.ensemble.predict(X)

        elapsed = time.perf_counter() - t0
        latency_ms = (elapsed / max(len(X), 1)) * 1000

        y_proba = self.ensemble.predict_proba(X)[:, 1]
        metrics = self._compute_metrics(y, y_pred, y_proba)
        metrics["latency_ms_per_flow"] = round(latency_ms, 4)
        metrics["throughput_flows_s"]  = round(len(X) / elapsed, 1)

        print(f"\n[PG-Def] Evaluation on {dataset_name}:")
        print(f"  TPR       : {metrics['tpr']:.4f}")
        print(f"  FPR       : {metrics['fpr']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  F1-Score  : {metrics['f1']:.4f}")
        print(f"  AUC-ROC   : {metrics['auc']:.4f}")
        print(f"  Latency   : {metrics['latency_ms_per_flow']:.3f} ms/flow")
        print(f"  Throughput: {metrics['throughput_flows_s']:,.0f} flows/s")

        if self.use_adaptive and self.adaptive is not None:
            adap = self.adaptive.summary()
            print(f"  Cache hit : {adap['cache_hit_rate']:.2%}")
            print(f"  Borderline: {adap['borderline_rate']:.2%}")
            metrics["adaptive"] = adap

        return metrics

    def memory_footprint_mb(self) -> float:
        """
        Estimate total memory footprint.
        Flow table: 100,000 * 128 bytes = 12.8 MB
        RF + XGBoost: ~13 MB
        LR: ~2 MB
        Adaptive: <=8 KB
        Total: ~27.8 MB
        """
        flow_table_mb = (100_000 * 128) / (1024 ** 2)   # 12.8 MB
        ensemble_mb   = 15.0                              # RF+XGB+LR
        adaptive_mb   = 0.008 if self.use_adaptive else 0.0
        return round(flow_table_mb + ensemble_mb + adaptive_mb, 2)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        self.ensemble.save(self.model_dir)
        print(f"[PG-Def] Pipeline saved to {self.model_dir}")

    def load(self) -> None:
        self.ensemble = PGDefEnsemble.load(self.model_dir)
        if self.use_adaptive:
            self.adaptive = AdaptiveDefenseSystem(self.ensemble)
        print(f"[PG-Def] Pipeline loaded from {self.model_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        y_true:  np.ndarray,
        y_pred:  np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                           labels=[0, 1]).ravel()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = 0.0
        return {
            "tpr":       round(tpr,       4),
            "fpr":       round(fpr,       4),
            "precision": round(precision, 4),
            "f1":        round(f1,        4),
            "auc":       round(auc,       4),
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
        }
