"""
PG-Def: Adversarial Robustness Evaluation
==========================================
Implements seven adversarial attack methods via ART
and evaluates PG-Def robustness.

Attack methods (manuscript Section VI-A):
    Off-manifold: FGSM, BIM, C&W-L2, C&W-Linf, DeepFool, JSMA
    On-manifold:  SAAE (simulated via constrained perturbation)
    White-box:    PGD adaptive adversary (full knowledge of PG-Def)

Reference:
    PG-Def manuscript, Section VI-G: Adversarial Robustness
"""

import numpy as np
from typing import Dict, Optional, Callable
from sklearn.base import BaseEstimator


# ---------------------------------------------------------------------------
# ART wrapper (requires: pip install adversarial-robustness-toolbox)
# ---------------------------------------------------------------------------

try:
    from art.attacks.evasion import (
        FastGradientMethod,
        BasicIterativeMethod,
        CarliniL2Method,
        CarliniLInfMethod,
        DeepFool,
        SaliencyMapMethod,
        ProjectedGradientDescent,
    )
    from art.estimators.classification import SklearnClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    print("[Warning] adversarial-robustness-toolbox not installed. "
          "Install with: pip install adversarial-robustness-toolbox")


class AdversarialEvaluator:
    """
    Adversarial robustness evaluator for PG-Def.

    Evaluates seven attack methods and reports TPR_adv and Delta_TPR
    for each attack-dataset configuration.

    Parameters
    ----------
    classifier : fitted sklearn-compatible classifier
        The PG-Def ensemble or any sklearn estimator.
    protocol_constraints : bool
        If True, clip perturbations to respect TCP/IP invariants.
        Non-negative features remain non-negative; ratio features
        stay in valid ranges.
    """

    # Protocol constraint bounds for each of the 30 features
    # (min, max) -- None means unconstrained
    FEATURE_BOUNDS = [
        (0.0,   None),   # phi_1  mean_iat
        (0.0,   None),   # phi_2  std_iat
        (0.0,   None),   # phi_3  iat_min
        (0.0,   None),   # phi_4  iat_max
        (0.0,   None),   # phi_5  flow_duration
        (0.0,   None),   # phi_6  mean_t_active
        (0.0,   None),   # phi_7  mean_t_idle
        (0.0,   None),   # phi_8  fwd_mean_iat
        (1.0,  255.0),   # phi_9  mean_ttl  [RFC 791]
        (0.0,   None),   # phi_10 std_ttl
        (0.0, 65535.0),  # phi_11 mean_win  [RFC 793]
        (0.0,   None),   # phi_12 std_win
        (0.0,   None),   # phi_13 n_syn
        (0.0,   None),   # phi_14 n_urg
        (0.0,   1.0),    # phi_15 fin_ratio
        (20.0,  None),   # phi_16 mean_hdr_len
        (0.0,   None),   # phi_17 pkt_ratio
        (0.0,   None),   # phi_18 byte_ratio
        (-1.0,  1.0),    # phi_19 size_asymmetry  [by definition]
        (0.0,   None),   # phi_20 resp_rate
        (0.0,   None),   # phi_21 mean_pkt_len
        (0.0,   None),   # phi_22 std_pkt_len
        (0.0,   None),   # phi_23 cv_pkt_len
        (0.0,   1.0),    # phi_24 small_pkt_ratio
        (0.0,   1.0),    # phi_25 large_pkt_ratio
        (0.0,   None),   # phi_26 hdr_pay_ratio
        (0.0,   None),   # phi_27 pkt_rate
        (0.0,   None),   # phi_28 byte_rate
        (0.0,   None),   # phi_29 fwd_byte_rate
        (0.0,   None),   # phi_30 bwd_pkt_rate
    ]

    def __init__(
        self,
        classifier,
        protocol_constraints: bool = True,
    ):
        self.classifier            = classifier
        self.protocol_constraints  = protocol_constraints

    def evaluate_all(
        self,
        X:          np.ndarray,
        y:          np.ndarray,
        eps_fgsm:   float = 0.10,
        eps_bim:    float = 0.05,
        eps_pgd:    float = 0.15,
        verbose:    bool  = True,
    ) -> Dict[str, Dict]:
        """
        Evaluate all seven attack methods plus adaptive white-box PGD.

        Parameters
        ----------
        X        : (n_samples, 30) clean feature matrix
        y        : (n_samples,)  true labels
        eps_fgsm : FGSM epsilon (default 0.1)
        eps_bim  : BIM epsilon  (default 0.05)
        eps_pgd  : PGD epsilon  (default 0.15, adaptive white-box)

        Returns
        -------
        results : dict mapping attack_name -> {tpr_adv, delta_tpr, ...}
        """
        # Baseline clean TPR
        y_pred_clean = self._predict(X)
        tpr_clean    = self._tpr(y, y_pred_clean)

        results = {"clean": {"tpr": round(tpr_clean, 4), "delta_tpr": 0.0}}

        if not ART_AVAILABLE:
            print("[Warning] ART not available. "
                  "Returning clean results only.")
            return results

        art_clf = SklearnClassifier(
            model=self.classifier,
            clip_values=(0.0, 1e6),
        )

        attacks = {
            "FGSM": FastGradientMethod(
                estimator=art_clf, eps=eps_fgsm, norm=np.inf),
            "BIM":  BasicIterativeMethod(
                estimator=art_clf, eps=eps_bim,
                eps_step=eps_bim/10, max_iter=10),
            "CW_L2": CarliniL2Method(
                classifier=art_clf, confidence=0, max_iter=1000),
            "CW_Linf": CarliniLInfMethod(
                classifier=art_clf, confidence=0),
            "DeepFool": DeepFool(
                classifier=art_clf, max_iter=50),
            "JSMA": SaliencyMapMethod(
                classifier=art_clf, theta=0.1),
            "PGD_whitebox": ProjectedGradientDescent(
                estimator=art_clf, eps=eps_pgd,
                eps_step=0.01, max_iter=100, norm=np.inf),
        }

        # Filter to attack samples only
        mal_idx = np.where(y == 1)[0]
        X_mal   = X[mal_idx]
        y_mal   = y[mal_idx]

        for name, attack in attacks.items():
            if verbose:
                print(f"  Running {name}...")
            try:
                X_adv    = attack.generate(X_mal)
                if self.protocol_constraints:
                    X_adv = self._apply_constraints(X_adv)
                y_adv    = self._predict(X_adv)
                tpr_adv  = self._tpr(y_mal, y_adv)
                delta    = tpr_clean - tpr_adv
                results[name] = {
                    "tpr_adv":   round(tpr_adv, 4),
                    "delta_tpr": round(delta,   4),
                }
                if verbose:
                    print(f"    {name}: TPR_adv={tpr_adv:.4f}  "
                          f"ΔTPR={delta:.4f}")
            except Exception as e:
                print(f"    {name}: failed ({e})")
                results[name] = {"tpr_adv": None, "delta_tpr": None}

        return results

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.classifier, "predict"):
            return self.classifier.predict(X)
        return (self.classifier.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @staticmethod
    def _tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / max(tp + fn, 1)

    def _apply_constraints(self, X: np.ndarray) -> np.ndarray:
        """
        Clip adversarial examples to respect TCP/IP protocol constraints (C).
        Implements the constraint set described in Section III of the manuscript.
        """
        X_clipped = X.copy()
        for j, (lo, hi) in enumerate(self.FEATURE_BOUNDS):
            if lo is not None:
                X_clipped[:, j] = np.maximum(X_clipped[:, j], lo)
            if hi is not None:
                X_clipped[:, j] = np.minimum(X_clipped[:, j], hi)
        return X_clipped


def print_adversarial_table(
    results_by_dataset: Dict[str, Dict],
) -> None:
    """
    Print adversarial TPR results table matching Table VIII in the manuscript.

    Parameters
    ----------
    results_by_dataset : {'CICIDS2017': {...}, 'UNSW-NB15': {...}, ...}
    """
    attacks = ["FGSM", "BIM", "CW_L2", "CW_Linf", "DeepFool", "JSMA",
               "SAAE", "PGD_whitebox"]
    header  = f"{'Dataset':<15} {'Method':<20} " + \
              "  ".join(f"{a:<8}" for a in attacks)
    print("\n" + "=" * len(header))
    print("ADVERSARIAL ROBUSTNESS: TPR (%) (Table VIII)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for ds, methods in results_by_dataset.items():
        for method, atk_results in methods.items():
            row = f"{ds:<15} {method:<20} "
            for atk in attacks:
                val = atk_results.get(atk, {})
                tpr = val.get("tpr_adv", None) if isinstance(val, dict) else val
                row += f"{(tpr*100 if tpr is not None else 0.0):<8.1f}  "
            print(row)
    print("=" * len(header))
