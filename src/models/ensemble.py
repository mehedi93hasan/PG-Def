"""
PG-Def: Lightweight Ensemble Classifier
========================================
Implements Tier 3: Weighted Soft-Voting Ensemble.

Combines Random Forest, XGBoost, and Logistic Regression with
weights w_RF=0.4, w_XGB=0.4, w_LR=0.2 selected via grid search
on held-out 20% validation partition.

All three classifiers satisfy the edge-device budget:
    B = (M_max=100 MB, T_max=100 ms, E_max=2 W)

Reference:
    PG-Def manuscript, Section V-B: Classifier Justification
"""

import os
import pickle
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ---------------------------------------------------------------------------
# Ensemble configuration (manuscript Section V-B)
# ---------------------------------------------------------------------------

RF_CONFIG = dict(
    n_estimators  = 50,
    max_depth     = 10,
    random_state  = 42,
    n_jobs        = -1,
    class_weight  = "balanced",
)

XGB_CONFIG = dict(
    n_estimators      = 100,
    max_depth         = 6,
    learning_rate     = 0.1,   # eta = 0.1 (grid search over {0.01,0.05,0.1,0.2,0.3})
    reg_lambda        = 1.0,   # L2 regularisation
    gamma             = 0.1,   # minimum loss reduction for split
    use_label_encoder = False,
    eval_metric       = "logloss",
    random_state      = 42,
    n_jobs            = -1,
)

LR_CONFIG = dict(
    C            = 1.0,        # lambda_LR = 1.0 (standard L2)
    penalty      = "l2",
    solver       = "lbfgs",
    max_iter     = 1000,
    random_state = 42,
)

# Voting weights (grid search on 20% validation split of CICIDS2017)
ENSEMBLE_WEIGHTS = {
    "rf":  0.4,
    "xgb": 0.4,
    "lr":  0.2,
}


class PGDefEnsemble:
    """
    Weighted soft-voting ensemble for PG-Def (Tier 3).

    Decision rule:
        f_ens(F) = argmax_c  sum_m  w_m * 1[f_m(F) = c]

    Weights are fixed for all evaluation domains ---
    no per-dataset tuning is performed (cross-domain claim).

    Parameters
    ----------
    weights : dict
        Voting weights for {'rf', 'xgb', 'lr'}.
        Must sum to 1.0.
    """

    def __init__(self, weights: Dict[str, float] = ENSEMBLE_WEIGHTS):
        assert abs(sum(weights.values()) - 1.0) < 1e-9, \
            "Ensemble weights must sum to 1.0"
        self.weights  = weights
        self.rf       = RandomForestClassifier(**RF_CONFIG)
        self.xgb      = xgb.XGBClassifier(**XGB_CONFIG)
        self.lr       = LogisticRegression(**LR_CONFIG)
        self.scaler   = StandardScaler()
        self._fitted  = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
    ) -> "PGDefEnsemble":
        """
        Train all three classifiers.

        Parameters
        ----------
        X_train : (n_samples, 30) feature matrix
        y_train : (n_samples,)  binary labels {0=benign, 1=malicious}
        X_val   : validation set for XGBoost early stopping (optional)
        y_val   : validation labels (optional)
        """
        # Scale features (important for LR; RF/XGB are scale-invariant)
        X_scaled = self.scaler.fit_transform(X_train)

        print("[PG-Def] Training Random Forest (T=50, D=10)...")
        self.rf.fit(X_scaled, y_train)

        print("[PG-Def] Training XGBoost (M=100, D=6, η=0.1)...")
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.xgb.fit(
                X_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
        else:
            self.xgb.fit(X_scaled, y_train)

        print("[PG-Def] Training Logistic Regression (L2, λ=1.0)...")
        self.lr.fit(X_scaled, y_train)

        self._fitted = True
        print("[PG-Def] Ensemble training complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted soft-voting probability.

        Returns
        -------
        proba : (n_samples, 2) array  [P(benign), P(malicious)]
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X)

        p_rf  = self.rf.predict_proba(X_scaled)
        p_xgb = self.xgb.predict_proba(X_scaled)
        p_lr  = self.lr.predict_proba(X_scaled)

        return (
            self.weights["rf"]  * p_rf  +
            self.weights["xgb"] * p_xgb +
            self.weights["lr"]  * p_lr
        )

    def predict(
        self,
        X:         np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Hard binary prediction.

        Parameters
        ----------
        threshold : float
            Voting threshold tau_vote (default 0.5).
            Dynamically adjusted by Component 3 of adaptive defense.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def confidence_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return ensemble confidence c_ens = max_c sum_m w_m * 1[f_m(F)=c].
        Used by Component 2 (Confidence Monitor) of adaptive defense.
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save ensemble to disk (~15 MB total)."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "rf.pkl"),  "wb") as f:
            pickle.dump(self.rf, f)
        with open(os.path.join(path, "xgb.pkl"), "wb") as f:
            pickle.dump(self.xgb, f)
        with open(os.path.join(path, "lr.pkl"),  "wb") as f:
            pickle.dump(self.lr, f)
        with open(os.path.join(path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(path, "weights.pkl"), "wb") as f:
            pickle.dump(self.weights, f)
        print(f"[PG-Def] Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PGDefEnsemble":
        """Load ensemble from disk."""
        with open(os.path.join(path, "weights.pkl"), "rb") as f:
            weights = pickle.load(f)
        obj = cls(weights=weights)
        with open(os.path.join(path, "rf.pkl"),  "rb") as f:
            obj.rf = pickle.load(f)
        with open(os.path.join(path, "xgb.pkl"), "rb") as f:
            obj.xgb = pickle.load(f)
        with open(os.path.join(path, "lr.pkl"),  "rb") as f:
            obj.lr = pickle.load(f)
        with open(os.path.join(path, "scaler.pkl"), "rb") as f:
            obj.scaler = pickle.load(f)
        obj._fitted = True
        print(f"[PG-Def] Ensemble loaded from {path}")
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Ensemble is not fitted. Call fit() before predict().")
