"""
PG-Def: Adaptive Defense System
================================
Implements Algorithm 2 from the manuscript: Adaptive Defense System (Tier 4).

Four components operating in order C1 → C2 → C3 → C4:

    C1: Attack Fingerprint Cache    -- Bloom filter, +128 bytes memory
    C2: Ensemble Confidence Monitor -- Borderline flow re-inspection
    C3: Dynamic Threshold Adaptation -- PI controller for FPR <= 2%
    C4: Protocol Feature Drift Monitor -- Two-level Welford drift detection

Total overhead: <=8 KB memory, <0.05 ms latency per flow.

Reference:
    PG-Def manuscript, Section V-C: Adaptive Defense System
"""

import math
import hashlib
import numpy as np
from typing import Optional, Tuple
from collections import deque

from src.models.ensemble import PGDefEnsemble


# ---------------------------------------------------------------------------
# Critical feature indices (0-indexed) for fingerprint and re-inspection
# phi_2, phi_10, phi_17 -> indices 1, 9, 16
# phi_18, phi_22        -> indices 17, 21
# ---------------------------------------------------------------------------
CRITICAL_IDX     = [1, 9, 16]        # phi_2, phi_10, phi_17
SUBSUMED_IDX     = [17, 21]          # phi_18, phi_22
INSPECTION_IDX   = CRITICAL_IDX + SUBSUMED_IDX   # {phi_2,phi_10,phi_17,phi_18,phi_22}


# ---------------------------------------------------------------------------
# Bloom Filter (C1: Attack Fingerprint Cache)
# m=1,024 bits, k=3 hash functions -> FPR < 1% for 200 fingerprints
# ---------------------------------------------------------------------------

class BloomFilter:
    """
    Lightweight Bloom filter for attack fingerprint caching.

    Parameters
    ----------
    m : int   Number of bits (default 1,024).
    k : int   Number of hash functions (default 3).
    """

    def __init__(self, m: int = 1024, k: int = 3):
        self.m    = m
        self.k    = k
        self._bits = bytearray(math.ceil(m / 8))

    def _hash_positions(self, item: bytes) -> list:
        positions = []
        for i in range(self.k):
            h = int(hashlib.md5(item + i.to_bytes(2, "big")).hexdigest(), 16)
            positions.append(h % self.m)
        return positions

    def insert(self, fingerprint: bytes) -> None:
        for pos in self._hash_positions(fingerprint):
            byte_idx, bit_idx = divmod(pos, 8)
            self._bits[byte_idx] |= (1 << bit_idx)

    def query(self, fingerprint: bytes) -> bool:
        for pos in self._hash_positions(fingerprint):
            byte_idx, bit_idx = divmod(pos, 8)
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def memory_bytes(self) -> int:
        return len(self._bits)  # 128 bytes for m=1024


def make_fingerprint(features: np.ndarray) -> bytes:
    """
    Construct 3-feature binary fingerprint for Bloom filter.
    <floor(phi_2 * 10), floor(phi_10), floor(phi_17)>
    """
    fp = (
        int(features[1]  * 10),   # phi_2  * 10
        int(features[9]),          # phi_10
        int(features[16]),         # phi_17
    )
    return str(fp).encode()


# ---------------------------------------------------------------------------
# Welford drift accumulator (C4)
# ---------------------------------------------------------------------------

class _WelfordDriftAcc:
    """
    Single Welford accumulator for drift monitoring.
    O(1) memory: stores only (n, mean, M2).
    """
    def __init__(self):
        self.n    = 0
        self.mean = 0.0
        self.m2   = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta1 = x - self.mean
        self.mean += delta1 / self.n
        delta2 = x - self.mean
        self.m2 += delta1 * delta2

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n >= 2 else 0.0


# ---------------------------------------------------------------------------
# Protocol-specific drift bounds (from Table I, Principle 2)
# Indexed by feature index 0..29
# ---------------------------------------------------------------------------
_DRIFT_BOUNDS = np.array([
    0.30,  # phi_1  High
    0.15,  # phi_2  Critical
    0.30,  # phi_3  High
    0.40,  # phi_4  Medium
    0.40,  # phi_5  Medium
    0.40,  # phi_6  Medium
    0.40,  # phi_7  Medium
    0.30,  # phi_8  High
    0.30,  # phi_9  High
    0.15,  # phi_10 Critical
    0.30,  # phi_11 High
    0.40,  # phi_12 Medium
    0.30,  # phi_13 High
    0.40,  # phi_14 Medium
    0.40,  # phi_15 Medium
    0.40,  # phi_16 Medium
    0.15,  # phi_17 Critical
    0.15,  # phi_18 Critical
    0.30,  # phi_19 High
    0.40,  # phi_20 Medium
    0.50,  # phi_21 Baseline (denominator only)
    0.15,  # phi_22 Critical
    0.30,  # phi_23 High
    0.40,  # phi_24 Medium
    0.40,  # phi_25 Medium
    0.30,  # phi_26 High
    0.30,  # phi_27 High
    0.30,  # phi_28 High
    0.30,  # phi_29 High
    0.40,  # phi_30 Medium
])


# ---------------------------------------------------------------------------
# Main Adaptive Defense System
# ---------------------------------------------------------------------------

class AdaptiveDefenseSystem:
    """
    Tier 4: Adaptive Defense System (Algorithm 2).

    Wraps a trained PGDefEnsemble and applies four adaptive components
    in order C1 → C2 → C3 → C4.

    Parameters
    ----------
    ensemble : PGDefEnsemble
        Fitted base ensemble (Tier 3).
    tau_conf : float
        Initial confidence threshold (default 0.75).
    tau_vote : float
        Initial voting threshold (default 0.5).
    fpr_target : float
        Target false positive rate (default 0.02 = 2%).
    alpha_p : float
        PI controller proportional gain (default 0.01).
    alpha_i : float
        PI controller integral gain (default 0.001).
    W_short : int
        Short-term drift window in flows (default 10,000).
    W_long : int
        Long-term drift window in flows (default 100,000).
    bloom_m : int
        Bloom filter bit size (default 1,024).
    bloom_k : int
        Number of Bloom filter hash functions (default 3).
    """

    def __init__(
        self,
        ensemble:   PGDefEnsemble,
        tau_conf:   float = 0.75,
        tau_vote:   float = 0.50,
        fpr_target: float = 0.02,
        alpha_p:    float = 0.01,
        alpha_i:    float = 0.001,
        W_short:    int   = 10_000,
        W_long:     int   = 100_000,
        bloom_m:    int   = 1024,
        bloom_k:    int   = 3,
    ):
        self.ensemble   = ensemble
        self.tau_conf   = tau_conf
        self.tau_vote   = tau_vote
        self.fpr_target = fpr_target
        self.alpha_p    = alpha_p
        self.alpha_i    = alpha_i
        self.W_short    = W_short
        self.W_long     = W_long

        # C1: Bloom filter fingerprint cache
        self._bloom   = BloomFilter(m=bloom_m, k=bloom_k)

        # C3: PI controller state
        self._integral: float = 0.0
        self._fp_window = deque(maxlen=W_short)  # sliding window for FPR

        # C4: Per-feature two-level Welford drift accumulators
        self._short_acc = [_WelfordDriftAcc() for _ in range(30)]
        self._long_acc  = [_WelfordDriftAcc() for _ in range(30)]
        self._drift_alerts: list = []

        # Statistics
        self.n_total       = 0
        self.n_cache_hit   = 0
        self.n_borderline  = 0
        self.n_reclassified = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        features: np.ndarray,
        true_label: Optional[int] = None,
    ) -> Tuple[int, float, str]:
        """
        Classify a single flow using all four adaptive components.

        Parameters
        ----------
        features   : (30,) feature vector from Tier 2
        true_label : ground-truth label for FPR tracking (optional)

        Returns
        -------
        (prediction, confidence, component_used)
            prediction     : int {0=benign, 1=malicious}
            confidence     : float [0,1]
            component_used : str  {'C1_cache','C2_high','C2_borderline','ensemble'}
        """
        self.n_total += 1
        F = features.reshape(1, -1)

        # ---- C4: Update drift monitors (background, does not affect decision) ----
        self._update_drift(features)

        # ---- C1: Fingerprint Cache ----
        fp = make_fingerprint(features)
        if self._bloom.query(fp):
            self.n_cache_hit += 1
            self._update_fpr_tracker(1, true_label)
            return 1, 1.0, "C1_cache"

        # ---- C2: Ensemble Confidence Monitor ----
        c_ens = float(self.ensemble.confidence_scores(F)[0])

        if c_ens >= self.tau_conf:
            # High confidence: malicious
            self._bloom.insert(fp)
            self._update_fpr_tracker(1, true_label)
            return 1, c_ens, "C2_high"

        elif c_ens <= (1.0 - self.tau_conf):
            # High confidence: benign
            self._update_fpr_tracker(0, true_label)
            return 0, c_ens, "C2_high"

        else:
            # Borderline: secondary inspection on critical subspace
            # Build full 30-dim vector retaining only critical features;
            # all other positions zeroed so scaler dimension matches.
            self.n_borderline += 1
            F_ext = np.zeros((1, 30), dtype=np.float64)
            for idx in INSPECTION_IDX:
                F_ext[0, idx] = features[idx]
            c_ext = float(self.ensemble.confidence_scores(F_ext)[0])
            pred  = int(c_ext >= self.tau_vote)
            if pred != int(c_ens >= 0.5):
                self.n_reclassified += 1
            self._update_fpr_tracker(pred, true_label)
            return pred, c_ext, "C2_borderline"

    def classify_batch(
        self,
        X:           np.ndarray,
        true_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Classify a batch of flows.

        Parameters
        ----------
        X           : (n_samples, 30) feature matrix
        true_labels : (n_samples,) optional ground truth

        Returns
        -------
        predictions : (n_samples,) int array
        """
        preds = np.zeros(len(X), dtype=int)
        for i, feat in enumerate(X):
            lbl = true_labels[i] if true_labels is not None else None
            preds[i], _, _ = self.classify(feat, true_label=lbl)
            # C3: update threshold after each flow
            self._update_threshold()
        return preds

    def get_drift_alerts(self) -> list:
        """Return list of (feature_idx, delta_k) for triggered drift alerts."""
        alerts = list(self._drift_alerts)
        self._drift_alerts.clear()
        return alerts

    def memory_overhead_bytes(self) -> int:
        """Total memory overhead of all four adaptive components."""
        return (
            self._bloom.memory_bytes()   +   # C1: 128 bytes
            1  * 8                        +   # C2: 1 float comparison
            2  * 8                        +   # C3: 2 floats (tau_vote, integral)
            30 * 2 * 3 * 8                    # C4: 30 features * 2 levels * 3 fields
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_drift(self, features: np.ndarray) -> None:
        """C4: Update two-level Welford drift accumulators for all 30 features."""
        for k in range(30):
            x = float(features[k])
            self._short_acc[k].update(x)
            self._long_acc[k].update(x)

            # Check drift condition when long accumulator has enough data
            if self._long_acc[k].n >= 100:
                sigma_long = self._long_acc[k].std
                if sigma_long > 1e-9:
                    delta_k = abs(
                        self._short_acc[k].mean -
                        self._long_acc[k].mean
                    ) / sigma_long

                    if delta_k > _DRIFT_BOUNDS[k]:
                        self._drift_alerts.append((k + 1, delta_k))  # 1-indexed

    def _update_fpr_tracker(
        self, pred: int, true_label: Optional[int]
    ) -> None:
        """Track false positives for C3 PI controller."""
        if true_label is not None:
            is_fp = int(pred == 1 and true_label == 0)
            self._fp_window.append(is_fp)

    def _update_threshold(self) -> None:
        """
        C3: PI controller update.
        tau_vote^{t+1} = tau_vote^t + alpha_p * e^t + alpha_i * sum e^s
        """
        if len(self._fp_window) < 100:
            return

        fpr_current   = sum(self._fp_window) / len(self._fp_window)
        error         = fpr_current - self.fpr_target
        self._integral += error

        self.tau_vote = np.clip(
            self.tau_vote + self.alpha_p * error + self.alpha_i * self._integral,
            0.3, 0.8
        )
        # Confidence threshold co-adaptation
        self.tau_conf = np.clip(
            self.tau_conf + (self.alpha_p / 2) * error,
            0.6, 0.9
        )

    def summary(self) -> dict:
        """Return adaptive defense statistics."""
        hit_rate = self.n_cache_hit / max(self.n_total, 1)
        border_rate = self.n_borderline / max(self.n_total, 1)
        reclassify_rate = self.n_reclassified / max(self.n_borderline, 1)
        return {
            "total_flows":       self.n_total,
            "cache_hit_rate":    round(hit_rate, 4),
            "borderline_rate":   round(border_rate, 4),
            "reclassify_rate":   round(reclassify_rate, 4),
            "tau_vote":          round(self.tau_vote, 4),
            "tau_conf":          round(self.tau_conf, 4),
            "memory_bytes":      self.memory_overhead_bytes(),
        }
