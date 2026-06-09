"""
PG-Def: Protocol-Grounded Feature Extractor
============================================
Implements Algorithm 1 from the manuscript:
Stream-Based Protocol-Grounded Feature Extraction (Tier 2).

Extracts 30 protocol-grounded features from raw network packets
using Welford's online algorithm with O(1) per-flow memory.

Feature Categories:
    - Category 1: Time Dynamics       (phi_1  -- phi_8)
    - Category 2: Header Invariants   (phi_9  -- phi_16)
    - Category 3: Traffic Symmetry    (phi_17 -- phi_20)
    - Category 4: Payload Dynamics    (phi_21 -- phi_26)
    - Category 5: Velocity            (phi_27 -- phi_30)

Reference:
    PG-Def: A Protocol-Grounded Lightweight Defense Framework
    for Adversarially Robust Network Intrusion Detection
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Per-flow state structure (128 bytes per flow)
# ---------------------------------------------------------------------------

@dataclass
class FlowState:
    """
    Per-flow accumulator state (128 bytes).
    Maintains running Welford accumulators without storing packet history.
    Satisfies Principle 3 (Streaming Computability).
    """
    # Packet counter
    n: int = 0

    # Timestamps
    t_first: float = 0.0
    t_last:  float = 0.0

    # Category 1: Time Dynamics -- Welford accumulators for IAT
    mean_iat:  float = 0.0
    m2_iat:    float = 0.0          # Welford M2 for sigma_IAT
    iat_min:   float = float('inf')
    iat_max:   float = float('-inf')

    # Forward-direction IAT (phi_8)
    n_fwd_iat:    int   = 0
    mean_fwd_iat: float = 0.0
    m2_fwd_iat:   float = 0.0

    # Active / idle burst times (phi_6, phi_7)
    n_active:      int   = 0
    mean_t_active: float = 0.0
    m2_t_active:   float = 0.0
    n_idle:        int   = 0
    mean_t_idle:   float = 0.0
    m2_t_idle:     float = 0.0
    burst_start:   float = 0.0
    in_burst:      bool  = False

    # Category 2: Header Invariants -- Welford accumulators
    mean_ttl:  float = 0.0
    m2_ttl:    float = 0.0
    mean_win:  float = 0.0
    m2_win:    float = 0.0
    n_syn:     int   = 0
    n_urg:     int   = 0
    n_fin:     int   = 0
    mean_hdr:  float = 0.0
    m2_hdr:    float = 0.0

    # Category 3: Traffic Symmetry -- directional counters
    n_fwd:  int   = 0
    n_bwd:  int   = 0
    b_fwd:  int   = 0    # forward bytes
    b_bwd:  int   = 0    # backward bytes

    # Category 4: Payload Dynamics -- Welford accumulators for pkt length
    mean_len: float = 0.0
    m2_len:   float = 0.0
    n_small:  int   = 0   # packets < 64 bytes
    n_large:  int   = 0   # packets > 1200 bytes
    sum_hdr:  int   = 0   # sum of header lengths
    sum_pay:  int   = 0   # sum of payload lengths


IDLE_TIMEOUT:  float = 120.0   # seconds (robust default, Idrissi et al. 2024)
BURST_THRESH:  float = 0.1     # seconds (min inter-burst gap, CICIDS2017)
SMALL_PKT_THR: int   = 64      # bytes
LARGE_PKT_THR: int   = 1200    # bytes


# ---------------------------------------------------------------------------
# Welford online update helpers
# ---------------------------------------------------------------------------

def _welford_update(n: int, mean: float, m2: float,
                    x: float) -> Tuple[float, float]:
    """
    Single-pass Welford update.
    Returns updated (mean, M2).
    sigma = sqrt(M2 / (n-1)) for n >= 2.
    Uses n-1 denominator (Bessel's correction).
    """
    delta1 = x - mean
    mean  += delta1 / n
    delta2 = x - mean
    m2    += delta1 * delta2
    return mean, m2


def _welford_std(m2: float, n: int) -> float:
    """
    Compute sample standard deviation from Welford M2 accumulator.
    Returns 0 if n < 2.
    """
    if n < 2:
        return 0.0
    return math.sqrt(m2 / (n - 1))


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

class ProtocolGroundedExtractor:
    """
    Tier 2: Stream-Based Protocol-Grounded Feature Extraction.

    Extracts all 30 protocol-grounded features from a packet stream
    using Welford's online algorithm.  Memory footprint is O(1) per flow
    (one FlowState per active 5-tuple key).

    Parameters
    ----------
    idle_timeout : float
        Flow eviction timeout in seconds (default 120s).
    burst_threshold : float
        Active/idle segmentation threshold in seconds (default 0.1s).
    max_flows : int
        Maximum concurrent flows in hash table (default 100,000).
    """

    FEATURE_NAMES = [
        # Category 1: Time Dynamics
        "phi_01_mean_iat",        # mu_IAT
        "phi_02_std_iat",         # sigma_IAT  [CRITICAL]
        "phi_03_iat_min",         # IAT_min
        "phi_04_iat_max",         # IAT_max
        "phi_05_flow_duration",   # T_flow
        "phi_06_mean_t_active",   # mu_T_active
        "phi_07_mean_t_idle",     # mu_T_idle
        "phi_08_fwd_mean_iat",    # mu_fwd_IAT
        # Category 2: Header Invariants
        "phi_09_mean_ttl",        # mu_TTL
        "phi_10_std_ttl",         # sigma_TTL  [CRITICAL]
        "phi_11_mean_win",        # mu_win
        "phi_12_std_win",         # sigma_win
        "phi_13_n_syn",           # N_SYN
        "phi_14_n_urg",           # N_URG
        "phi_15_fin_ratio",       # R_FIN
        "phi_16_mean_hdr_len",    # mu_hd
        # Category 3: Traffic Symmetry
        "phi_17_pkt_ratio",       # R_pkt  [CRITICAL]
        "phi_18_byte_ratio",      # R_byte [CRITICAL]
        "phi_19_size_asymmetry",  # A_size
        "phi_20_resp_rate",       # lambda_resp
        # Category 4: Payload Dynamics
        "phi_21_mean_pkt_len",    # mu_len  [denominator only]
        "phi_22_std_pkt_len",     # sigma_len [CRITICAL]
        "phi_23_cv_pkt_len",      # CV_len
        "phi_24_small_pkt_ratio", # R_small
        "phi_25_large_pkt_ratio", # R_large
        "phi_26_hdr_pay_ratio",   # R_hd/pay
        # Category 5: Velocity
        "phi_27_pkt_rate",        # lambda_pkt
        "phi_28_byte_rate",       # lambda_byte
        "phi_29_fwd_byte_rate",   # lambda_fwd
        "phi_30_bwd_pkt_rate",    # lambda_bwd
    ]

    # Feature groups per Table I of the manuscript
    NOVEL_FEATURES = {2,6,7,8,10,12,15,16,18,19,20,23,24,25,26}  # 1-indexed
    NOVEL_ADV_FEATURES = {1,3,9,13,22}
    BASELINE_FEATURES = {4,5,11,14,17,21,27,28,29,30}
    CRITICAL_FEATURES = {2,10,17,18,22}  # provably infeasible to normalise

    def __init__(
        self,
        idle_timeout:    float = IDLE_TIMEOUT,
        burst_threshold: float = BURST_THRESH,
        max_flows:       int   = 100_000,
    ):
        self.idle_timeout    = idle_timeout
        self.burst_threshold = burst_threshold
        self.max_flows       = max_flows
        self._flow_table: Dict[tuple, FlowState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_packet(
        self,
        timestamp:   float,
        ip_src:      str,
        ip_dst:      str,
        port_src:    int,
        port_dst:    int,
        protocol:    int,
        pkt_length:  int,
        hdr_length:  int,
        ttl:         int,
        tcp_window:  int,
        flag_syn:    bool,
        flag_urg:    bool,
        flag_fin:    bool,
        direction:   str = "forward",   # "forward" or "backward"
    ) -> Optional[Tuple[tuple, np.ndarray]]:
        """
        Process one packet and update the corresponding flow state.

        Returns
        -------
        (flow_key, feature_vector) if the flow is complete, else None.
        feature_vector has shape (30,) matching FEATURE_NAMES.
        """
        key = (ip_src, ip_dst, port_src, port_dst, protocol)

        # Initialise new flow
        if key not in self._flow_table:
            if len(self._flow_table) >= self.max_flows:
                self._evict_oldest()
            s = FlowState()
            s.t_first    = timestamp
            s.burst_start = timestamp
            s.in_burst   = True
            self._flow_table[key] = s

        s = self._flow_table[key]
        s.n += 1

        # ------ Category 1: Time Dynamics ------
        if s.n > 1:
            iat = timestamp - s.t_last

            # Welford update for mu_IAT / sigma_IAT (phi_1, phi_2)
            s.mean_iat, s.m2_iat = _welford_update(
                s.n - 1, s.mean_iat, s.m2_iat, iat)

            # phi_3, phi_4
            s.iat_min = min(s.iat_min, iat)
            s.iat_max = max(s.iat_max, iat)

            # phi_6, phi_7: active/idle burst segmentation
            if iat <= self.burst_threshold:
                if not s.in_burst:
                    s.in_burst   = True
                    s.burst_start = timestamp
            else:
                if s.in_burst:
                    burst_dur = s.t_last - s.burst_start
                    s.n_active += 1
                    s.mean_t_active, s.m2_t_active = _welford_update(
                        s.n_active, s.mean_t_active, s.m2_t_active, burst_dur)
                s.n_idle += 1
                s.mean_t_idle, s.m2_t_idle = _welford_update(
                    s.n_idle, s.mean_t_idle, s.m2_t_idle, iat)
                s.in_burst = False

            # phi_8: forward-direction IAT
            if direction == "forward":
                s.n_fwd_iat += 1
                s.mean_fwd_iat, s.m2_fwd_iat = _welford_update(
                    s.n_fwd_iat, s.mean_fwd_iat, s.m2_fwd_iat, iat)

        s.t_last = timestamp

        # ------ Category 2: Header Invariants ------
        # phi_9, phi_10: TTL
        s.mean_ttl, s.m2_ttl = _welford_update(
            s.n, s.mean_ttl, s.m2_ttl, float(ttl))
        # phi_11, phi_12: window size
        s.mean_win, s.m2_win = _welford_update(
            s.n, s.mean_win, s.m2_win, float(tcp_window))
        # phi_13, phi_14, phi_15: flags
        if flag_syn: s.n_syn += 1
        if flag_urg: s.n_urg += 1
        if flag_fin: s.n_fin += 1
        # phi_16: header length
        s.mean_hdr, s.m2_hdr = _welford_update(
            s.n, s.mean_hdr, s.m2_hdr, float(hdr_length))

        # ------ Category 3: Traffic Symmetry ------
        if direction == "forward":
            s.n_fwd += 1
            s.b_fwd += pkt_length
        else:
            s.n_bwd += 1
            s.b_bwd += pkt_length

        # ------ Category 4: Payload Dynamics ------
        payload_len = max(0, pkt_length - hdr_length)
        s.mean_len, s.m2_len = _welford_update(
            s.n, s.mean_len, s.m2_len, float(pkt_length))
        if pkt_length < SMALL_PKT_THR: s.n_small += 1
        if pkt_length > LARGE_PKT_THR: s.n_large += 1
        s.sum_hdr += hdr_length
        s.sum_pay += payload_len

        # ------ Flow completion check ------
        flow_done = flag_fin or (timestamp - s.t_first > self.idle_timeout)
        if flow_done and s.n >= 2:
            features = self._extract(s)
            del self._flow_table[key]
            return key, features

        return None

    def flush_all(self) -> Dict[tuple, np.ndarray]:
        """
        Flush all active flows and extract features.
        Call at end of PCAP replay or experiment.
        """
        results = {}
        for key, s in list(self._flow_table.items()):
            if s.n >= 2:
                results[key] = self._extract(s)
        self._flow_table.clear()
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, s: FlowState) -> np.ndarray:
        """
        Compute all 30 features from Welford accumulators.
        Returns numpy array of shape (30,).
        """
        n   = s.n
        dur = max(s.t_last - s.t_first, 1e-9)   # avoid division by zero

        # --- Category 1: Time Dynamics ---
        phi_1 = s.mean_iat
        phi_2 = _welford_std(s.m2_iat, n - 1)   # sigma_IAT, n-1 denominator
        phi_3 = s.iat_min if s.iat_min != float('inf')  else 0.0
        phi_4 = s.iat_max if s.iat_max != float('-inf') else 0.0
        phi_5 = dur
        phi_6 = s.mean_t_active
        phi_7 = s.mean_t_idle
        phi_8 = s.mean_fwd_iat

        # --- Category 2: Header Invariants ---
        phi_9  = s.mean_ttl
        phi_10 = _welford_std(s.m2_ttl, n)
        phi_11 = s.mean_win
        phi_12 = _welford_std(s.m2_win, n)
        phi_13 = float(s.n_syn)
        phi_14 = float(s.n_urg)
        phi_15 = s.n_fin / n if n > 0 else 0.0
        phi_16 = s.mean_hdr

        # --- Category 3: Traffic Symmetry ---
        phi_17 = s.n_fwd / max(s.n_bwd, 1)
        phi_18 = s.b_fwd / max(s.b_bwd, 1)
        b_tot  = s.b_fwd + s.b_bwd
        phi_19 = (s.b_fwd - s.b_bwd) / max(b_tot, 1)  # A_size in [-1,1]
        phi_20 = s.n_bwd / dur                           # lambda_resp

        # --- Category 4: Payload Dynamics ---
        phi_21 = s.mean_len                              # denominator only
        phi_22 = _welford_std(s.m2_len, n)              # sigma_len [CRITICAL]
        phi_23 = phi_22 / max(phi_21, 1e-9)             # CV_len
        phi_24 = s.n_small / n
        phi_25 = s.n_large / n
        pay    = max(s.sum_pay, 1)
        phi_26 = s.sum_hdr / pay                         # R_hd/pay

        # --- Category 5: Velocity ---
        phi_27 = n / dur                                 # lambda_pkt
        phi_28 = (s.b_fwd + s.b_bwd) / dur             # lambda_byte
        phi_29 = s.b_fwd / dur                           # lambda_fwd
        phi_30 = s.n_bwd / dur                           # lambda_bwd

        return np.array([
            phi_1,  phi_2,  phi_3,  phi_4,  phi_5,
            phi_6,  phi_7,  phi_8,  phi_9,  phi_10,
            phi_11, phi_12, phi_13, phi_14, phi_15,
            phi_16, phi_17, phi_18, phi_19, phi_20,
            phi_21, phi_22, phi_23, phi_24, phi_25,
            phi_26, phi_27, phi_28, phi_29, phi_30,
        ], dtype=np.float64)

    def _evict_oldest(self) -> None:
        """Evict the oldest flow when table is full."""
        if self._flow_table:
            oldest_key = next(iter(self._flow_table))
            del self._flow_table[oldest_key]
