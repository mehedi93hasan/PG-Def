"""
PG-Def: Dataset Loader
=======================
Loads and preprocesses CICIDS2017, UNSW-NB15, and Edge-IIoTset
datasets for training and evaluation.

Preprocessing:
    - Maps CICFlowMeter columns to 30 protocol-grounded features
    - Handles class consolidation (manuscript Section VI-A)
    - Applies stratified 80:20 train-test split

Reference:
    PG-Def manuscript, Section VI-A: Datasets, Attacks, and Metrics
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Column mappings: CICFlowMeter -> PG-Def 30 features
# ---------------------------------------------------------------------------

CICIDS_COLUMN_MAP = {
    # Category 1: Time Dynamics
    "Flow IAT Mean":          "phi_01_mean_iat",
    "Flow IAT Std":           "phi_02_std_iat",
    "Flow IAT Min":           "phi_03_iat_min",
    "Flow IAT Max":           "phi_04_iat_max",
    "Flow Duration":          "phi_05_flow_duration",
    "Active Mean":            "phi_06_mean_t_active",
    "Idle Mean":              "phi_07_mean_t_idle",
    "Fwd IAT Mean":           "phi_08_fwd_mean_iat",
    # Category 2: Header Invariants (approximated from flow stats)
    "TTL":                    "phi_09_mean_ttl",
    "Fwd Header Length":      "phi_16_mean_hdr_len",
    "SYN Flag Count":         "phi_13_n_syn",
    "URG Flag Count":         "phi_14_n_urg",
    "FIN Flag Count":         "phi_15_fin_ratio",
    # Category 3: Traffic Symmetry
    "Fwd Packets/s":          "phi_27_pkt_rate",   # used to derive ratios
    "Bwd Packets/s":          "phi_30_bwd_pkt_rate",
    "Fwd Packet Length Mean": "phi_08_fwd_mean_iat",
    # Category 4: Payload Dynamics
    "Packet Length Mean":     "phi_21_mean_pkt_len",
    "Packet Length Std":      "phi_22_std_pkt_len",
    "Packet Length Variance": "_variance",          # used to derive CV
    # Category 5: Velocity
    "Flow Packets/s":         "phi_27_pkt_rate",
    "Flow Bytes/s":           "phi_28_byte_rate",
}

# Attack class consolidation (manuscript Section VI-A)
CICIDS_LABEL_MAP = {
    "BENIGN":               0,
    "DDoS":                 1,
    "DoS Hulk":             1,
    "DoS GoldenEye":        1,
    "DoS Slowloris":        1,
    "DoS Slowhttptest":     1,
    "Heartbleed":           1,
    "PortScan":             1,
    "Bot":                  1,
    "Infiltration":         1,
    "Web Attack Brute Force":   1,
    "Web Attack XSS":           1,
    "Web Attack Sql Injection": 1,
    "FTP-Patator":          1,
    "SSH-Patator":          1,
}

UNSWNB_LABEL_MAP = {
    "Normal": 0,
    "Fuzzers":        1, "Analysis":       1,
    "Backdoors":      1, "DoS":            1,
    "Exploits":       1, "Generic":        1,
    "Reconnaissance": 1, "Shellcode":      1,
    "Worms":          1,
}

EDGEIIOTSET_LABEL_MAP = {
    "Normal": 0,
    "DDoS_HTTP": 1, "DDoS_ICMP": 1, "DDoS_TCP":  1, "DDoS_UDP":  1,
    "Scanning":  1, "MITM":      1, "Injection":  1, "Backdoor":  1,
    "Password":  1, "Ransomware":1, "Uploading":  1, "XSS":       1,
    "Fingerprinting": 1, "Port_Scanning": 1,
}


class DatasetLoader:
    """
    Unified dataset loader for CICIDS2017, UNSW-NB15, and Edge-IIoTset.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility (default 42).
    test_size : float
        Fraction of data for testing (default 0.2).
    """

    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        self.random_state = random_state
        self.test_size    = test_size

    # ------------------------------------------------------------------
    # CICIDS2017
    # ------------------------------------------------------------------

    def load_cicids2017(self, data_dir: str) -> Tuple:
        """
        Load and preprocess CICIDS2017 dataset.

        Parameters
        ----------
        data_dir : path to directory containing CICIDS2017 CSV files

        Returns
        -------
        X_train, X_test, y_train, y_test : numpy arrays
        """
        print("[DataLoader] Loading CICIDS2017...")
        dfs = []
        for fname in os.listdir(data_dir):
            if fname.endswith(".csv"):
                path = os.path.join(data_dir, fname)
                try:
                    df = pd.read_csv(path, low_memory=False)
                    df.columns = df.columns.str.strip()
                    dfs.append(df)
                    print(f"  Loaded {fname}: {len(df):,} rows")
                except Exception as e:
                    print(f"  Warning: could not load {fname}: {e}")

        if not dfs:
            raise FileNotFoundError(
                f"No CSV files found in {data_dir}. "
                "Download CICIDS2017 from "
                "https://www.unb.ca/cic/datasets/ids-2017.html"
            )

        df = pd.concat(dfs, ignore_index=True)
        return self._process_cicids(df)

    def _process_cicids(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        label_col = " Label" if " Label" in df.columns else "Label"
        df[label_col] = df[label_col].str.strip()
        df["y"] = df[label_col].map(CICIDS_LABEL_MAP).fillna(1).astype(int)

        X = self._extract_cicids_features(df)
        y = df["y"].values

        X, y = self._clean(X, y)
        print(f"  Total: {len(y):,} flows | "
              f"Benign: {(y==0).sum():,} | "
              f"Malicious: {(y==1).sum():,}")

        return train_test_split(X, y, test_size=self.test_size,
                                stratify=y, random_state=self.random_state)

    def _extract_cicids_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and map 30 protocol-grounded features from CICFlowMeter."""
        feature_cols = [
            "Flow IAT Mean", "Flow IAT Std", "Flow IAT Min", "Flow IAT Max",
            "Flow Duration", "Active Mean", "Idle Mean", "Fwd IAT Mean",
            # Header invariants approximations
            "Fwd Header Length", "SYN Flag Count", "URG Flag Count",
            "FIN Flag Count",
            # Traffic symmetry
            "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets",
            # Payload
            "Packet Length Mean", "Packet Length Std",
            "Packet Length Variance",
            # Velocity
            "Flow Packets/s", "Flow Bytes/s",
            "Fwd Packets/s", "Bwd Packets/s",
        ]

        available = [c for c in feature_cols if c in df.columns]
        sub = df[available].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Build 30-feature matrix
        n = len(sub)
        X = np.zeros((n, 30), dtype=np.float64)

        def _col(name, default=0.0):
            return sub[name].values if name in sub.columns \
                else np.full(n, default)

        # Cat 1: Time Dynamics
        X[:, 0]  = _col("Flow IAT Mean")
        X[:, 1]  = _col("Flow IAT Std")
        X[:, 2]  = _col("Flow IAT Min")
        X[:, 3]  = _col("Flow IAT Max")
        X[:, 4]  = _col("Flow Duration")
        X[:, 5]  = _col("Active Mean")
        X[:, 6]  = _col("Idle Mean")
        X[:, 7]  = _col("Fwd IAT Mean")
        # Cat 2: Header Invariants (TTL not in CICFlowMeter; use 0)
        X[:, 8]  = 64.0          # default TTL
        X[:, 9]  = 0.0           # sigma_TTL not available in CICIDS
        X[:, 10] = 0.0           # mu_win not available
        X[:, 11] = 0.0
        X[:, 12] = _col("SYN Flag Count")
        X[:, 13] = _col("URG Flag Count")
        fwd_hdr  = _col("Fwd Header Length")
        X[:, 14] = (_col("FIN Flag Count") /
                    np.maximum(_col("Total Fwd Packets") +
                               _col("Total Backward Packets"), 1))
        X[:, 15] = fwd_hdr / np.maximum(
            _col("Total Fwd Packets"), 1)
        # Cat 3: Traffic Symmetry
        n_fwd = _col("Total Fwd Packets")
        n_bwd = _col("Total Backward Packets")
        b_fwd = _col("Total Length of Fwd Packets")
        b_bwd = _col("Total Length of Bwd Packets")
        X[:, 16] = n_fwd / np.maximum(n_bwd, 1)
        X[:, 17] = b_fwd / np.maximum(b_bwd, 1)
        b_tot    = b_fwd + b_bwd
        X[:, 18] = (b_fwd - b_bwd) / np.maximum(b_tot, 1)
        dur      = np.maximum(_col("Flow Duration"), 1e-9)
        X[:, 19] = n_bwd / dur
        # Cat 4: Payload
        X[:, 20] = _col("Packet Length Mean")
        X[:, 21] = _col("Packet Length Std")
        mu_len   = np.maximum(X[:, 20], 1e-9)
        X[:, 22] = X[:, 21] / mu_len
        X[:, 23] = 0.0   # R_small not in CICFlowMeter
        X[:, 24] = 0.0   # R_large not in CICFlowMeter
        X[:, 25] = fwd_hdr / np.maximum(
            b_fwd - fwd_hdr, 1)
        # Cat 5: Velocity
        X[:, 26] = _col("Flow Packets/s")
        X[:, 27] = _col("Flow Bytes/s")
        X[:, 28] = _col("Fwd Packets/s") * mu_len  # approx fwd bytes/s
        X[:, 29] = _col("Bwd Packets/s")
        return X

    # ------------------------------------------------------------------
    # UNSW-NB15
    # ------------------------------------------------------------------

    def load_unswnb15(self, data_dir: str) -> Tuple:
        """Load and preprocess UNSW-NB15 dataset."""
        print("[DataLoader] Loading UNSW-NB15...")
        dfs = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith(".csv"):
                path = os.path.join(data_dir, fname)
                try:
                    df = pd.read_csv(path, low_memory=False, header=None)
                    dfs.append(df)
                except Exception as e:
                    print(f"  Warning: {e}")

        if not dfs:
            raise FileNotFoundError(
                f"No CSV files found in {data_dir}. "
                "Download UNSW-NB15 from "
                "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
            )

        df = pd.concat(dfs, ignore_index=True)
        return self._process_generic(df, label_col=df.columns[-2],
                                     label_map=None, is_binary=True,
                                     binary_col=df.columns[-1])

    # ------------------------------------------------------------------
    # Edge-IIoTset
    # ------------------------------------------------------------------

    def load_edgeiiotset(self, csv_path: str) -> Tuple:
        """Load and preprocess Edge-IIoTset dataset."""
        print("[DataLoader] Loading Edge-IIoTset...")
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        label_col = "Attack_type" if "Attack_type" in df.columns else df.columns[-1]
        df["y"] = df[label_col].apply(
            lambda x: 0 if str(x).lower() in ["normal", "0"] else 1
        ).astype(int)
        X = self._extract_cicids_features(df)
        y = df["y"].values
        X, y = self._clean(X, y)
        print(f"  Total: {len(y):,} | Benign: {(y==0).sum():,} | "
              f"Malicious: {(y==1).sum():,}")
        return train_test_split(X, y, test_size=self.test_size,
                                stratify=y, random_state=self.random_state)

    # ------------------------------------------------------------------
    # Generic CSV loader (for custom datasets)
    # ------------------------------------------------------------------

    def load_csv(
        self,
        csv_path:  str,
        label_col: str,
        label_map: Optional[Dict] = None,
    ) -> Tuple:
        """
        Generic CSV loader for custom datasets with CICFlowMeter features.
        """
        print(f"[DataLoader] Loading {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()

        if label_map:
            df["y"] = df[label_col].map(label_map).fillna(1).astype(int)
        else:
            df["y"] = df[label_col].astype(int)

        X = self._extract_cicids_features(df)
        y = df["y"].values
        X, y = self._clean(X, y)
        return train_test_split(X, y, test_size=self.test_size,
                                stratify=y, random_state=self.random_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_generic(
        self, df, label_col, label_map, is_binary=False, binary_col=None
    ) -> Tuple:
        if is_binary and binary_col is not None:
            y = df[binary_col].astype(int).values
        elif label_map:
            y = df[label_col].map(label_map).fillna(1).astype(int).values
        else:
            y = df[label_col].astype(int).values

        X = self._extract_cicids_features(df)
        X, y = self._clean(X, y)
        return train_test_split(X, y, test_size=self.test_size,
                                stratify=y, random_state=self.random_state)

    @staticmethod
    def _clean(
        X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove inf, NaN, and align X/y."""
        mask = np.isfinite(X).all(axis=1)
        return X[mask], y[mask]
