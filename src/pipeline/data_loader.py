"""
PG-Def: Dataset Loader
=======================
Loads CICIDS2017, UNSW-NB15, and Edge-IIoTset CSV files
and maps available CICFlowMeter columns to the 30 protocol-grounded
features defined in Table I of the manuscript.

Feature Availability from CICFlowMeter CSV:
--------------------------------------------
AVAILABLE (26/30):
    phi_1  -- Flow IAT Mean
    phi_2  -- Flow IAT Std         [CRITICAL]
    phi_3  -- Flow IAT Min
    phi_4  -- Flow IAT Max
    phi_5  -- Flow Duration
    phi_6  -- Active Mean
    phi_7  -- Idle Mean
    phi_8  -- Fwd IAT Mean
    phi_13 -- SYN Flag Count
    phi_14 -- URG Flag Count
    phi_15 -- FIN Flag Count / Total packets
    phi_16 -- Fwd Header Length / Fwd packets
    phi_17 -- Total Fwd Packets / Total Bwd Packets  [CRITICAL]
    phi_18 -- Fwd bytes / Bwd bytes                  [CRITICAL]
    phi_19 -- (Fwd-Bwd bytes) / (Fwd+Bwd bytes)
    phi_20 -- Total Bwd Packets / Flow Duration
    phi_21 -- Packet Length Mean
    phi_22 -- Packet Length Std    [CRITICAL]
    phi_23 -- Packet Length Std / Packet Length Mean
    phi_26 -- Fwd Header bytes / Payload bytes
    phi_27 -- Flow Packets/s
    phi_28 -- Flow Bytes/s
    phi_29 -- Fwd Packets/s * mean_len (approx fwd bytes/s)
    phi_30 -- Bwd Packets/s

NOT IN CICFlowMeter CSV (4/30) -- require PCAP-level extraction:
    phi_9  -- mu_TTL      (set to OS-default 64; RFC 1700)
    phi_10 -- sigma_TTL   (set to 0; not extractable from CSV)
    phi_11 -- mu_win      (set to 0; not in CICFlowMeter)
    phi_12 -- sigma_win   (set to 0; not in CICFlowMeter)

NOT EXPORTED by CICFlowMeter (2/30):
    phi_24 -- R_small (<64 byte packets ratio) -- set to 0
    phi_25 -- R_large (>1200 byte packets ratio) -- set to 0

NOTE: phi_9, phi_10, phi_11, phi_12, phi_24, phi_25 are set to their
neutral values. For full 30-feature extraction, use the Welford
extractor (src/features/welford_extractor.py) with raw PCAP files.

Reference:
    PG-Def manuscript, Section VI-A: Datasets, Attacks, and Metrics
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Attack class consolidation (manuscript Section VI-A)
# ---------------------------------------------------------------------------

CICIDS_LABEL_MAP = {
    "BENIGN":                         0,
    "DDoS":                           1,
    "DoS Hulk":                       1,
    "DoS GoldenEye":                  1,
    "DoS Slowloris":                  1,
    "DoS Slowhttptest":               1,
    "Heartbleed":                     1,
    "PortScan":                       1,
    "Bot":                            1,
    "Infiltration":                   1,
    "Web Attack \x00Brute Force":     1,
    "Web Attack \x00XSS":             1,
    "Web Attack \x00Sql Injection":   1,
    "Web Attack  Brute Force":        1,
    "Web Attack  XSS":                1,
    "Web Attack  Sql Injection":      1,
    "Web Attack Brute Force":         1,
    "Web Attack XSS":                 1,
    "Web Attack Sql Injection":       1,
    "FTP-Patator":                    1,
    "SSH-Patator":                    1,
}

UNSWNB_LABEL_MAP = {
    "Normal":         0,
    "Fuzzers":        1, "Analysis":       1,
    "Backdoors":      1, "DoS":            1,
    "Exploits":       1, "Generic":        1,
    "Reconnaissance": 1, "Shellcode":      1,
    "Worms":          1,
}

EDGEIIOTSET_LABEL_MAP = {
    "Normal":   0, "normal": 0, "0": 0,
    "DDoS_HTTP":1, "DDoS_ICMP":1, "DDoS_TCP": 1, "DDoS_UDP": 1,
    "Scanning": 1, "MITM":     1, "Injection": 1, "Backdoor": 1,
    "Password": 1, "Ransomware":1, "Uploading": 1, "XSS":     1,
    "Fingerprinting": 1, "Port_Scanning": 1,
}


class DatasetLoader:
    """
    Unified dataset loader for CICIDS2017, UNSW-NB15, and Edge-IIoTset.

    Maps CICFlowMeter columns to the 30 protocol-grounded features
    (Table I of the manuscript). Features not available in CSV format
    (phi_9 mu_TTL, phi_10 sigma_TTL, phi_11 mu_win, phi_12 sigma_win,
    phi_24 R_small, phi_25 R_large) are set to neutral values with a
    clear warning. Full extraction of all 30 features requires
    PCAP-level processing via src/features/welford_extractor.py.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility (default 42).
    test_size : float
        Fraction for test set (default 0.2, per manuscript Section VI-A).
    """

    # CICFlowMeter column names (with variants for different versions)
    _IAT_MEAN    = ["Flow IAT Mean",    " Flow IAT Mean"]
    _IAT_STD     = ["Flow IAT Std",     " Flow IAT Std"]
    _IAT_MIN     = ["Flow IAT Min",     " Flow IAT Min"]
    _IAT_MAX     = ["Flow IAT Max",     " Flow IAT Max"]
    _DURATION    = ["Flow Duration",    " Flow Duration"]
    _ACTIVE_MEAN = ["Active Mean",      " Active Mean"]
    _IDLE_MEAN   = ["Idle Mean",        " Idle Mean"]
    _FWD_IAT     = ["Fwd IAT Mean",     " Fwd IAT Mean",
                    "Fwd IAT Mean ",    "Fwd IAT Total"]
    _SYN         = ["SYN Flag Count",   " SYN Flag Count"]
    _URG         = ["URG Flag Count",   " URG Flag Count"]
    _FIN         = ["FIN Flag Count",   " FIN Flag Count"]
    _FWD_HDR     = ["Fwd Header Length"," Fwd Header Length"]
    _N_FWD       = ["Total Fwd Packets"," Total Fwd Packets"]
    _N_BWD       = ["Total Backward Packets", " Total Backward Packets",
                    "Total Bwd Packets"]
    _B_FWD       = ["Total Length of Fwd Packets",
                    " Total Length of Fwd Packets",
                    "Fwd Packet Length Sum"]
    _B_BWD       = ["Total Length of Bwd Packets",
                    " Total Length of Bwd Packets",
                    "Bwd Packet Length Sum"]
    _PKT_MEAN    = ["Packet Length Mean",  " Packet Length Mean"]
    _PKT_STD     = ["Packet Length Std",   " Packet Length Std"]
    _FLOW_PPS    = ["Flow Packets/s",      " Flow Packets/s"]
    _FLOW_BPS    = ["Flow Bytes/s",        " Flow Bytes/s"]
    _FWD_PPS     = ["Fwd Packets/s",       " Fwd Packets/s"]
    _BWD_PPS     = ["Bwd Packets/s",       " Bwd Packets/s"]

    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        self.random_state = random_state
        self.test_size    = test_size

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_cicids2017(self, data_dir: str) -> Tuple:
        """
        Load CICIDS2017 dataset from directory of CICFlowMeter CSV files.

        Download from:
            https://www.unb.ca/cic/datasets/ids-2017.html

        Parameters
        ----------
        data_dir : str
            Path to directory containing CICIDS2017 CSV files.

        Returns
        -------
        X_train, X_test, y_train, y_test : numpy arrays
            X shape: (n_samples, 30) -- 30 protocol-grounded features
            y shape: (n_samples,)    -- binary {0=benign, 1=malicious}
        """
        print("[DataLoader] Loading CICIDS2017...")
        dfs = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.lower().endswith(".csv"):
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
                f"No CSV files found in {data_dir}.\n"
                "Download CICIDS2017 from:\n"
                "https://www.unb.ca/cic/datasets/ids-2017.html"
            )

        df = pd.concat(dfs, ignore_index=True)

        # Map labels
        label_col = "Label" if "Label" in df.columns else " Label"
        df[label_col] = df[label_col].astype(str).str.strip()
        df["y"] = df[label_col].map(CICIDS_LABEL_MAP)
        df["y"] = df["y"].fillna(1).astype(int)

        X = self._extract_features(df)
        y = df["y"].values
        X, y = self._clean(X, y)

        self._print_summary("CICIDS2017", y)
        return train_test_split(X, y, test_size=self.test_size,
                                stratify=y, random_state=self.random_state)

    def load_unswnb15(self, data_dir: str) -> Tuple:
        """
        Load UNSW-NB15 dataset from directory of CSV files.

        Download from:
            https://research.unsw.edu.au/projects/unsw-nb15-dataset

        Parameters
        ----------
        data_dir : str
            Path to directory containing UNSW_NB15_*.csv files.
        """
        print("[DataLoader] Loading UNSW-NB15...")
        dfs = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.lower().endswith(".csv"):
                path = os.path.join(data_dir, fname)
                try:
                    # UNSW-NB15 has no header in some versions
                    df = pd.read_csv(path, low_memory=False)
                    df.columns = df.columns.str.strip()
                    dfs.append(df)
                    print(f"  Loaded {fname}: {len(df):,} rows")
                except Exception as e:
                    print(f"  Warning: {e}")

        if not dfs:
            raise FileNotFoundError(
                f"No CSV files found in {data_dir}.\n"
                "Download UNSW-NB15 from:\n"
                "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
            )

        df = pd.concat(dfs, ignore_index=True)

        # Binary label: column 'label' or last column
        if "label" in df.columns:
            df["y"] = df["label"].astype(int)
        elif "Label" in df.columns:
            df["y"] = df["Label"].map(UNSWNB_LABEL_MAP).fillna(1).astype(int)
        else:
            df["y"] = df.iloc[:, -1].astype(int)

        X = self._extract_features(df)
        y = df["y"].values
        X, y = self._clean(X, y)
        self._print_summary("UNSW-NB15", y)
        return train_test_split(X, y, test_size=self.test_size,
                                stratify=y, random_state=self.random_state)

    def load_edgeiiotset(self, csv_path: str) -> Tuple:
        """
        Load Edge-IIoTset dataset from a single CSV file.

        Download from:
            https://ieee-dataport.org/documents/edge-iiotset

        Parameters
        ----------
        csv_path : str
            Path to Edge-IIoTset CSV file.
        """
        print("[DataLoader] Loading Edge-IIoTset...")
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()

        label_col = next((c for c in ["Attack_type", "label", "Label",
                                       "attack_type"] if c in df.columns),
                          df.columns[-1])
        df["y"] = df[label_col].astype(str).str.strip().map(
            lambda x: 0 if x.lower() in ["normal", "0"] else 1
        ).astype(int)

        X = self._extract_features(df)
        y = df["y"].values
        X, y = self._clean(X, y)
        self._print_summary("Edge-IIoTset", y)
        return train_test_split(X, y, test_size=self.test_size,
                                stratify=y, random_state=self.random_state)

    def load_csv(
        self,
        csv_path:  str,
        label_col: str,
        label_map: Optional[Dict] = None,
    ) -> Tuple:
        """
        Generic loader for any CICFlowMeter-format CSV file.

        Parameters
        ----------
        csv_path  : str   path to CSV file
        label_col : str   name of label column
        label_map : dict  optional string->int mapping
        """
        print(f"[DataLoader] Loading {os.path.basename(csv_path)}...")
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()

        if label_map:
            df["y"] = df[label_col].astype(str).str.strip() \
                                    .map(label_map).fillna(1).astype(int)
        else:
            df["y"] = df[label_col].astype(int)

        X = self._extract_features(df)
        y = df["y"].values
        X, y = self._clean(X, y)
        self._print_summary(os.path.basename(csv_path), y)
        return train_test_split(X, y, test_size=self.test_size,
                                stratify=y, random_state=self.random_state)

    # ------------------------------------------------------------------
    # Core feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract 30 protocol-grounded features from a CICFlowMeter
        DataFrame. Each phi maps directly to Table I of the manuscript.

        Features requiring PCAP-level extraction (phi_9, phi_10, phi_11,
        phi_12, phi_24, phi_25) are set to their neutral/default values
        and flagged in the feature availability summary.
        """
        n = len(df)
        X = np.zeros((n, 30), dtype=np.float64)

        def col(candidates, default=0.0):
            """Return first matching column as numpy array."""
            for name in candidates:
                if name in df.columns:
                    return pd.to_numeric(
                        df[name], errors="coerce"
                    ).fillna(default).values
            return np.full(n, default)

        # ----------------------------------------------------------------
        # Category 1: Time Dynamics (phi_1 -- phi_8)
        # ----------------------------------------------------------------
        X[:, 0]  = col(self._IAT_MEAN)        # phi_1  mu_IAT
        X[:, 1]  = col(self._IAT_STD)         # phi_2  sigma_IAT [CRITICAL]
        X[:, 2]  = col(self._IAT_MIN)         # phi_3  IAT_min
        X[:, 3]  = col(self._IAT_MAX)         # phi_4  IAT_max
        X[:, 4]  = col(self._DURATION)        # phi_5  T_flow
        X[:, 5]  = col(self._ACTIVE_MEAN)     # phi_6  mu_T_active
        X[:, 6]  = col(self._IDLE_MEAN)       # phi_7  mu_T_idle
        X[:, 7]  = col(self._FWD_IAT)         # phi_8  mu_fwd_IAT

        # ----------------------------------------------------------------
        # Category 2: Header Invariants (phi_9 -- phi_16)
        # phi_9, phi_10, phi_11, phi_12 NOT in CICFlowMeter CSV.
        # Set to protocol-default values with explicit note.
        # ----------------------------------------------------------------
        # phi_9: mu_TTL -- default 64 (Linux OS, RFC 1700)
        # NOTE: sigma_TTL (phi_10) cannot be extracted from CSV;
        #       requires per-packet TTL from PCAP.
        X[:, 8]  = 64.0    # phi_9  mu_TTL    (OS default Linux, RFC 1700)
        X[:, 9]  = 0.0     # phi_10 sigma_TTL (PCAP required)
        X[:, 10] = 0.0     # phi_11 mu_win    (PCAP required)
        X[:, 11] = 0.0     # phi_12 sigma_win (PCAP required)

        X[:, 12] = col(self._SYN)             # phi_13 N_SYN
        X[:, 13] = col(self._URG)             # phi_14 N_URG

        # phi_15: R_FIN = FIN count / total packets
        n_fwd   = col(self._N_FWD, 1.0)
        n_bwd   = col(self._N_BWD, 0.0)
        n_total = np.maximum(n_fwd + n_bwd, 1.0)
        fin_cnt = col(self._FIN)
        X[:, 14] = fin_cnt / n_total          # phi_15 R_FIN

        # phi_16: mu_hd = mean header length
        fwd_hdr  = col(self._FWD_HDR)
        X[:, 15] = fwd_hdr / np.maximum(n_fwd, 1.0)  # phi_16 mu_hd

        # ----------------------------------------------------------------
        # Category 3: Traffic Symmetry (phi_17 -- phi_20)
        # ----------------------------------------------------------------
        b_fwd    = col(self._B_FWD)
        b_bwd    = col(self._B_BWD)
        dur      = np.maximum(col(self._DURATION), 1e-9)

        X[:, 16] = n_fwd / np.maximum(n_bwd, 1.0)    # phi_17 R_pkt [CRITICAL]
        X[:, 17] = b_fwd / np.maximum(b_bwd, 1.0)    # phi_18 R_byte [CRITICAL]
        b_tot    = b_fwd + b_bwd
        X[:, 18] = (b_fwd - b_bwd) / np.maximum(b_tot, 1.0)  # phi_19 A_size
        X[:, 19] = n_bwd / dur                        # phi_20 lambda_resp

        # ----------------------------------------------------------------
        # Category 4: Payload Dynamics (phi_21 -- phi_26)
        # phi_24, phi_25 (R_small, R_large) not in CICFlowMeter.
        # ----------------------------------------------------------------
        mu_len   = col(self._PKT_MEAN)
        sigma_len = col(self._PKT_STD)

        X[:, 20] = mu_len                             # phi_21 mu_len (denominator)
        X[:, 21] = sigma_len                          # phi_22 sigma_len [CRITICAL]
        X[:, 22] = sigma_len / np.maximum(mu_len, 1e-9)  # phi_23 CV_len
        X[:, 23] = 0.0     # phi_24 R_small (CICFlowMeter does not export)
        X[:, 24] = 0.0     # phi_25 R_large (CICFlowMeter does not export)

        # phi_26: R_hd/pay = header bytes / payload bytes
        hdr_sum  = fwd_hdr * n_fwd
        pay_sum  = np.maximum(b_fwd - hdr_sum, 1.0)
        X[:, 25] = hdr_sum / pay_sum                  # phi_26 R_hd/pay

        # ----------------------------------------------------------------
        # Category 5: Velocity (phi_27 -- phi_30)
        # ----------------------------------------------------------------
        X[:, 26] = col(self._FLOW_PPS)                # phi_27 lambda_pkt
        X[:, 27] = col(self._FLOW_BPS)                # phi_28 lambda_byte
        fwd_pps  = col(self._FWD_PPS)
        X[:, 28] = fwd_pps * np.maximum(mu_len, 1.0) # phi_29 lambda_fwd (approx)
        X[:, 29] = col(self._BWD_PPS)                 # phi_30 lambda_bwd

        return X

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(X: np.ndarray, y: np.ndarray) -> Tuple:
        """Remove inf/NaN rows and align X, y."""
        mask = np.isfinite(X).all(axis=1)
        return X[mask], y[mask]

    @staticmethod
    def _print_summary(name: str, y: np.ndarray) -> None:
        print(f"  {name}: {len(y):,} flows | "
              f"Benign: {(y==0).sum():,} | "
              f"Malicious: {(y==1).sum():,}")
        print(f"  Feature availability note: phi_9/phi_10 (TTL), "
              f"phi_11/phi_12 (window), phi_24/phi_25 (pkt size ratios) "
              f"require PCAP-level extraction (set to default values).")

    def feature_availability_report(self) -> None:
        """Print which of the 30 features are available from CSV."""
        print("\nPG-Def Feature Availability from CICFlowMeter CSV")
        print("=" * 58)
        availability = {
            "phi_1  mu_IAT":          ("Flow IAT Mean",             True),
            "phi_2  sigma_IAT*":      ("Flow IAT Std",              True),
            "phi_3  IAT_min":         ("Flow IAT Min",              True),
            "phi_4  IAT_max":         ("Flow IAT Max",              True),
            "phi_5  T_flow":          ("Flow Duration",             True),
            "phi_6  mu_T_active":     ("Active Mean",               True),
            "phi_7  mu_T_idle":       ("Idle Mean",                 True),
            "phi_8  mu_fwd_IAT":      ("Fwd IAT Mean",              True),
            "phi_9  mu_TTL":          ("PCAP only",                 False),
            "phi_10 sigma_TTL*":      ("PCAP only",                 False),
            "phi_11 mu_win":          ("PCAP only",                 False),
            "phi_12 sigma_win":       ("PCAP only",                 False),
            "phi_13 N_SYN":           ("SYN Flag Count",            True),
            "phi_14 N_URG":           ("URG Flag Count",            True),
            "phi_15 R_FIN":           ("FIN Flag Count / n",        True),
            "phi_16 mu_hd":           ("Fwd Header Length / n_fwd", True),
            "phi_17 R_pkt*":          ("Fwd/Bwd packet counts",     True),
            "phi_18 R_byte*":         ("Fwd/Bwd byte totals",       True),
            "phi_19 A_size":          ("Derived from bytes",        True),
            "phi_20 lambda_resp":     ("Bwd packets / duration",    True),
            "phi_21 mu_len":          ("Packet Length Mean",        True),
            "phi_22 sigma_len*":      ("Packet Length Std",         True),
            "phi_23 CV_len":          ("Std / Mean",                True),
            "phi_24 R_small":         ("Not in CICFlowMeter",       False),
            "phi_25 R_large":         ("Not in CICFlowMeter",       False),
            "phi_26 R_hd/pay":        ("Derived from headers/bytes",True),
            "phi_27 lambda_pkt":      ("Flow Packets/s",            True),
            "phi_28 lambda_byte":     ("Flow Bytes/s",              True),
            "phi_29 lambda_fwd":      ("Fwd Packets/s * mean_len",  True),
            "phi_30 lambda_bwd":      ("Bwd Packets/s",             True),
        }
        available = sum(1 for _, (_, a) in availability.items() if a)
        for feat, (src, avail) in availability.items():
            status = "YES" if avail else " NO"
            print(f"  {status}  {feat:<22} <- {src}")
        print(f"\n  Available from CSV: {available}/30 features")
        print(f"  Requires PCAP:      {30-available}/30 features")
        print("  * = CRITICAL feature (formally proven infeasible to normalise)")
