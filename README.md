# PG-Def: Protocol-Grounded Defense Framework

> **PG-Def: A Protocol-Grounded Lightweight Defense Framework for Adversarially Robust Network Intrusion Detection**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![IEEE Standard](https://img.shields.io/badge/Standard-IEEE%20Top--Tier-green.svg)]()

---

## Overview

PG-Def is a **protocol-grounded lightweight defense framework** that achieves certifiable adversarial robustness for Network Intrusion Detection Systems (NIDS) without the computational cost or cross-domain fragility of deep learning approaches.

**Key results from the manuscript:**

| Metric | PG-Def | Best DL Baseline |
|--------|--------|-----------------|
| Memory footprint | **27.8 MB** | 770 MB (FA-CNN) |
| Latency per flow | **0.8 ms** | 175.6 ms (FA-CNN) |
| Throughput | **1,250 flows/s** | 5.7 flows/s (FA-CNN) |
| Adversarial TPR (avg) | **91.1%** | 81.4% (Adv. Retrain) |
| Cross-domain TPR drop | **11.4%** | 23–32% (all DL) |
| Edge deployment | **833 flows/s @ 1.3 W** | Not deployable |

---

## Architecture

PG-Def operates through four sequential tiers:

```
Raw Packets
    │
    ▼
┌─────────────────────────────────┐
│  Tier 1: Flow Aggregation       │
│  eBPF kernel-space capture      │
│  MurmurHash3 flow table         │
│  100,000 concurrent flows       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Tier 2: Feature Extraction     │
│  Welford Online Algorithm       │
│  30 protocol-grounded features  │
│  O(1) memory per flow           │
│  128 bytes per-flow state       │
└─────────────────────────────────┘
    │  F ∈ R^30
    ▼
┌─────────────────────────────────┐
│  Tier 3: Ensemble Classification│
│  Random Forest  (w=0.4, T=50)  │
│  XGBoost        (w=0.4, M=100) │
│  Logistic Reg.  (w=0.2, L2)    │
│  Weighted soft-voting           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Tier 4: Adaptive Defense       │
│  C1: Attack Fingerprint Cache   │
│  C2: Confidence Monitor         │
│  C3: Dynamic Threshold Adapt.   │
│  C4: Feature Drift Monitor      │
│  Total overhead: ≤8 KB, <0.05ms │
└─────────────────────────────────┘
    │
    ▼
  BENIGN / MALICIOUS
```

---

## The 30 Protocol-Grounded Features

All features are extracted from immutable TCP/IP protocol fields (Principles 1–3):

| Category | Features | Critical Features |
|----------|----------|-------------------|
| Time Dynamics (φ₁–φ₈) | μ_IAT, **σ_IAT**, IAT_min, IAT_max, T_flow, μ_T_active, μ_T_idle, μ_fwd_IAT | **σ_IAT (φ₂)** |
| Header Invariants (φ₉–φ₁₆) | μ_TTL, **σ_TTL**, μ_win, σ_win, N_SYN, N_URG, R_FIN, μ_hd | **σ_TTL (φ₁₀)** |
| Traffic Symmetry (φ₁₇–φ₂₀) | **R_pkt**, **R_byte**, A_size, λ_resp | **R_pkt (φ₁₇), R_byte (φ₁₈)** |
| Payload Dynamics (φ₂₁–φ₂₆) | μ_len†, **σ_len**, CV_len, R_small, R_large, R_hd/pay | **σ_len (φ₂₂)** |
| Velocity (φ₂₇–φ₃₀) | λ_pkt, λ_byte, λ_fwd, λ_bwd | — |

> **†** φ₂₁ (μ_len) serves as normalisation denominator only; not individually claimed under Principle 2.

**Feature Groups:**
- **N** (Novel, 15 features): φ₂, φ₆–φ₈, φ₁₀, φ₁₂, φ₁₅, φ₁₆, φ₁₈–φ₂₀, φ₂₃–φ₂₆
- **NA** (Novel Adversarial Analysis, 5 features): φ₁, φ₃, φ₉, φ₁₃, φ₂₂
- **B** (Baseline, 10 features): φ₄, φ₅, φ₁₁, φ₁₄, φ₁₇, φ₂₁, φ₂₇–φ₃₀

---

## Formal Guarantees

**Theorem 1 (Jitter–Symmetry Co-Evasion):**
Simultaneous normalisation of σ_IAT ≥ 0.1 s and R_pkt ∈ [0.5, 2.0] for a volumetric DDoS attack with λ_pkt > 1,000 pps and T_attack ≤ 60 s reduces attack volume N by **at least 95%**.

**Corollary 1 (Multi-Feature Infeasibility):**
Under |Botnet| > 100, simultaneous normalisation of {φ₂, φ₁₀, φ₁₇} forces attack effectiveness degradation of **at least 95%** — an environment-independent guarantee intrinsic to the TCP/IP protocol stack.

---

## Installation

```bash
git clone https://github.com/your-username/pgdef.git
cd pgdef
pip install -r requirements.txt
```

**Optional: adversarial evaluation**
```bash
pip install adversarial-robustness-toolbox
```

---

## Datasets

PG-Def is evaluated on three benchmark datasets:

| Dataset | Flows | Environment | Download |
|---------|-------|-------------|----------|
| CICIDS2017 | 2,830,743 | Enterprise | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) |
| UNSW-NB15 | 2,540,044 | Academic / Cyber Range | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| Edge-IIoTset | 1,194,464 | IoT / IIoT | [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot) |

**Directory structure:**
```
data/
├── cicids2017/
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   └── ...
├── unswnb15/
│   ├── UNSW-NB15_1.csv
│   └── ...
└── edgeiiotset/
    └── Edge-IIoTset.csv
```

---

## Usage

### Train and evaluate on CICIDS2017

```bash
python experiments/run_experiments.py \
    --dataset cicids2017 \
    --data_dir data/cicids2017 \
    --mode train_eval
```

### Adversarial robustness evaluation

```bash
python experiments/run_experiments.py \
    --dataset cicids2017 \
    --data_dir data/cicids2017 \
    --mode adversarial
```

### Cross-domain generalisation (3×3 transfer matrix)

```bash
python experiments/run_experiments.py \
    --dataset cicids2017 \
    --data_dir data \
    --mode cross_domain
```

### Disable adaptive defense (base PG-Def only)

```bash
python experiments/run_experiments.py \
    --dataset cicids2017 \
    --data_dir data/cicids2017 \
    --mode train_eval \
    --no_adaptive
```

---

## Python API

```python
from src.pipeline.pgdef_pipeline import PGDefPipeline
from src.pipeline.data_loader import DatasetLoader
import numpy as np

# Load data
loader = DatasetLoader()
X_train, X_test, y_train, y_test = loader.load_cicids2017("data/cicids2017")

# Train
pipeline = PGDefPipeline(use_adaptive=True)
cv_results = pipeline.train(X_train, y_train)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test, dataset_name="CICIDS2017")
print(f"TPR: {metrics['tpr']:.4f}  FPR: {metrics['fpr']:.4f}")
print(f"Memory: {pipeline.memory_footprint_mb():.1f} MB")

# Save
pipeline.save()
```

### Use the Welford feature extractor directly

```python
from src.features.welford_extractor import ProtocolGroundedExtractor

extractor = ProtocolGroundedExtractor()

# Process individual packets
result = extractor.process_packet(
    timestamp=1000.025,
    ip_src="192.168.1.10", ip_dst="10.0.0.5",
    port_src=54321, port_dst=80,
    protocol=6,
    pkt_length=512, hdr_length=20,
    ttl=64, tcp_window=65535,
    flag_syn=False, flag_urg=False, flag_fin=True,
    direction="forward",
)

if result:
    flow_key, features = result
    print(f"Flow {flow_key}: F ∈ R^{len(features)}")
    print(f"  sigma_IAT (phi_2): {features[1]:.4f}")
    print(f"  sigma_TTL (phi_10): {features[9]:.4f}")
    print(f"  R_pkt    (phi_17): {features[16]:.4f}")

# Flush remaining flows
remaining = extractor.flush_all()
```

### Use the adaptive defense system

```python
from src.defense.adaptive_defense import AdaptiveDefenseSystem
from src.models.ensemble import PGDefEnsemble

# Load trained ensemble
ensemble = PGDefEnsemble.load("models/pgdef")

# Create adaptive defense
ads = AdaptiveDefenseSystem(
    ensemble,
    tau_conf=0.75,
    tau_vote=0.50,
    fpr_target=0.02,
)

# Classify a flow
features = np.zeros(30)   # replace with real features
pred, confidence, component = ads.classify(features)
print(f"Prediction: {'MALICIOUS' if pred else 'BENIGN'}")
print(f"Confidence: {confidence:.4f}")
print(f"Component:  {component}")

# Print statistics
print(ads.summary())
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test class
pytest tests/test_pgdef.py::TestWelfordUpdate -v
```

**Expected test output:**
```
tests/test_pgdef.py::TestWelfordUpdate::test_mean_correctness PASSED
tests/test_pgdef.py::TestWelfordUpdate::test_std_correctness PASSED
tests/test_pgdef.py::TestWelfordUpdate::test_n1_denominator PASSED
tests/test_pgdef.py::TestWelfordUpdate::test_single_value_returns_zero_std PASSED
tests/test_pgdef.py::TestWelfordUpdate::test_o1_memory PASSED
tests/test_pgdef.py::TestProtocolGroundedExtractor::... PASSED
tests/test_pgdef.py::TestBloomFilter::... PASSED
tests/test_pgdef.py::TestPGDefEnsemble::... PASSED
tests/test_pgdef.py::TestAdaptiveDefense::... PASSED
tests/test_pgdef.py::TestPipelineMemory::... PASSED
```

---

## Project Structure

```
pgdef/
├── src/
│   ├── features/
│   │   └── welford_extractor.py   # Tier 2: Algorithm 1
│   ├── models/
│   │   └── ensemble.py            # Tier 3: RF+XGB+LR
│   ├── defense/
│   │   └── adaptive_defense.py    # Tier 4: Algorithm 2
│   └── pipeline/
│       ├── pgdef_pipeline.py      # Full pipeline
│       ├── data_loader.py         # Dataset loader
│       └── adversarial_eval.py    # Adversarial evaluation
├── experiments/
│   └── run_experiments.py         # Main experiment runner
├── tests/
│   └── test_pgdef.py              # Unit tests
├── data/                          # Dataset directory (not included)
├── models/                        # Saved models (generated)
├── results/                       # Experiment results (generated)
├── requirements.txt
└── README.md
```

---

## Experimental Results

Results from the manuscript (Section VI):

### Clean-Data Detection Performance

| Method | CIC TPR | UNSW TPR | Edge TPR | Memory |
|--------|---------|----------|----------|--------|
| FA-CNN | 99.7% | 97.8% | 99.1% | 770 MB |
| GTAE-IDS | 99.5% | 97.1% | 98.6% | 470 MB |
| Adv. Retrain | 98.4% | 96.3% | 97.8% | 830 MB |
| **PG-Def (ours)** | **97.9%** | **95.6%** | **96.8%** | **27.8 MB** |

### Adversarial Robustness (TPR %)

| Method | FGSM | BIM | C&W-L∞ | SAAE | Avg |
|--------|------|-----|--------|------|-----|
| FA-CNN | 82.4 | 79.1 | 75.3 | 94.4 | 81.4 |
| GTAE-IDS | 68.3 | 64.7 | 59.8 | 52.3 | 63.1 |
| **PG-Def** | **93.8** | **93.1** | **92.1** | **91.7** | **93.0** |

### Cross-Domain Generalisation (Relative TPR Drop)

| Method | Same-Domain | Cross-Domain | Rel. Drop |
|--------|-------------|--------------|-----------|
| FA-CNN | 98.9% | 67.3% | 32.0% |
| GTAE-IDS | 98.4% | 71.1% | 27.7% |
| **PG-Def** | **96.8%** | **85.8%** | **11.4%** |

---

## Citation

If you use PG-Def in your research, please cite:

```bibtex
@article{pgdef2025,
  title   = {PG-Def: A Protocol-Grounded Lightweight Defense Framework
             for Adversarially Robust Network Intrusion Detection},
  author  = {Hasan, Mehedi and Islam, Rafiqul and Mamun, Quazi},
  journal = {IEEE Transactions on Network and Service Management},
  year    = {2025},
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

This research was conducted at the Connectivity Innovation Network (CIN),
Charles Sturt University, Albury, NSW, Australia.

Supervisors: A/Prof. Rafiqul Islam and A/Prof. Quazi Mamun.
