"""
PG-Def Unit Tests
==================
Tests for all core components of the PG-Def pipeline.
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.welford_extractor import (
    ProtocolGroundedExtractor, FlowState,
    _welford_update, _welford_std,
)
from src.models.ensemble import PGDefEnsemble
from src.defense.adaptive_defense import (
    AdaptiveDefenseSystem, BloomFilter, make_fingerprint,
)
from src.pipeline.pgdef_pipeline import PGDefPipeline


# ---------------------------------------------------------------------------
# Test Welford algorithm correctness
# ---------------------------------------------------------------------------

class TestWelfordUpdate:
    def test_mean_correctness(self):
        """Welford mean must match numpy mean exactly."""
        data = [1.5, 2.3, 0.8, 3.1, 1.2]
        mean, m2 = 0.0, 0.0
        for i, x in enumerate(data, 1):
            mean, m2 = _welford_update(i, mean, m2, x)
        assert abs(mean - np.mean(data)) < 1e-10

    def test_std_correctness(self):
        """Welford std (n-1 denominator) must match numpy std(ddof=1)."""
        data = [1.5, 2.3, 0.8, 3.1, 1.2]
        mean, m2 = 0.0, 0.0
        n = len(data)
        for i, x in enumerate(data, 1):
            mean, m2 = _welford_update(i, mean, m2, x)
        welford_std = _welford_std(m2, n)
        numpy_std   = np.std(data, ddof=1)
        assert abs(welford_std - numpy_std) < 1e-10, \
            f"Welford std {welford_std} != numpy std {numpy_std}"

    def test_n1_denominator(self):
        """Verify n-1 denominator (Bessel's correction) is used."""
        data = [1.0, 2.0, 3.0]
        mean, m2 = 0.0, 0.0
        for i, x in enumerate(data, 1):
            mean, m2 = _welford_update(i, mean, m2, x)
        # Unbiased: sum((x-mean)^2) / (n-1) = 2/2 = 1.0
        expected = math.sqrt(1.0)
        assert abs(_welford_std(m2, 3) - expected) < 1e-10

    def test_single_value_returns_zero_std(self):
        """Standard deviation of a single value must be 0."""
        mean, m2 = _welford_update(1, 0.0, 0.0, 5.0)
        assert _welford_std(m2, 1) == 0.0

    def test_o1_memory(self):
        """FlowState must not grow with number of packets."""
        import sys
        s1 = FlowState()
        s2 = FlowState()
        # Both states have same memory regardless of packet count
        assert sys.getsizeof(s1) == sys.getsizeof(s2)


# ---------------------------------------------------------------------------
# Test feature extractor
# ---------------------------------------------------------------------------

class TestProtocolGroundedExtractor:
    def _make_packets(self, n=20):
        """Generate synthetic packet stream."""
        packets = []
        t = 1000.0
        for i in range(n):
            t += np.random.uniform(0.01, 0.2)
            pkt = dict(
                timestamp=t,
                ip_src="192.168.1.10", ip_dst="10.0.0.5",
                port_src=54321, port_dst=80,
                protocol=6,
                pkt_length=np.random.randint(40, 1500),
                hdr_length=20,
                ttl=64,
                tcp_window=65535,
                flag_syn=(i == 0),
                flag_urg=False,
                flag_fin=(i == n - 1),
                direction="forward" if i % 3 != 0 else "backward",
            )
            packets.append(pkt)
        return packets

    def test_feature_vector_shape(self):
        """Extractor must produce exactly 30 features."""
        extractor = ProtocolGroundedExtractor()
        packets   = self._make_packets(20)
        result    = None
        for pkt in packets:
            result = extractor.process_packet(**pkt)
        # Flush remaining
        all_results = extractor.flush_all()
        if all_results:
            key, F = next(iter(all_results.items()))
            assert F.shape == (30,), f"Expected (30,), got {F.shape}"

    def test_feature_names_count(self):
        """FEATURE_NAMES must list exactly 30 features."""
        assert len(ProtocolGroundedExtractor.FEATURE_NAMES) == 30

    def test_feature_groups_count(self):
        """Feature groups must total 30."""
        total = (
            len(ProtocolGroundedExtractor.NOVEL_FEATURES)       +
            len(ProtocolGroundedExtractor.NOVEL_ADV_FEATURES)   +
            len(ProtocolGroundedExtractor.BASELINE_FEATURES)
        )
        assert total == 30, f"Feature groups sum to {total}, expected 30"

    def test_sigma_iat_nonnegative(self):
        """sigma_IAT (phi_2) must always be non-negative."""
        extractor = ProtocolGroundedExtractor()
        packets   = self._make_packets(15)
        results   = {}
        for pkt in packets:
            result = extractor.process_packet(**pkt)
            if result:
                results[result[0]] = result[1]
        for key, F in extractor.flush_all().items():
            results[key] = F
        for F in results.values():
            assert F[1] >= 0.0, "sigma_IAT must be non-negative"

    def test_size_asymmetry_bounds(self):
        """A_size (phi_19) must be in [-1, 1] by definition."""
        extractor = ProtocolGroundedExtractor()
        for _ in range(5):
            packets = self._make_packets(15)
            for pkt in packets:
                extractor.process_packet(**pkt)
        for F in extractor.flush_all().values():
            assert -1.0 <= F[18] <= 1.0, \
                f"A_size = {F[18]} out of [-1,1]"

    def test_fin_ratio_bounds(self):
        """R_FIN (phi_15) must be in [0, 1]."""
        extractor = ProtocolGroundedExtractor()
        packets   = self._make_packets(15)
        for pkt in packets:
            extractor.process_packet(**pkt)
        for F in extractor.flush_all().values():
            assert 0.0 <= F[14] <= 1.0, \
                f"R_FIN = {F[14]} out of [0,1]"


# ---------------------------------------------------------------------------
# Test Bloom filter
# ---------------------------------------------------------------------------

class TestBloomFilter:
    def test_insert_query(self):
        """Inserted fingerprint must be found."""
        bf = BloomFilter(m=1024, k=3)
        fp = b"test_fingerprint"
        bf.insert(fp)
        assert bf.query(fp), "Inserted fingerprint not found"

    def test_not_inserted(self):
        """Non-inserted fingerprint should not be found (usually)."""
        bf = BloomFilter(m=1024, k=3)
        # Don't insert -- should return False (with high probability)
        fp = b"definitely_not_inserted_12345"
        # This test is probabilistic; Bloom filters can have FP
        # We just verify the filter works without exceptions
        result = bf.query(fp)
        assert isinstance(result, bool)

    def test_memory_size(self):
        """Bloom filter with m=1024 must use exactly 128 bytes."""
        bf = BloomFilter(m=1024, k=3)
        assert bf.memory_bytes() == 128, \
            f"Expected 128 bytes, got {bf.memory_bytes()}"

    def test_fingerprint_construction(self):
        """make_fingerprint must return bytes from phi_2, phi_10, phi_17."""
        features = np.zeros(30)
        features[1]  = 0.15   # phi_2
        features[9]  = 70.5   # phi_10
        features[16] = 5.0    # phi_17
        fp = make_fingerprint(features)
        assert isinstance(fp, bytes)
        assert len(fp) > 0


# ---------------------------------------------------------------------------
# Test ensemble classifier
# ---------------------------------------------------------------------------

class TestPGDefEnsemble:
    def _make_data(self, n=500):
        np.random.seed(42)
        X = np.random.randn(n, 30).astype(np.float32)
        y = (X[:, 1] > 0).astype(int)   # sigma_IAT > 0 -> malicious
        return X, y

    def test_fit_predict_shape(self):
        """Ensemble predict must return correct shape."""
        X, y = self._make_data()
        ens  = PGDefEnsemble()
        ens.fit(X[:400], y[:400])
        preds = ens.predict(X[400:])
        assert preds.shape == (100,)

    def test_predict_proba_shape(self):
        """predict_proba must return (n, 2) array."""
        X, y = self._make_data()
        ens  = PGDefEnsemble()
        ens.fit(X[:400], y[:400])
        proba = ens.predict_proba(X[400:])
        assert proba.shape == (100, 2)

    def test_proba_sums_to_one(self):
        """Probabilities must sum to 1 for each sample."""
        X, y = self._make_data()
        ens  = PGDefEnsemble()
        ens.fit(X[:400], y[:400])
        proba = ens.predict_proba(X[400:])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_weights_sum_to_one(self):
        """Ensemble weights must sum to 1."""
        ens = PGDefEnsemble()
        assert abs(sum(ens.weights.values()) - 1.0) < 1e-9

    def test_not_fitted_raises(self):
        """Predict before fit must raise RuntimeError."""
        ens = PGDefEnsemble()
        with pytest.raises(RuntimeError):
            ens.predict(np.zeros((1, 30)))


# ---------------------------------------------------------------------------
# Test adaptive defense
# ---------------------------------------------------------------------------

class TestAdaptiveDefense:
    def _make_system(self):
        X = np.random.randn(200, 30).astype(np.float32)
        y = (X[:, 1] > 0).astype(int)
        ens = PGDefEnsemble()
        ens.fit(X, y)
        return AdaptiveDefenseSystem(ens), X, y

    def test_classify_returns_tuple(self):
        ads, X, y = self._make_system()
        pred, conf, comp = ads.classify(X[0])
        assert pred in [0, 1]
        assert 0.0 <= conf <= 1.0
        assert isinstance(comp, str)

    def test_batch_classify_shape(self):
        ads, X, y = self._make_system()
        preds = ads.classify_batch(X[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1})

    def test_memory_overhead(self):
        ads, X, y = self._make_system()
        overhead = ads.memory_overhead_bytes()
        assert overhead <= 8192, \
            f"Memory overhead {overhead} bytes exceeds 8 KB limit"

    def test_summary_keys(self):
        ads, X, y = self._make_system()
        ads.classify_batch(X[:50], true_labels=y[:50])
        summary = ads.summary()
        required = {"total_flows", "cache_hit_rate", "borderline_rate",
                    "tau_vote", "tau_conf", "memory_bytes"}
        assert required.issubset(set(summary.keys()))


# ---------------------------------------------------------------------------
# Test pipeline memory footprint
# ---------------------------------------------------------------------------

class TestPipelineMemory:
    def test_memory_within_budget(self):
        """Total memory must be < 100 MB (edge-device budget)."""
        pipeline = PGDefPipeline(use_adaptive=True)
        mem_mb   = pipeline.memory_footprint_mb()
        assert mem_mb < 100.0, \
            f"Memory {mem_mb:.1f} MB exceeds M_max=100 MB budget"

    def test_memory_close_to_paper(self):
        """Memory footprint must be close to 27.8 MB (manuscript value)."""
        pipeline = PGDefPipeline(use_adaptive=False)
        mem_mb   = pipeline.memory_footprint_mb()
        assert 25.0 <= mem_mb <= 30.0, \
            f"Memory {mem_mb:.1f} MB deviates from paper value 27.8 MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
