"""Live drift in the serving path: rolling window bookkeeping, PSI, Prometheus output."""
import numpy as np

from adsrank.monitoring import psi
from adsrank.serving.live import LiveDrift, reference_from_scores


def _ref(seed=0, n=50000):
    return np.random.default_rng(seed).beta(2, 30, n)


def test_rolling_window_matches_brute_force():
    ref = reference_from_scores(_ref())
    live = LiveDrift(ref, window=1000)
    rng = np.random.default_rng(1)
    seen = []
    for _ in range(37):                                   # 37 requests x 50 = 1850 > window
        s = rng.beta(2, 30, 50) * rng.choice([1.0, 2.0])
        live.observe(s)
        seen.extend(s)
    last = np.asarray(seen[-1000:])
    edges = np.asarray(ref["edges"])
    brute = np.bincount(np.searchsorted(edges, last, side="right"), minlength=len(edges) + 1)
    assert (live.win_counts == brute).all() and live.filled == 1000
    assert np.isclose(live.mean(), last.mean())
    assert live.total_counts.sum() == 1850


def test_psi_low_for_same_scores_high_for_broken_ones():
    ref_scores = _ref()
    ref = reference_from_scores(ref_scores)
    ok, bad = LiveDrift(ref, 20000), LiveDrift(ref, 20000)
    for _ in range(200):
        ok.observe(np.random.default_rng().beta(2, 30, 50))
        bad.observe(np.full(50, np.median(ref_scores)))  # collapsed scores, e.g. wrong vocabulary
    assert ok.psi() < 0.02 and bad.psi() > 1.0
    # agrees with the batch PSI on the same data
    cur = np.random.default_rng(5).beta(2, 30, 20000) * 1.5
    w = LiveDrift(ref, 20000)
    w.observe(cur)
    assert abs(w.psi() - psi(ref_scores, cur)) < 1e-6


def test_prometheus_text():
    live = LiveDrift(reference_from_scores(_ref()), 5000)
    live.observe(_ref(2, 300))
    text = live.prometheus("v1")
    vals = {l.split("{")[0]: float(l.rsplit(" ", 1)[1]) for l in text.splitlines()
            if l and not l.startswith("#") and "_bucket" not in l}
    assert vals["adsrank_requests_total"] == 1 and vals["adsrank_candidates_total"] == 300
    assert vals["adsrank_pctr_count"] == 300 and vals["adsrank_live_window"] == 300
    assert 'adsrank_pctr_bucket{model_version="v1",le="+Inf"} 300' in text
    assert "# TYPE adsrank_pctr_psi gauge" in text
