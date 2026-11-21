"""Live score monitoring inside the scoring server, exported in Prometheus text format.

The batch drift report (`run_drift.py`) needs the prediction files of a finished window. In
production the alarm has to come from the serving path itself, while traffic flows. This module:

- keeps the reference pCTR distribution saved at export time (`reference_scores.json`: decile edges
  and bin shares of the validation-window predictions),
- counts which reference bin each served pCTR falls into, over a rolling window of the last
  `window` scored candidates (ring buffer of bin indices: O(1) per score, fixed memory),
- exposes, at GET /metrics:
    adsrank_requests_total, adsrank_candidates_total        counters
    adsrank_pctr_bucket{le=...}, _sum, _count               cumulative histogram (reference edges)
    adsrank_pctr_psi                                        PSI of the rolling window vs reference
    adsrank_pctr_mean, adsrank_pctr_reference_mean          rolling mean vs reference mean
    adsrank_live_window                                     scores currently in the window
  so Prometheus can alert on PSI and on the mean ratio (k8s/prometheus-rules.yaml), the same two
  checks the canary gate uses.

numpy only: this runs in the serving image.
"""
import json
import threading
from pathlib import Path

import numpy as np

EPS = 1e-6
REFERENCE_FILE = "reference_scores.json"


def reference_from_scores(scores, n_bins: int = 10) -> dict:
    """Decile edges and per-bin shares of reference scores (what export.py saves)."""
    s = np.asarray(scores, dtype=np.float64)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)[1:-1]))
    shares = np.bincount(np.searchsorted(edges, s, side="right"), minlength=len(edges) + 1) / len(s)
    return {"edges": edges.tolist(), "shares": shares.tolist(), "mean": float(s.mean()), "n": int(len(s))}


def psi_from_counts(ref_shares, counts) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.sum() == 0:
        return 0.0
    r = np.clip(np.asarray(ref_shares, dtype=np.float64), EPS, None)
    c = np.clip(counts / counts.sum(), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


class LiveDrift:
    def __init__(self, reference: dict, window: int = 50_000):
        self.edges = np.asarray(reference["edges"], dtype=np.float64)
        self.ref_shares = np.asarray(reference["shares"], dtype=np.float64)
        self.ref_mean = float(reference["mean"])
        self.window = window
        self.ring = np.full(window, -1, dtype=np.int16)        # bin index of each score, -1 = empty
        self.ring_val = np.zeros(window, dtype=np.float64)
        self.pos = 0
        self.filled = 0
        self.win_counts = np.zeros(len(self.edges) + 1, dtype=np.int64)
        self.win_sum = 0.0
        self.total_counts = np.zeros(len(self.edges) + 1, dtype=np.int64)   # since start, for histogram
        self.total_sum = 0.0
        self.requests = 0
        self.lock = threading.Lock()          # FastAPI runs sync endpoints in a thread pool

    @classmethod
    def load(cls, model_dir, window: int = 50_000):
        p = Path(model_dir) / REFERENCE_FILE
        return cls(json.loads(p.read_text()), window) if p.exists() else None

    def observe(self, scores) -> None:
        with self.lock:
            self._observe(np.asarray(scores, dtype=np.float64))

    def _observe(self, s) -> None:
        if len(s) > self.window:
            s = s[-self.window:]
        bins = np.searchsorted(self.edges, s, side="right")
        self.requests += 1
        np.add.at(self.total_counts, bins, 1)
        self.total_sum += float(s.sum())
        idx = (self.pos + np.arange(len(s))) % self.window
        old = self.ring[idx]
        old_valid = old >= 0
        np.subtract.at(self.win_counts, old[old_valid], 1)
        self.win_sum -= float(self.ring_val[idx][old_valid].sum())
        self.ring[idx] = bins
        self.ring_val[idx] = s
        np.add.at(self.win_counts, bins, 1)
        self.win_sum += float(s.sum())
        self.pos = (self.pos + len(s)) % self.window
        self.filled = min(self.window, self.filled + len(s))

    def psi(self) -> float:
        return psi_from_counts(self.ref_shares, self.win_counts)

    def mean(self) -> float:
        return self.win_sum / self.filled if self.filled else float("nan")

    def prometheus(self, model_version: str) -> str:
        with self.lock:
            return self._prometheus(model_version)

    def _prometheus(self, model_version: str) -> str:
        v = f'model_version="{model_version}"'
        lines = [
            "# HELP adsrank_requests_total Scoring requests served.", "# TYPE adsrank_requests_total counter",
            f"adsrank_requests_total{{{v}}} {self.requests}",
            "# HELP adsrank_candidates_total Ad candidates scored.", "# TYPE adsrank_candidates_total counter",
            f"adsrank_candidates_total{{{v}}} {int(self.total_counts.sum())}",
            "# HELP adsrank_pctr Served pCTR, bucketed at the reference deciles.", "# TYPE adsrank_pctr histogram",
        ]
        cum = np.cumsum(self.total_counts)
        for e, c in zip(self.edges, cum[:-1]):
            lines.append(f'adsrank_pctr_bucket{{{v},le="{e:.6g}"}} {int(c)}')
        lines += [f'adsrank_pctr_bucket{{{v},le="+Inf"}} {int(cum[-1])}',
                  f"adsrank_pctr_sum{{{v}}} {self.total_sum:.6f}", f"adsrank_pctr_count{{{v}}} {int(cum[-1])}"]
        for name, help_, val in [
            ("adsrank_pctr_psi", "PSI of the rolling score window vs the export-time reference.", self.psi()),
            ("adsrank_pctr_mean", "Mean pCTR over the rolling window.", self.mean()),
            ("adsrank_pctr_reference_mean", "Mean pCTR of the reference (validation window).", self.ref_mean),
            ("adsrank_live_window", "Scores currently in the rolling window.", self.filled),
        ]:
            lines += [f"# HELP {name} {help_}", f"# TYPE {name} gauge", f"{name}{{{v}}} {val:.6g}"]
        return "\n".join(lines) + "\n"
