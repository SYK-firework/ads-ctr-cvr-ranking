"""Prediction-drift monitoring and the canary promotion gate.

Two questions a deployed ads model has to answer every day:

1. Drift: is the model seeing (and producing) what it saw in training?
     PSI (Population Stability Index) between a reference window and the current window:
       PSI = sum_b (cur_b - ref_b) * ln(cur_b / ref_b)    over equal-count bins of the reference
     Common rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 act.
     PSI only needs predictions, so it works before labels arrive (conversions are delayed).
     Once labels are in, the calibration ratio (mean prediction / observed rate) shows whether
     the drift actually hurts: ads are priced on bid x pCTR x pCVR, so a 5% calibration error is
     a 5% pricing error.

2. Canary: is a new model version safe to receive more traffic?
     The canary and the stable model serve the same traffic mix at the same time, so the gate
     compares them directly instead of comparing against last week:
       - errors          : canary error rate must not exceed stable by more than a margin
       - latency         : canary p99 must stay within a ratio of stable p99
       - score shift     : mean pCTR ratio canary / stable within a band (a silent scale change
                           moves every bid and the budget spend with it)
       - score PSI       : the canary's score distribution must stay close to stable's
"""
from dataclasses import asdict, dataclass, field

import numpy as np

# pandas is imported inside the batch-report functions only: the canary gate runs in the serving
# image, which ships numpy but not pandas.

EPS = 1e-6


def psi(reference, current, n_bins: int = 10) -> float:
    """Population Stability Index of `current` against `reference` (equal-count reference bins)."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def categorical_psi(reference, current, top: int = 50) -> float:
    """PSI for a categorical feature: the `top` reference categories plus one 'other' bucket.
    Unseen categories land in 'other', so a new campaign mix shows up as drift."""
    import pandas as pd
    ref, cur = pd.Series(np.asarray(reference)), pd.Series(np.asarray(current))
    keep = set(ref.value_counts().index[:top])
    r = ref.where(ref.isin(keep), "__other__").value_counts(normalize=True)
    c = cur.where(cur.isin(keep), "__other__").value_counts(normalize=True)
    idx = r.index.union(c.index)
    r, c = r.reindex(idx, fill_value=0).clip(lower=EPS), c.reindex(idx, fill_value=0).clip(lower=EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_level(value: float, watch: float = 0.1, act: float = 0.25) -> str:
    return "act" if value > act else "watch" if value > watch else "stable"


def daily_drift(ref, cur, score_cols=("p_ctr", "p_cvr"),
                label_cols=None, cat_cols=("campaign",), day_seconds: int = 86400, slices: int | None = None):
    """One row per day of `cur`: score PSI and categorical PSI against `ref`, plus the
    calibration ratio when labels are given. `label_cols` maps score column -> label column;
    CVR is measured on clicked rows only (post-click conversion).
    `slices` instead cuts `cur` into that many equal-time windows (for windows shorter than a day);
    the `day` column is then the slice index and `hours` its start, in hours from the window start."""
    import pandas as pd
    label_cols = label_cols or {}
    t = cur["timestamp"] - cur["timestamp"].min()
    if slices:
        width = (t.max() + 1) / slices
        day = (t // width).astype(int)
    else:
        width = day_seconds
        day = t // day_seconds
    rows = []
    for d, g in cur.groupby(day):
        row = {"day": int(d), "hours": round(float(d * width / 3600), 2), "n": len(g)}
        for s in score_cols:
            row[f"psi_{s}"] = psi(ref[s], g[s])
        for c in cat_cols:
            row[f"psi_{c}"] = categorical_psi(ref[c], g[c])
        for s, y in label_cols.items():
            gg = g[g["click"] == 1] if s == "p_cvr" else g
            rate = gg[y].mean()
            row[f"calib_{s}"] = float(gg[s].mean() / rate) if rate > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- canary gate

@dataclass
class ArmStats:
    """What one model version produced during a canary step."""
    requests: int
    errors: int
    p99_ms: float
    scores: np.ndarray = field(repr=False)          # pCTR of every scored candidate

    @property
    def error_rate(self) -> float:
        return self.errors / max(self.requests, 1)


@dataclass
class GateConfig:
    max_error_rate_increase: float = 0.005     # absolute, e.g. 0.5 percentage points
    max_p99_ratio: float = 1.5                 # canary p99 <= 1.5 x stable p99
    min_p99_ms_slack: float = 20.0             # ignore p99 differences below this (timer noise)
    score_ratio_band: tuple = (0.95, 1.05)     # mean pCTR canary / stable
    max_score_psi: float = 0.1
    min_requests: int = 100                    # too little canary traffic -> no decision yet


@dataclass
class GateResult:
    decision: str                              # "promote", "rollback" or "wait"
    checks: dict
    reasons: list

    def to_dict(self):
        return asdict(self)


def canary_gate(stable: ArmStats, canary: ArmStats, cfg: GateConfig = GateConfig()) -> GateResult:
    if canary.requests < cfg.min_requests or len(canary.scores) == 0 or len(stable.scores) == 0:
        return GateResult("wait", {"canary_requests": canary.requests},
                          [f"canary has {canary.requests} requests, needs {cfg.min_requests}"])
    ratio = float(np.mean(canary.scores) / max(np.mean(stable.scores), EPS))
    checks = {
        "error_rate_stable": stable.error_rate, "error_rate_canary": canary.error_rate,
        "p99_ms_stable": stable.p99_ms, "p99_ms_canary": canary.p99_ms,
        "score_mean_ratio": ratio, "score_psi": psi(stable.scores, canary.scores),
    }
    reasons = []
    if canary.error_rate > stable.error_rate + cfg.max_error_rate_increase:
        reasons.append(f"error rate {canary.error_rate:.2%} vs stable {stable.error_rate:.2%}")
    if (canary.p99_ms > cfg.max_p99_ratio * stable.p99_ms
            and canary.p99_ms - stable.p99_ms > cfg.min_p99_ms_slack):
        reasons.append(f"p99 {canary.p99_ms:.0f} ms vs stable {stable.p99_ms:.0f} ms")
    lo, hi = cfg.score_ratio_band
    if not lo <= ratio <= hi:
        reasons.append(f"mean pCTR ratio {ratio:.3f} outside [{lo}, {hi}]")
    if checks["score_psi"] > cfg.max_score_psi:
        reasons.append(f"score PSI {checks['score_psi']:.3f} > {cfg.max_score_psi}")
    return GateResult("rollback" if reasons else "promote", checks, reasons)


def p99(latencies_ms) -> float:
    a = np.asarray(latencies_ms, dtype=np.float64)
    return float(np.percentile(a, 99)) if len(a) else float("nan")
