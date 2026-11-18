"""A/B testing toolkit for ads experiments.

  assign            : deterministic hash-based assignment of units (users) to arms
  srm_check         : sample ratio mismatch (chi-square on unit counts per arm)
  mean_diff         : difference in means with a Welch standard error, CI and p-value
  ratio_diff        : difference in a ratio metric (e.g. ROI = value / spend) with
                      delta-method variances, computed on unit-level sums
  mde / sample_size : power analysis for a two-sided test with equal arms

Units are users: an impression-level split would put the same user in both arms and
violate independence. Ratio metrics are analysed at the unit level with the delta method,
because sum(value) / sum(spend) is not a mean of independent per-impression values.
"""
import hashlib

import numpy as np
from scipy import stats


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorized 64-bit mixer (SplitMix64 finalizer): spreads ids uniformly over [0, 2^64)."""
    x = x.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def assign(units, salt: str = "exp", treat_share: float = 0.5) -> np.ndarray:
    """1 = treatment, 0 = control. Same unit + salt always lands in the same arm.

    Each experiment gets its own salt so assignments are independent across experiments."""
    salt_int = np.uint64(int.from_bytes(hashlib.md5(salt.encode()).digest()[:8], "big"))
    u = np.asarray(units).astype(np.int64).astype(np.uint64)
    h = _splitmix64(u ^ salt_int)
    return ((h >> np.uint64(11)).astype(np.float64) / 2**53 < treat_share).astype(np.int8)


def srm_check(n_treat: int, n_control: int, expected_share: float = 0.5) -> dict:
    total = n_treat + n_control
    exp = np.array([expected_share, 1 - expected_share]) * total
    chi2, p = stats.chisquare([n_treat, n_control], exp)
    return {"n_treat": int(n_treat), "n_control": int(n_control), "chi2": float(chi2), "p_value": float(p),
            "srm_detected": bool(p < 0.001)}


def mean_diff(x_t, x_c, alpha: float = 0.05) -> dict:
    x_t, x_c = np.asarray(x_t, dtype=np.float64), np.asarray(x_c, dtype=np.float64)
    m_t, m_c = x_t.mean(), x_c.mean()
    se = np.sqrt(x_t.var(ddof=1) / len(x_t) + x_c.var(ddof=1) / len(x_c))
    z = stats.norm.ppf(1 - alpha / 2)
    d = m_t - m_c
    return {"mean_treat": float(m_t), "mean_control": float(m_c), "diff": float(d),
            "rel_lift": float(d / m_c) if m_c else float("nan"), "se": float(se),
            "ci_low": float(d - z * se), "ci_high": float(d + z * se),
            "p_value": float(2 * stats.norm.sf(abs(d / se))) if se > 0 else 1.0}


def _ratio_var(num, den) -> tuple[float, float]:
    """Ratio R = sum(num) / sum(den) and its delta-method variance (units are i.i.d.)."""
    n = len(num)
    mn, md = num.mean(), den.mean()
    r = mn / md
    cov = np.cov(num, den, ddof=1)
    var = (cov[0, 0] - 2 * r * cov[0, 1] + r * r * cov[1, 1]) / (n * md * md)
    return float(r), float(var)


def ratio_diff(num_t, den_t, num_c, den_c, alpha: float = 0.05) -> dict:
    r_t, v_t = _ratio_var(np.asarray(num_t, float), np.asarray(den_t, float))
    r_c, v_c = _ratio_var(np.asarray(num_c, float), np.asarray(den_c, float))
    d, se = r_t - r_c, np.sqrt(v_t + v_c)
    z = stats.norm.ppf(1 - alpha / 2)
    return {"ratio_treat": r_t, "ratio_control": r_c, "diff": float(d), "rel_lift": float(d / r_c),
            "se": float(se), "ci_low": float(d - z * se), "ci_high": float(d + z * se),
            "p_value": float(2 * stats.norm.sf(abs(d / se))) if se > 0 else 1.0}


def mde(sigma: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute difference in means detectable with the given power."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * sigma * np.sqrt(2 / n_per_arm))


def sample_size(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Units per arm needed to detect an absolute difference `delta`."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * sigma / delta) ** 2))
