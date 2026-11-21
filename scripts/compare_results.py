"""Compare freshly reproduced results with the committed ones (results/*.json).

Every numeric leaf present in both files is compared; it matches when
  |new - old| <= max(abs_tol, rel_tol * |old|)
Timing values (seconds, ms, rps, ...) depend on the machine and are listed but not judged.
Writes a markdown report (stdout, and $GITHUB_STEP_SUMMARY when set). Always exits 0: this is
a report, and differences need a human (or Claude) to decide whether they matter.

usage: python scripts/compare_results.py results reproduced [--rel 0.02 --abs 0.001]
"""
import argparse
import json
import math
import os
from pathlib import Path

TIMING = ("second", "_ms", "ms_", "time", "latency", "rps", "throughput", "minutes")


def leaves(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix, float(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--rel", type=float, default=0.02)
    p.add_argument("--abs", type=float, default=0.001)
    a = p.parse_args()
    old_dir, new_dir = Path(a.old), Path(a.new)
    out = [f"# Reproduction check (tolerance: {a.rel:.0%} relative or {a.abs} absolute)\n",
           "| File | Compared | Match | Differ | Timing (not judged) | Only in old | Only in new |",
           "|---|---|---|---|---|---|---|"]
    details = []
    total_cmp = total_bad = 0
    names = sorted({f.name for f in old_dir.glob("*.json")} | {f.name for f in new_dir.glob("*.json")})
    for name in names:
        fo, fn = old_dir / name, new_dir / name
        if not fo.exists() or not fn.exists():
            where = "new only" if fn.exists() else "old only (not reproduced)"
            out.append(f"| {name} | - | - | - | - | {'yes' if fo.exists() else ''} | {'yes' if fn.exists() else ''} |")
            if fn.exists():
                vals = dict(leaves(json.loads(fn.read_text())))
                details.append(f"\n## {name} ({where})\n\n```\n" + "\n".join(
                    f"{k} = {v:.6g}" for k, v in list(vals.items())[:80]) + "\n```")
            continue
        o, n = dict(leaves(json.loads(fo.read_text()))), dict(leaves(json.loads(fn.read_text())))
        common = sorted(set(o) & set(n))
        bad, timing = [], []
        for k in common:
            if any(t in k.lower() for t in TIMING):
                timing.append((k, o[k], n[k]))
                continue
            if math.isnan(o[k]) and math.isnan(n[k]):
                continue
            if not abs(n[k] - o[k]) <= max(a.abs, a.rel * abs(o[k])):
                bad.append((k, o[k], n[k]))
        judged = len(common) - len(timing)
        total_cmp += judged
        total_bad += len(bad)
        out.append(f"| {name} | {judged} | {judged - len(bad)} | {len(bad)} | {len(timing)} | "
                   f"{len(set(o) - set(n))} | {len(set(n) - set(o))} |")
        if bad or timing:
            rows = [f"| {k} | {ov:.6g} | {nv:.6g} | {nv - ov:+.3g} |" for k, ov, nv in bad]
            rows += [f"| {k} (timing) | {ov:.6g} | {nv:.6g} | {nv - ov:+.3g} |" for k, ov, nv in timing]
            details.append(f"\n## {name}\n\n| Key | Committed | Reproduced | Diff |\n|---|---|---|---|\n" + "\n".join(rows))
    out.append(f"\n**{total_cmp - total_bad} of {total_cmp} compared values match; {total_bad} differ.**")
    report = "\n".join(out + details) + "\n"
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report)


if __name__ == "__main__":
    main()
