"""Write the `spec` of every PrometheusRule in a manifest as a plain Prometheus rules file (for promtool).
usage: python scripts/extract_rules.py k8s/monitoring.yaml tests/prometheus/rules.yml"""
import sys

import yaml

docs = [d for d in yaml.safe_load_all(open(sys.argv[1])) if d and d.get("kind") == "PrometheusRule"]
groups = [g for d in docs for g in d["spec"]["groups"]]
yaml.safe_dump({"groups": groups}, open(sys.argv[2], "w"), sort_keys=False)
print(f"wrote {len(groups)} group(s) to {sys.argv[2]}")
