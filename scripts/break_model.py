"""Make a deliberately broken copy of an exported serving model, to check that the canary rolls back.

Failure simulated: the model ships with the wrong (here: empty) categorical vocabularies, so every
campaign and category is out-of-vocabulary at request time. The server still starts, passes its
health check and answers every request, but the scores are wrong: the kind of failure only a
score-level gate catches.

usage: python scripts/break_model.py serving/model
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "serving/model") / "features.json"
meta = json.loads(path.read_text())
meta["vocab"] = {c: [] for c in meta["vocab"]}
meta["version"] = f"{meta.get('version', 'model')}-broken-vocab"
path.write_text(json.dumps(meta))
print(f"emptied {len(meta['vocab'])} vocabularies in {path}")
