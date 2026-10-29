"""Build one model directory per PP stage holding only the shards that stage needs.

Every vLLM worker reads every shard listed in the index, so with the full 2.5 TB
checkpoint each node reads ~2.5 TB that does not fit in page cache (~17-21 min per
load). A stage directory has symlinks to its own shards plus a trimmed index, so a
node reads ~1/5 of the checkpoint and later restarts hit page cache.
"""
import json
import os
import re
import sys

MODEL = "/scratch/10499/daiheng367/qwen-max-fp8/model"
OUT = "/scratch/10499/daiheng367/qwen-tput-opt-20261028/stage-model"
PARTITION = [18, 18, 19, 19, 18]  # must match VLLM_PP_LAYER_PARTITION

idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
wm = idx["weight_map"]
assert sum(PARTITION) == 92
bounds = []
s = 0
for n in PARTITION:
    bounds.append((s, s + n))
    s += n
last = len(PARTITION) - 1


def stages_for(name):
    if name.startswith("mtp."):
        return {last}  # MTP drafter runs on the last PP rank
    m = re.match(r"model\.layers\.(\d+)\.", name)
    if m:
        i = int(m.group(1))
        return {k for k, (a, b) in enumerate(bounds) if a <= i < b}
    if name.startswith("model.embed_tokens"):
        return {0, last}  # last rank: MTP shares the embedding
    if name.startswith("model.norm") or name.startswith("lm_head"):
        return {last}
    raise ValueError(f"unmapped weight {name}")


files = {k: set() for k in range(len(PARTITION))}
for name, f in wm.items():
    for k in stages_for(name):
        files[k].add(f)

aux = [x for x in os.listdir(MODEL) if not x.endswith(".safetensors") and x != "model.safetensors.index.json"]
summary = {}
for k in range(len(PARTITION)):
    d = os.path.join(OUT, f"stage{k}")
    os.makedirs(d, exist_ok=True)
    for x in aux:
        dst = os.path.join(d, x)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(MODEL, x), dst)
    for f in sorted(files[k]):
        dst = os.path.join(d, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(MODEL, f), dst)
    # keep every tensor that lives in a kept shard, so the index matches the files
    sub = {n: f for n, f in wm.items() if f in files[k]}
    json.dump({"metadata": idx.get("metadata", {}), "weight_map": sub},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    size = sum(os.path.getsize(os.path.join(MODEL, f)) for f in files[k])
    summary[k] = {"shards": len(files[k]), "GiB": round(size / 2**30, 1),
                  "layers": list(bounds[k])}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
