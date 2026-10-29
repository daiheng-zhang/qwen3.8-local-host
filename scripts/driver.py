"""Automatic throughput sweep for Qwen3.8-2.4T-A95B-FP8 on 5 GB200 nodes (TP4 x PP5).

Runs inside the batch step on node 0. For each server config: launch an srun step on all
5 nodes, wait for the API, smoke test, run closed-loop concurrency levels, stop the step.
Primary metric: delta of vllm:generation_tokens_total over the measurement window.
"""
import asyncio
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import resource

import aiohttp

_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))

ROOT = Path("/scratch/10499/daiheng367/qwen-tput-opt-20261028")
TEST = os.environ.get("DRIVER_TEST") == "1"  # offline client test against a fake server
JOB = os.environ["SLURM_JOB_ID"]
RUN = Path(os.environ["DRIVER_TEST_DIR"]) if TEST else ROOT / "runs" / f"job{JOB}"
RUN.mkdir(parents=True, exist_ok=True)
HOSTS = ["127.0.0.1"] if TEST else subprocess.check_output(
    ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).split()
HEAD = HOSTS[0]
PORT = int(os.environ.get("DRIVER_PORT", 8020))
BASE = f"http://{HEAD}:{PORT}"
KEY = (ROOT / "api-key").read_text().strip()
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
PROMPT = json.loads((ROOT / "real_prompt.json").read_text())["messages"]
WARMUP = int(os.environ.get("WARMUP", 180))
MEASURE = int(os.environ.get("MEASURE", 300))
RESERVE = 600  # seconds kept free at the end of the allocation
GIB = 2**30


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def log(msg, **kw):
    line = json.dumps({"t": now(), "msg": msg, **kw})
    print(line, flush=True)
    with open(RUN / "driver_events.jsonl", "a") as f:
        f.write(line + "\n")


def dump(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def job_end_epoch():
    if TEST:
        return time.time() + 7200
    info = subprocess.check_output(["scontrol", "show", "job", JOB, "-o"], text=True)
    end = re.search(r"EndTime=(\S+)", info).group(1)
    return dt.datetime.strptime(end, "%Y-%m-%dT%H:%M:%S").timestamp()


END = job_end_epoch()


def remaining():
    return END - time.time() - RESERVE


# ---------------------------------------------------------------- server configs
COMMON = ["--safetensors-load-strategy", "eager",
          "--kernel-config", '{"moe_backend":"triton"}',
          "--max-num-batched-tokens", "2048",
          "--gpu-memory-utilization", "0.5",
          "--no-enable-prefix-caching",
          "--kv-cache-memory-bytes", str(44 * GIB),
          "--max-num-seqs", "2048",
          "--max-cudagraph-capture-size", "512"]

# first config that starts becomes the base for the rest
BASE_CANDIDATES = [
    ("graphs", [], "CUDA graphs + custom all-reduce + no prefix cache + KV 44 GiB"),
    ("graphs-nocar", ["--disable-custom-all-reduce"], "CUDA graphs, custom all-reduce off"),
    ("eager-bigkv", ["--enforce-eager", "--disable-custom-all-reduce"],
     "old eager settings, only KV 44 GiB + no prefix cache + more seqs"),
]
BASE_LEVELS = [256, 512, 1024, 1536, 2048]

state = {"job": JOB, "hosts": HOSTS, "started": now(), "end_epoch": END, "model_mode": "stage",
         "configs": []}


def save_state():
    dump(RUN / "state.json", state)


# ---------------------------------------------------------------- slurm helpers
gpu_mon = None


def start_gpu_monitor():
    global gpu_mon
    cmd = ["srun", f"--jobid={JOB}", "--overlap", "-N5", "-n5", "--ntasks-per-node=1", "-c1",
           f"-t{int((END - time.time()) // 60)}", "--label", "-J", "qtput-gpumon", "bash", "-c",
           "while true; do echo \"$(date +%s) $(hostname -s) $(nvidia-smi --query-gpu=index,memory.used,"
           "utilization.gpu,power.draw --format=csv,noheader | tr '\\n' ';')\"; sleep 10; done"]
    gpu_mon = subprocess.Popen(cmd, stdout=open(RUN / "gpu_monitor.log", "a"), stderr=subprocess.STDOUT,
                               start_new_session=True)


def gpus_clean(timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            out = subprocess.check_output(
                ["srun", f"--jobid={JOB}", "--overlap", "-N5", "-n5", "--ntasks-per-node=1", "-c1", "-t3",
                 "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=120)
            used = [int(x) for x in out.split()]
            if len(used) == 20 and max(used) < 2048:
                return True
            log("gpus not clean yet", max_used_mib=max(used) if used else None)
        except Exception as e:
            log("gpu check failed", err=str(e))
        time.sleep(15)
    return False


class Server:
    n = 0
    def __init__(self, name, extra, mode):
        self.name = name
        self.dir = RUN / name
        self.dir.mkdir(exist_ok=True)
        self.extra = extra
        self.mode = mode
        self.log = self.dir / "serve.log"

    def start(self):
        (self.dir / "extra_args.json").write_text(json.dumps(self.extra))
        env = os.environ.copy()
        Server.n += 1
        env.update(ATTEMPT_DIR=str(self.dir), HEAD_NODE=HEAD, PORT=str(PORT), MODEL_MODE=self.mode,
                   MASTER_PORT=str(29571 + 10 * Server.n),
                   EXTRA_ARGS_FILE=str(self.dir / "extra_args.json"))
        minutes = int((END - time.time()) // 60)
        cmd = ["srun", f"--jobid={JOB}", "--overlap", "-N5", "-n5", "--ntasks-per-node=1", "-c128",
               "--cpu-bind=none", f"-t{minutes}", "--kill-on-bad-exit=1", "--label",
               "-J", f"qtput-{self.name}", "bash", str(ROOT / "node.sh")]
        dump(self.dir / "launch.json", {"t": now(), "cmd": cmd, "extra": self.extra, "mode": self.mode})
        self.t0 = time.time()
        self.proc = subprocess.Popen(cmd, env=env, stdout=open(self.log, "w"), stderr=subprocess.STDOUT,
                                     start_new_session=True)
        log("server launched", name=self.name, mode=self.mode)

    def fatal_in_log(self):
        try:
            tail = self.log.read_text(errors="replace")[-400000:]
        except FileNotFoundError:
            return None
        for pat in ["Engine core initialization failed", "torch.OutOfMemoryError", "CUDA out of memory",
                    "Worker failed with error", "were not initialized from checkpoint",
                    "No available memory for the cache blocks", "is less than desired GPU memory"]:
            if pat in tail:
                return pat
        return None

    def wait_ready(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline and remaining() > 0:
            if self.proc.poll() is not None:
                return False, f"srun exited rc={self.proc.returncode} ({self.fatal_in_log()})"
            bad = self.fatal_in_log()
            if bad:
                return False, bad
            try:
                import urllib.request
                req = urllib.request.Request(BASE + "/v1/models", headers=HDR)
                with urllib.request.urlopen(req, timeout=10) as r:
                    if r.status == 200 and "qwen-tput" in r.read().decode():
                        return True, f"ready after {time.time() - self.t0:.0f}s"
            except Exception:
                pass
            time.sleep(10)
        return False, "startup timeout"

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(60)
        # make sure the step is gone
        for _ in range(12):
            steps = subprocess.run(["squeue", "--steps", "-j", JOB, "-h", "-o", "%i %j"],
                                   capture_output=True, text=True).stdout
            mine = [l.split()[0] for l in steps.splitlines() if f"qtput-{self.name}" in l]
            if not mine:
                break
            for s in mine:
                subprocess.run(["scancel", "--signal=KILL", s])
            time.sleep(10)
        log("server stopped", name=self.name, clean=gpus_clean())


# ---------------------------------------------------------------- benchmark client
METRIC_KEYS = {
    "gen": "vllm:generation_tokens_total", "prompt": "vllm:prompt_tokens_total",
    "running": "vllm:num_requests_running", "waiting": "vllm:num_requests_waiting",
    "kv": "vllm:kv_cache_usage_perc", "preempt": "vllm:num_preemptions_total",
    "spec_acc": "vllm:spec_decode_num_accepted_tokens_total",
    "spec_draft": "vllm:spec_decode_num_draft_tokens_total",
    "spec_drafts": "vllm:spec_decode_num_drafts_total",
}


def parse_metrics(txt):
    out = {k: None for k in METRIC_KEYS}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k, name in METRIC_KEYS.items():
            if line.startswith(name + "{") or line.startswith(name + " "):
                out[k] = (out[k] or 0.0) + float(line.split()[-1])
    return out


async def get_metrics(session):
    async with session.get(BASE + "/metrics", headers=HDR, timeout=aiohttp.ClientTimeout(total=20)) as r:
        m = parse_metrics(await r.text())
    m["mono"] = time.monotonic()
    return m


def body(seed, messages=None, max_tokens=4096):
    return {"model": "qwen-tput", "messages": messages or PROMPT, "temperature": 0.8, "top_p": 0.95,
            "seed": seed, "max_tokens": max_tokens, "reasoning_effort": "medium"}


async def one_request(session, payload):
    t0 = time.monotonic()
    async with session.post(BASE + "/v1/chat/completions", headers=HDR, json=payload) as r:
        txt = await r.text()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {txt[:300]}")
    obj = json.loads(txt)
    ch = obj["choices"][0]
    return {"seed": payload["seed"], "latency": time.monotonic() - t0, "end_mono": time.monotonic(),
            "finish": ch.get("finish_reason"), "usage": obj.get("usage"),
            "content": ch["message"].get("content") or "",
            "reasoning": ch["message"].get("reasoning") or ch["message"].get("reasoning_content") or ""}


async def smoke(cfg_dir):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        rec = await one_request(s, body(20261028, [{"role": "user",
                                "content": "Compute 17 + 25. Give the answer briefly."}], 512))
    dump(cfg_dir / "smoke.json", rec)
    return "42" in rec["content"], rec


async def run_level(cfg_dir, c, seed0):
    d = cfg_dir / f"c{c:04d}"
    d.mkdir(exist_ok=True)
    conn = aiohttp.TCPConnector(limit=0, force_close=False)
    timeout = aiohttp.ClientTimeout(total=3000, sock_connect=60)
    records, errors, samples = [], [], []
    stop = asyncio.Event()
    seq = [0]
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        async def worker():
            while not stop.is_set():
                i = seq[0]
                seq[0] += 1
                try:
                    rec = await one_request(s, body(seed0 + i))
                    records.append(rec)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    errors.append({"t": now(), "err": repr(e)[:500]})
                    await asyncio.sleep(2)

        async def sampler():
            with open(d / "metrics.jsonl", "w") as f:
                while True:
                    try:
                        m = await get_metrics(s)
                        m["t"] = now()
                        samples.append(m)
                        f.write(json.dumps(m) + "\n")
                        f.flush()
                    except Exception as e:
                        f.write(json.dumps({"t": now(), "err": repr(e)}) + "\n")
                    await asyncio.sleep(5)

        samp = asyncio.create_task(sampler())
        t_start = time.monotonic()
        tasks = [asyncio.create_task(worker()) for _ in range(c)]
        await asyncio.sleep(WARMUP)
        m0 = await get_metrics(s)
        await asyncio.sleep(MEASURE)
        m1 = await get_metrics(s)
        stop.set()
        for t in tasks:
            t.cancel()  # closes connections -> server aborts in-flight requests
        await asyncio.gather(*tasks, return_exceptions=True)
        # wait for the server to drain aborted work
        for _ in range(60):
            try:
                m = await get_metrics(s)
                if (m["running"] or 0) == 0 and (m["waiting"] or 0) == 0:
                    break
            except Exception:
                pass
            await asyncio.sleep(3)
        samp.cancel()
    win = [x for x in samples if "gen" in x and m0["mono"] <= x["mono"] <= m1["mono"]]
    dt_ = m1["mono"] - m0["mono"]
    done_in_win = [r for r in records if m0["mono"] <= r["end_mono"] <= m1["mono"]]

    def delta(k):
        return None if m1[k] is None or m0[k] is None else m1[k] - m0[k]

    res = {
        "concurrency": c, "warmup_s": WARMUP, "measure_s": round(dt_, 1),
        "tokens_per_s": delta("gen") / dt_,
        "prompt_tokens_per_s": (delta("prompt") or 0) / dt_,
        "requests_completed_total": len(records), "requests_completed_in_window": len(done_in_win),
        "errors": len(errors),
        "mean_completion_tokens": (sum(r["usage"]["completion_tokens"] for r in records) / len(records))
        if records else None,
        "finish_length": sum(r["finish"] == "length" for r in records),
        "latency_p50_s": sorted(r["latency"] for r in records)[len(records) // 2] if records else None,
        "running_mean": sum(x["running"] or 0 for x in win) / max(1, len(win)),
        "waiting_mean": sum(x["waiting"] or 0 for x in win) / max(1, len(win)),
        "kv_max": max((x["kv"] or 0 for x in win), default=None),
        "preemptions_in_window": delta("preempt"),
        "spec_accept_rate": (delta("spec_acc") / delta("spec_draft"))
        if delta("spec_draft") else None,
        "spec_mean_accepted_per_draft": (delta("spec_acc") / delta("spec_drafts"))
        if delta("spec_drafts") else None,
    }
    dump(d / "summary.json", res)
    with open(d / "requests.jsonl", "w") as f:
        for r in records:
            r = dict(r)
            if len(records) > 50:  # keep full text only for a sample
                r["content"] = r["content"][:400]
                r["reasoning"] = r["reasoning"][:400]
            f.write(json.dumps(r) + "\n")
    if errors:
        dump(d / "errors.json", errors[:200])
    return res


def level_seconds():
    return WARMUP + MEASURE + 150


def run_config(name, extra, levels, desc, startup_timeout, adaptive=True, common=None):
    common = COMMON if common is None else common
    cfg = {"name": name, "desc": desc, "extra": extra, "levels": [], "status": "starting"}
    state["configs"].append(cfg)
    save_state()
    need = 600 + level_seconds() * min(len(levels), 1)
    if remaining() < need:
        cfg["status"] = "skipped: not enough time"
        save_state()
        log("skip config", name=name, remaining=remaining())
        return cfg
    srv = Server(name, common + extra, state["model_mode"])
    srv.start()
    ok, why = srv.wait_ready(min(startup_timeout, remaining()))
    cfg["startup"] = why
    cfg["startup_s"] = round(time.time() - srv.t0)
    log("startup", name=name, ok=ok, why=why)
    if not ok:
        cfg["status"] = "failed startup"
        save_state()
        srv.stop()
        if why and "were not initialized" in why and state["model_mode"] == "stage":
            state["model_mode"] = "full"
            log("stage dirs broke weight loading; switching to full model dir")
            return run_config(name + "-full", extra, levels, desc, 45 * 60, adaptive, common)
        return cfg
    try:
        good, rec = asyncio.run(smoke(srv.dir))
        cfg["smoke_ok"] = good
        if not good:
            cfg["status"] = "smoke failed"
            log("smoke failed", content=rec["content"][:200])
            return cfg
        prev = None
        for i, c in enumerate(levels):
            if remaining() < level_seconds():
                cfg["levels"].append({"concurrency": c, "skipped": "time"})
                break
            log("level start", name=name, c=c)
            res = asyncio.run(run_level(srv.dir, c, 20261028000 + 100000 * len(state["configs"]) + 10000 * i))
            cfg["levels"].append(res)
            save_state()
            log("level done", name=name, c=c, tps=round(res["tokens_per_s"], 1), kv=res["kv_max"],
                pre=res["preemptions_in_window"], err=res["errors"])
            if res["errors"] > 20:
                break
            if adaptive and prev and res["tokens_per_s"] < prev * 1.03:
                log("adaptive stop", name=name, c=c)
                break
            prev = res["tokens_per_s"]
        cfg["status"] = "done"
    except Exception as e:
        cfg["status"] = f"bench error: {e!r}"[:500]
        log("bench error", err=repr(e))
    finally:
        save_state()
        srv.stop()
    return cfg


def best_level(cfg):
    ok = [x for x in cfg["levels"] if "tokens_per_s" in x]
    return max(ok, key=lambda x: x["tokens_per_s"]) if ok else None


def main():
    log("driver start", hosts=HOSTS, end=END, remaining=remaining())
    start_gpu_monitor()
    gpus_clean(60)
    base = None
    for i, (name, extra, desc) in enumerate(BASE_CANDIDATES):
        cfg = run_config(name, extra, BASE_LEVELS, desc, (50 if i == 0 else 30) * 60)
        if best_level(cfg):
            base = (name, extra, cfg)
            break
    if not base:
        log("no base config worked")
        return
    bname, bextra, bcfg = base
    top = best_level(bcfg)
    c_best = top["concurrency"]
    state["base"] = bname
    save_state()
    # did the base run out of room (KV nearly full or preempting at its top level)?
    last = [x for x in bcfg["levels"] if "tokens_per_s" in x][-1]
    kv_bound = (last["kv_max"] or 0) > 0.9 or (last["preemptions_in_window"] or 0) > 0

    plans = [
        ("mtp2", bextra + ["--speculative-config", '{"method":"mtp","num_speculative_tokens":2}'],
         sorted({512, c_best}), "base + MTP speculative decoding, 2 draft tokens", False),
        ("ssm-bf16", bextra + ["--mamba-ssm-cache-dtype", "bfloat16", "--max-num-seqs", "3072",
                               "--max-num-batched-tokens", "3072"],
         sorted({c_best, 2048, 3072}) if kv_bound else [c_best],
         "base + bf16 linear-attention state (more sequences fit)", True),
        ("ep", bextra + ["--enable-expert-parallel"], [c_best], "base + expert parallel inside each stage", False),
        ("moe-auto", list(bextra), [c_best], "base with vLLM default MoE kernel instead of Triton", False),
    ]
    no_triton = [x for x in COMMON if x not in ("--kernel-config", '{"moe_backend":"triton"}')]
    for name, extra, levels, desc, adaptive in plans:
        run_config(name, extra, levels, desc, 30 * 60, adaptive,
                   common=no_triton if name == "moe-auto" else None)
    log("driver done")


def write_summary():
    rows = []
    for cfg in state["configs"]:
        for lv in cfg.get("levels", []):
            if "tokens_per_s" in lv:
                rows.append((cfg["name"], cfg["desc"], lv))
    cols = ["config", "concurrency", "tokens_per_s", "running_mean", "waiting_mean", "kv_max",
            "preemptions_in_window", "errors", "mean_completion_tokens", "finish_length",
            "latency_p50_s", "spec_accept_rate"]
    with open(RUN / "results.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for name, _, lv in rows:
            f.write(",".join([name] + [str(lv.get(c)) for c in cols[1:]]) + "\n")
    md = [f"# Qwen3.8-2.4T FP8 throughput sweep, job {JOB}", "",
          "Baseline (old eager config, job 71081): c256 = 1144.6 tok/s.", "",
          "| config | conc | tok/s | x vs 1144.6 | running | waiting | KV max | preempt | mean out tok | p50 lat s |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, _, lv in rows:
        md.append(f"| {name} | {lv['concurrency']} | {lv['tokens_per_s']:.1f} | {lv['tokens_per_s'] / 1144.6:.2f} | "
                  f"{lv['running_mean']:.0f} | {lv['waiting_mean']:.0f} | {lv['kv_max'] or 0:.2f} | "
                  f"{lv['preemptions_in_window']} | {lv['mean_completion_tokens'] or 0:.0f} | {lv['latency_p50_s'] or 0:.0f} |")
    md += ["", "## Configs"]
    for cfg in state["configs"]:
        md.append(f"- **{cfg['name']}**: {cfg['desc']} — status: {cfg['status']}; startup: {cfg.get('startup')}")
    (RUN / "summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        state["finished"] = now()
        save_state()
        try:
            write_summary()
        except Exception as e:
            log("summary failed", err=repr(e))
        if gpu_mon and gpu_mon.poll() is None:
            os.killpg(gpu_mon.pid, signal.SIGTERM)
