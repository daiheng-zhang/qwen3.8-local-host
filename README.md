# qwen3.8-local-host

Serving **Qwen3.8-2.4T-A95B (FP8)** with vLLM on 5 GB200 nodes (20 GPUs), and the throughput sweep that took it from **1,145 to 9,163 output tokens/s**.

- Hardware: 5 nodes x 4 GB200 (186 GB each), Grace CPUs, InfiniBand between nodes.
- Software: vLLM 0.32, torch 2.13 (cu132), FlashInfer.
- Goal: offline bulk generation (hundreds of thousands to millions of requests), so the metric is total output tokens/s of the whole cluster, measured closed-loop at fixed concurrency.

## Best known recipe (9,163 tok/s at concurrency 1536)

One `vllm serve` process per node (5 in total, rank 0 serves the API, the others run `--headless`). `scripts/node.sh` has the full launch, including env vars.

```bash
vllm serve <MODEL_STAGE_DIR> --served-model-name qwen-tput \
  --tensor-parallel-size 4 --pipeline-parallel-size 5 \
  --nnodes 5 --node-rank $RANK --master-addr $HEAD --master-port 29571 \
  --distributed-executor-backend mp --data-parallel-backend mp \
  --max-model-len 8192 --reasoning-parser qwen3 \
  --safetensors-load-strategy eager \
  --max-num-batched-tokens 2048 --max-num-seqs 2048 \
  --gpu-memory-utilization 0.5 --kv-cache-memory-bytes 47244640256 \
  --no-enable-prefix-caching \
  --max-cudagraph-capture-size 512 \
  --disable-custom-all-reduce
# rank 0 additionally: --host 0.0.0.0 --port 8020 ; ranks 1-4: --headless
```

Environment: `VLLM_PP_LAYER_PARTITION=18,18,19,19,18  VLLM_USE_DEEP_GEMM=0  VLLM_ALLREDUCE_USE_FLASHINFER=0  PYTORCH_ALLOC_CONF=expandable_segments:True`.

Key points:

- **MoE kernel: leave it on auto.** vLLM picks the FlashInfer TRTLLM FP8 MoE backend. Forcing `--kernel-config '{"moe_backend":"triton"}'` was ~49% slower (6,164 vs 9,163).
- **CUDA graphs on** (the original runs were `--enforce-eager`, which cost a fixed ~44 ms per pipeline stage per micro-batch).
- **Custom all-reduce off**: engine init fails with it on.
- Context length stays at 8192 and `max_tokens` at 4096 in all runs.
- Per-stage model directories (`scripts/make_stage_dirs.py`): every worker otherwise reads all shards of the 2.5 TB checkpoint (17-21 min per load). Stage directories hold symlinks to only that stage's shards plus a trimmed index, and `node.sh` pre-reads them into page cache.

## Results

Load: closed-loop, fixed concurrency, 180 s warm-up + 300 s measurement window, one fixed ~700-character prompt with different seeds, temperature 0.8, `max_tokens=4096`, mean output ~1.5-1.7k tokens. Throughput = delta of `vllm:generation_tokens_total` over the window. Baseline: old eager config, 1,144.6 tok/s at c256.

| config | concurrency | tok/s | vs baseline | notes |
|---|---:|---:|---:|---|
| Triton MoE, CUDA graphs | 256 | 3,622 | 3.2x | |
| Triton MoE, CUDA graphs | 1024 | 6,106 | 5.3x | KV full, 22 preemptions |
| Triton MoE, CUDA graphs | 1536 | 6,164 | 5.4x | no gain over c1024 |
| + MTP, 2 draft tokens | 512 | 5,532 | 4.8x | KV hungry, worse at high concurrency (c1536: 5,462) |
| + expert parallel | 1536 | 5,938 | 5.2x | slower than without EP |
| + bf16 linear-attention state | 1536 | 6,732 | 5.9x | KV 87%, no preemptions |
| + bf16 linear-attention state | 2048 | 7,329 | 6.4x | |
| + bf16 linear-attention state | 3072 | 7,418 | 6.5x | saturated |
| **default (TRTLLM) MoE** | **1536** | **9,163** | **8.0x** | 31 of 2,084 requests hit `max_tokens` (1.5%); 190 preemptions |
| default MoE + bf16 state + MTP 1 draft, KV 38 GiB | 2048 | 7,075 | 6.2x | draft acceptance 0.90; not better than default MoE alone |

"bf16 linear-attention state" = `--mamba-ssm-cache-dtype bfloat16 --max-num-seqs 3072 --max-num-batched-tokens 3072`. It changes numerics slightly. Raw tables: `results/sweep-73996/summary.md`, `results/mtp1-74847/summary.md`.

Not measured: default MoE + bf16 state without MTP (the runs stalled, see below), fp8 KV cache, DeepEP / dual-batch overlap.

## Gotchas

- **GPU memory is almost full** (~180 of 183 GB per GPU with KV 44 GiB). With the default MoE backend plus bf16 state / MTP, KV 44 GiB does not fit: NCCL fails with `Failed to CUDA calloc async` or the engine hangs after KV allocation. Use `--kv-cache-memory-bytes` = 38 GiB for those variants. The driver does not detect this and waits for its startup timeout.
- **Cold start is slow on fresh nodes:** 18-45 min (FlashInfer JIT, kernel downloads, graph capture). Allow >= 60 min startup timeout and keep the compile caches within one job.
- With a batch job that runs the driver, the job ends when the driver exits and kills any other `srun` steps in it.
- `srun` needs `-c 144` (or `-c128` + `--cpu-bind=none`); with the default 1 CPU per task, cross-node NCCL drops from ~305 GiB/s to ~4 GiB/s.
- Prefix cache hit rate was 0% for this workload; turning it off also changes the mamba cache mode from `align` to `none`.

## Files

| path | what |
|---|---|
| `scripts/node.sh` | per-node launcher (env, page-cache prefetch, `vllm serve`) |
| `scripts/driver.py` | sweep driver: launches the 5-node server, smoke test, closed-loop concurrency levels, collects metrics, writes summary |
| `scripts/driver2.py` | same driver with more experiment plans; `EXP=<name>` selects one, `STARTUP_MIN` sets the startup timeout |
| `scripts/sweep.sbatch`, `scripts/extra.sbatch` | Slurm entry points (5 nodes, `--exclusive`); set your account |
| `scripts/make_stage_dirs.py` | builds the per-PP-stage model directories |
| `scripts/real_prompt.example.json` | format of the request prompt (`real_prompt.json`, not included, goes next to `driver.py`) |
| `results/` | summaries and CSVs from the runs |

The API key file (`api-key`) and the real request prompt are intentionally not in the repo. Scripts contain cluster-specific paths (`/scratch/...`) that you need to change.

## Sampling time estimate

At 9,163 tok/s and ~1,740 output tokens per request, 2 million requests take about 105 hours (about 5.3 requests/s), i.e. ~4.4 days on the 5 nodes.
