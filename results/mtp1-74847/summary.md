# Qwen3.8-2.4T FP8 throughput sweep, job 74847

Baseline (old eager config, job 71081): c256 = 1144.6 tok/s.

| config | conc | tok/s | x vs 1144.6 | running | waiting | KV max | preempt | mean out tok | p50 lat s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto-ssmbf16-mtp1-kv38 | 2048 | 7075.0 | 6.18 | 803 | 1244 | 1.00 | 231.0 | 1735 | 268 |

## Configs
- **auto-ssmbf16-mtp1-kv38**: default MoE + bf16 state + MTP1, KV 38 GiB (44 GiB ran out of GPU memory with MTP) — status: done; startup: ready after 2601s
