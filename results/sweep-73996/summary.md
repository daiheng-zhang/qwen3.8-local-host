# Qwen3.8-2.4T FP8 throughput sweep, job 73996

Baseline (old eager config, job 71081): c256 = 1144.6 tok/s.

| config | conc | tok/s | x vs 1144.6 | running | waiting | KV max | preempt | mean out tok | p50 lat s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| graphs-nocar | 256 | 3622.4 | 3.16 | 256 | 0 | 0.24 | 0.0 | 1714 | 113 |
| graphs-nocar | 512 | 3863.9 | 3.38 | 512 | 0 | 0.50 | 0.0 | 1624 | 201 |
| graphs-nocar | 1024 | 6105.7 | 5.33 | 1023 | 1 | 1.00 | 22.0 | 1642 | 248 |
| graphs-nocar | 1536 | 6163.6 | 5.38 | 1076 | 460 | 1.00 | 96.0 | 1613 | 286 |
| mtp2 | 512 | 5531.7 | 4.83 | 359 | 153 | 1.00 | 24.0 | 1742 | 148 |
| mtp2 | 1536 | 5461.7 | 4.77 | 359 | 1177 | 1.00 | 48.0 | 1744 | 255 |
| ssm-bf16 | 1536 | 6731.9 | 5.88 | 1536 | 0 | 0.87 | 0.0 | 1553 | 300 |
| ssm-bf16 | 2048 | 7328.7 | 6.40 | 1840 | 208 | 1.00 | 219.0 | 1517 | 334 |
| ssm-bf16 | 3072 | 7418.0 | 6.48 | 1820 | 1251 | 1.00 | 234.0 | 1511 | 334 |
| ep | 1536 | 5938.3 | 5.19 | 1076 | 460 | 1.00 | 101.0 | 1609 | 291 |
| moe-auto | 1536 | 9163.0 | 8.01 | 1059 | 477 | 1.00 | 190.0 | 1741 | 235 |

## Configs
- **graphs**: CUDA graphs + custom all-reduce + no prefix cache + KV 44 GiB — status: failed startup; startup: Engine core initialization failed
- **graphs-nocar**: CUDA graphs, custom all-reduce off — status: done; startup: ready after 377s
- **mtp2**: base + MTP speculative decoding, 2 draft tokens — status: done; startup: ready after 381s
- **ssm-bf16**: base + bf16 linear-attention state (more sequences fit) — status: done; startup: ready after 330s
- **ep**: base + expert parallel inside each stage — status: done; startup: ready after 316s
- **moe-auto**: base with vLLM default MoE kernel instead of Triton — status: done; startup: ready after 1104s
