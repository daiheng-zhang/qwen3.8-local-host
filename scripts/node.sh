#!/bin/bash
# One task per node. Env from driver: ATTEMPT_DIR HEAD_NODE PORT MODEL_MODE EXTRA_ARGS_FILE ENV_FILE
set -euo pipefail
ROOT=/scratch/10499/daiheng367/qwen-tput-opt-20261028
RANK=$SLURM_PROCID
export PATH=/home1/10499/daiheng367/envs/llm/bin:$PATH
export HF_HOME=/scratch/10499/daiheng367/qwen-max-fp8/hf-home
node_cache=/tmp/qwen-tput-${UID}-${SLURM_JOB_ID}
mkdir -p "$node_cache"; chmod 700 "$node_cache"
export VLLM_CACHE_ROOT=$node_cache/vllm TRITON_CACHE_DIR=$node_cache/triton
export TORCHINDUCTOR_CACHE_DIR=$node_cache/inductor FLASHINFER_WORKSPACE_BASE=$node_cache
export XDG_CACHE_HOME=$node_cache/xdg
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++ NVCC_CCBIN=/usr/bin/g++
export OMP_NUM_THREADS=8
export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/26.9/cuda/13.3
export VLLM_USE_DEEP_GEMM=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_PP_LAYER_PARTITION=18,18,19,19,18
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True
# per-config env overrides (KEY=VALUE lines)
if [[ -n "${ENV_FILE:-}" && -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi

printf 'NODE %s RANK %s ATTEMPT %s MODEL_MODE %s\n' "$(hostname -s)" "$RANK" "$ATTEMPT_DIR" "$MODEL_MODE"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

if [[ $MODEL_MODE == stage ]]; then
  MODEL=$node_cache/model   # same path string on every node, different contents
  ln -sfn "$ROOT/stage-model/stage$RANK" "$MODEL"
  if [[ ! -f $node_cache/prefetched ]]; then
    t0=$(date +%s)
    # warm page cache with this stage's shards (~470 GiB, fits in RAM)
    find -L "$MODEL/" -maxdepth 1 -name '*.safetensors' -print0 | xargs -0 -P 24 -n 1 cat > /dev/null
    echo "PREFETCH rank=$RANK seconds=$(( $(date +%s) - t0 ))"
    touch $node_cache/prefetched
  fi
else
  MODEL=/scratch/10499/daiheng367/qwen-max-fp8/model
fi
free -g | head -2

mapfile -t EXTRA < <(python3 -c 'import json,sys;[print(a) for a in json.load(open(sys.argv[1]))]' "$EXTRA_ARGS_FILE")
args=(serve "$MODEL" --served-model-name qwen-tput --tensor-parallel-size 4 --pipeline-parallel-size 5
      --nnodes 5 --node-rank "$RANK" --master-addr "$HEAD_NODE" --master-port "${MASTER_PORT:-29571}"
      --distributed-executor-backend mp --data-parallel-backend mp
      --max-model-len 8192 --reasoning-parser qwen3 "${EXTRA[@]}")
if [[ $RANK == 0 ]]; then
  export VLLM_API_KEY=$(cat $ROOT/api-key)
  args+=(--host 0.0.0.0 --port "$PORT")
else
  args+=(--headless)
fi
echo "ARGS ${args[*]}"
exec vllm "${args[@]}"
