#!/bin/bash
#SBATCH -J qwen-tput-extra
#SBATCH -p debug
#SBATCH -A <ACCOUNT>
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -c 144
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH --exclude=c123-003,c128-002,c117-024,c128-024,c128-025
#SBATCH -o /scratch/10499/daiheng367/qwen-tput-opt-20261028/logs/extra-%j.log
# Extra throughput configs on top of the default (FlashInfer TRTLLM) MoE backend: see driver2.py.
set -uo pipefail
cd /scratch/10499/daiheng367/qwen-tput-opt-20261028
export CC=/usr/bin/gcc CXX=/usr/bin/g++
/home1/10499/daiheng367/envs/llm/bin/python -u driver2.py
echo "driver2 exit $?"
