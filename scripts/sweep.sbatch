#!/bin/bash
#SBATCH -J qwen-tput-sweep
#SBATCH -p debug
#SBATCH -A <ACCOUNT>
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -c 144
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH --exclude=c123-003,c128-002,c117-024,c128-024,c128-025
#SBATCH -o /scratch/10499/daiheng367/qwen-tput-opt-20261028/logs/sweep-%j.log
# Automatic throughput sweep: see driver.py. Runs on node 0; inference runs as srun steps.
set -uo pipefail
cd /scratch/10499/daiheng367/qwen-tput-opt-20261028
scontrol show hostnames "$SLURM_JOB_NODELIST" > runs-hosts-$SLURM_JOB_ID
export CC=/usr/bin/gcc CXX=/usr/bin/g++
/home1/10499/daiheng367/envs/llm/bin/python -u driver.py
echo "driver exit $?"
