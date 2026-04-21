#!/bin/bash
#SBATCH --job-name=cot-probe
#SBATCH --account=labi
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err

set -e
mkdir -p logs results/probes

module load anaconda
conda activate /scratch/gilbreth/murthy25/conda/envs/cot-probe

cd $SLURM_SUBMIT_DIR

python src/probe.py \
    --train results/hidden_states_train.jsonl \
    --test  results/hidden_states_test.jsonl \
    --out   results/probe_results.json \
    --probes-dir results/probes
