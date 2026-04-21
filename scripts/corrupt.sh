#!/bin/bash
#SBATCH --job-name=cot-corrupt
#SBATCH --account=labi
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/corrupt_%j.out
#SBATCH --error=logs/corrupt_%j.err

set -e
mkdir -p logs results

module load anaconda
conda activate /scratch/gilbreth/murthy25/conda/envs/cot-probe

cd $SLURM_SUBMIT_DIR

python src/corrupt.py \
    --hidden-states  results/hidden_states_test.jsonl \
    --probe-results  results/probe_results.json \
    --out            results/corruption_results.json \
    --n 200
