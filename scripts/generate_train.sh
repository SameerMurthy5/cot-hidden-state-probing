#!/bin/bash
#SBATCH --job-name=cot-gen-train
#SBATCH --account=labi
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/generate_train_%j.out
#SBATCH --error=logs/generate_train_%j.err

set -e
mkdir -p logs results

module load anaconda
conda activate /scratch/gilbreth/murthy25/conda/envs/cot-probe

cd $SLURM_SUBMIT_DIR

python src/generate_cot.py \
    --split train \
    --n 500 \
    --out results/hidden_states_train.jsonl \
    --resume
