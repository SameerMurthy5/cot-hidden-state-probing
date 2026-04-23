#!/bin/bash
# Setup script for cot-hidden-state-probing
set -e

conda create -n cot-probe python=3.10 -y
conda activate cot-probe
pip install -r requirements.txt

echo "Setup complete. Activate with: conda activate cot-probe"
