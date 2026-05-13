#!/bin/bash
# V2 — seed 3, ordinal_loss_weight = 0.005. Single Stage-2 run from H6
# seed-3 Stage-1 best.pt (init via symlink to main repo).
set -euo pipefail
source /home/thom315/sheaf-MIRAS_LIght/.venv/bin/activate
cd /home/thom315/MBS-halting-h7
export PYTHONPATH=.

OUT=results/claim_strengthening/h7_ordinal_halting/seed3_w0005
mkdir -p ${OUT}/checkpoints

python -m mbs.train \
    --config configs/h7_ordinal_halting/rgcn_h7_seed3_w0005.yaml \
    --variant rgcn_h7_two_stage \
    --output-dir ${OUT} \
    --checkpoint-dir ${OUT}/checkpoints \
    > ${OUT}/run.log 2>&1
echo "V2 done."
