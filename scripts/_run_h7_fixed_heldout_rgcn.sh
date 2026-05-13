#!/bin/bash
# Held-out evaluation of H7-fixed (ordinal_loss_weight = 0.01) on the
# 4 RGCN+H6 seeds NOT used to choose the H7 weight. Seed 3 was the
# development seed and is NOT run by this script.
#
# Sequential : seeds 1, 2, 4, 5. Stops on first failure.
# ~14 min per seed × 4 ≈ 56 min wall-clock.
#
# Per the H7_HELDOUT_EVALUATION_PROTOCOL.md, OOD is never used in
# selection ; the val-only gate is applied per epoch.
set -euo pipefail

source /home/thom315/sheaf-MIRAS_LIght/.venv/bin/activate
cd /home/thom315/MBS-halting-h7
export PYTHONPATH=.

for SEED in 1 2 4 5; do
    OUT=results/claim_strengthening/h7_ordinal_halting/seed${SEED}_w001
    mkdir -p ${OUT}/checkpoints

    echo "===== held-out seed ${SEED} ====="
    python -m mbs.train \
        --config configs/h7_ordinal_halting/rgcn_h7_seed${SEED}_w001.yaml \
        --variant rgcn_h7_two_stage \
        --output-dir ${OUT} \
        --checkpoint-dir ${OUT}/checkpoints \
        > ${OUT}/run.log 2>&1
    echo "===== seed ${SEED} done ====="
done

echo "===== ALL HELD-OUT SEEDS DONE ====="
