#!/bin/bash
# Run RGCN+H6 two-stage protocol on seeds 2..5 sequentially.
# Each seed: Stage 1 (controller-only, init from rgcn_act_postpatch best.pt),
#            Stage 2 (co-train halting_controller + claim_selector_head,
#                     init from this seed's Stage 1 best.pt).
# Output goes to results/claim_strengthening/rgcn_h6_two_stage/seed{N}/.

set -euo pipefail
source /home/thom315/sheaf-MIRAS_LIght/.venv/bin/activate
cd /home/thom315/MBS-halting
export PYTHONPATH=.

ROOT=results/claim_strengthening/rgcn_h6_two_stage

for SEED in 2 3 4 5; do
    SD=${ROOT}/seed${SEED}
    mkdir -p ${SD}/stage1/checkpoints ${SD}/stage2/checkpoints

    echo "===== seed ${SEED} Stage 1 ====="
    python -m mbs.train \
        --config configs/rgcn_h6_stage1_seed${SEED}.yaml \
        --variant rgcn_h6_two_stage \
        --output-dir ${SD}/stage1 \
        --checkpoint-dir ${SD}/stage1/checkpoints \
        > ${SD}/stage1/run.log 2>&1

    echo "===== seed ${SEED} Stage 2 ====="
    python -m mbs.train \
        --config configs/rgcn_h6_stage2_seed${SEED}.yaml \
        --variant rgcn_h6_two_stage \
        --output-dir ${SD}/stage2 \
        --checkpoint-dir ${SD}/stage2/checkpoints \
        > ${SD}/stage2/run.log 2>&1

    echo "===== seed ${SEED} done ====="
done

echo "===== ALL SEEDS 2..5 DONE ====="
