#!/bin/bash
# Paper ablations (unseen split is the one reported):
#   (1) Complex vs Real fusion at matched parameters, 2- and 4-stream
#   (2) Audio imaginary stream: CycleGAN-translated phase vs raw IF
#
#   OVAVEL_DATASET=ovavel|ave_ov  GPU=0  SEED=123  bash run_ablations.sh
set -euo pipefail
cd "$(dirname "$0")"
STREAMS=2 FUSION=real    bash run_complex_fusion.sh
STREAMS=2 FUSION=complex bash run_complex_fusion.sh
STREAMS=4 FUSION=real    bash run_complex_fusion.sh
STREAMS=4 FUSION=complex bash run_complex_fusion.sh
STREAMS=4 FUSION=complex PHASE_MODE=imagebind bash run_complex_fusion.sh
