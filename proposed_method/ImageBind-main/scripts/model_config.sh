# Sourced by the run_*.sh scripts: turns STREAMS / FUSION / PHASE_MODE into
# CFG_OVERRIDES_JSON (see config.py) and a RUN_NAME.
STREAMS=${STREAMS:-4}
FUSION=${FUSION:-complex}
PHASE_MODE=${PHASE_MODE:-cyclegan}

case "${FUSION}" in
  complex|real) FUSION_ENABLED=true; FUSION_TYPE=${FUSION} ;;
  none)         FUSION_ENABLED=false; FUSION_TYPE=complex; STREAMS=2 ;;
  *) echo "FUSION must be complex|real|none (got ${FUSION})" >&2; exit 2 ;;
esac
case "${STREAMS}" in
  4) AUX=true ;;
  2) AUX=false ;;
  *) echo "STREAMS must be 2|4 (got ${STREAMS})" >&2; exit 2 ;;
esac
case "${PHASE_MODE}" in
  cyclegan|imagebind) ;;
  *) echo "PHASE_MODE must be cyclegan|imagebind (got ${PHASE_MODE})" >&2; exit 2 ;;
esac

CFG_OVERRIDES_JSON="{\"IHSV.ENABLED\": ${AUX}, \"PHASE.ENABLED\": ${AUX}, \"PHASE.MODE\": \"${PHASE_MODE}\", \"FUSION_CNN_ENABLED\": ${FUSION_ENABLED}, \"FUSION_CNN_TYPE\": \"${FUSION_TYPE}\"}"

if [[ "${FUSION}" == "none" ]]; then
  RUN_NAME="finetuned_baseline"
else
  RUN_NAME="${STREAMS}stream_${FUSION}"
  if [[ "${STREAMS}" == "4" && "${PHASE_MODE}" == "imagebind" ]]; then RUN_NAME="${RUN_NAME}_rawIF"; fi
fi
