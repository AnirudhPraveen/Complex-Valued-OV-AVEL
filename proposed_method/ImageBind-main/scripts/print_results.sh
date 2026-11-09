#!/bin/bash
# Print an Acc / Seg-F1 / Eve-F1 / Avg table from <root>/eval_{close,open,total}/*-Eval.log
ROOT=$1
printf "%-14s | %6s | %6s | %6s | %6s\n" "Split" "Acc" "Seg" "Eve" "Avg"
for SPLIT in close open total; do
    case $SPLIT in close) NAME="seen (close)";; open) NAME="unseen (open)";; total) NAME="total";; esac
    LOG=$(ls -t "${ROOT}/eval_${SPLIT}"/*-Eval.log 2>/dev/null | head -n 1 || true)
    LINE=$( [[ -n "${LOG}" ]] && grep "Evaluation results (acc)" "${LOG}" | tail -n 1 || true)
    if [[ -z "${LINE}" ]]; then printf "%-14s | %6s\n" "${NAME}" "N/A"; continue; fi
    echo "${LINE}" | awk -v n="${NAME}" '{
        match($0, /\(acc\):[ \t]*[0-9.]+/); a=substr($0,RSTART,RLENGTH); sub(/.*:[ \t]*/,"",a)
        match($0, /Segment-level F1score\):[ \t]*[0-9.]+/); s=substr($0,RSTART,RLENGTH); sub(/.*:[ \t]*/,"",s)
        match($0, /Event-level F1score\):[ \t]*[0-9.]+/); e=substr($0,RSTART,RLENGTH); sub(/.*:[ \t]*/,"",e)
        printf "%-14s | %6.1f | %6.1f | %6.1f | %6.1f\n", n, 100*a, 100*s, 100*e, 100*(a+s+e)/3 }'
done
