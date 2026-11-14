#!/bin/bash
# usage: scripts/run_job.sh ARM LOGNAME MODULE [module args...]      ARM = fast | exact
# Runs one analysis module through the layer launcher (CPU only, 2 threads, nice 10).
# Log: $SECONDLOOK_RUNS/rerun-exact/logs/LOGNAME.log, progress: $SECONDLOOK_RUNS/rerun-exact/chain.log
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${SECONDLOOK_RUNS:?set SECONDLOOK_RUNS}/rerun-exact
PY=${SECONDLOOK_PYTHON:-python}
cd "$ROOT"
export PYTHONPATH=$ROOT OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
ARM=$1; LOG=$2; shift 2
mkdir -p "$RUN/logs" "$RUN/$ARM"
echo "START $LOG $(date +%T)" >> "$RUN/chain.log"
nice -n 10 "$PY" -m secondlook.reruns.exact.launch "$ARM" "$@" > "$RUN/logs/$LOG.log" 2>&1
echo "DONE $LOG rc=$? $(date +%T)" >> "$RUN/chain.log"
