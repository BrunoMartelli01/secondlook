#!/bin/bash
# usage: scripts/cross_chain.sh ARM EXPERT    (leave-one-dataset-out and pooled chain: dump -> [fit -> serve]* -> evaluate -> assemble)
# ARM exact loops fit/serve until a fit finishes with no cache miss. Assemble needs the arm's own main chain and stability outputs.
set -u
ARM=$1; E=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
O=${SECONDLOOK_RUNS:?set SECONDLOOK_RUNS}/rerun-exact
PY=${SECONDLOOK_PYTHON:-python}
CD=$O/$ARM/crossdata; mkdir -p "$CD" "$O/logs"
cd "$ROOT"
export PYTHONPATH=$ROOT OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
CROSS=secondlook.reruns.exact.cross_dataset
RUNPY="nice -n 10 $PY -m secondlook.reruns.exact.launch $ARM"
for t in totaltext icdar2015 ctw1500; do
  [ -f "$CD/pool_${E}_$t.pkl" ] || $RUNPY $CROSS dump --expert $E --task $t --out "$CD" > "$O/logs/cdump_${ARM}_${E}_$t.log" 2>&1 &
done; wait
it=0
while [ ! -f "$CD/fits_$E.pkl" ]; do
  it=$((it+1)); echo "CROSS $ARM $E fit iteration $it $(date +%T)" >> "$O/chain.log"
  $RUNPY $CROSS fit --expert $E --out "$CD" >> "$O/logs/cfit_${ARM}_$E.log" 2>&1; rc=$?
  [ -f "$CD/fits_$E.pkl" ] && break
  if [ $rc -ne 75 ] || [ $it -gt 10 ]; then echo "CROSS $ARM $E FAILED rc=$rc it=$it" >> "$O/chain.log"; exit 1; fi
  for t in totaltext icdar2015 ctw1500; do
    $RUNPY $CROSS serve --expert $E --task $t --out "$CD" >> "$O/logs/cserve_${ARM}_${E}_$t.log" 2>&1 &
  done; wait
done
for t in totaltext icdar2015 ctw1500; do
  [ -f "$CD/eval_${E}_$t.npz" ] || $RUNPY $CROSS evaluate --expert $E --task $t --out "$CD" > "$O/logs/ceval_${ARM}_${E}_$t.log" 2>&1 &
done; wait
until [ -f "$O/$ARM/stability/${E}_totaltext/stability.json" ] && [ -f "$O/$ARM/stability/${E}_icdar2015/stability.json" ] && [ -f "$O/$ARM/stability/${E}_ctw1500/stability.json" ] \
   && [ -f "$O/$ARM/main/p3/counts_${E}_ctw1500_clean.npz" ] && [ -f "$O/$ARM/main/p3/counts_${E}_totaltext_clean.npz" ] && [ -f "$O/$ARM/main/p3/counts_${E}_icdar2015_clean.npz" ]; do sleep 60; done
$RUNPY $CROSS assemble --expert $E --out "$CD" > "$O/logs/cassemble_${ARM}_${E}.log" 2>&1
echo "CROSS $ARM $E DONE rc=$? $(date +%T)" >> "$O/chain.log"
