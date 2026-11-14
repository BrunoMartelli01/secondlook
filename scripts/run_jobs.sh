#!/bin/bash
# usage: scripts/run_jobs.sh JOBFILE NPAR     (xargs -P over run_job.sh; job files come from secondlook.reruns.exact.make_jobs)
cat "$1" | xargs -P "$2" -L 1 "$(cd "$(dirname "$0")" && pwd)/run_job.sh"
