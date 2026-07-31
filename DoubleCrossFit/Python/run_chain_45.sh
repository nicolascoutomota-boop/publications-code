#!/bin/bash
# Auto-chain: wait for study 3c to finish, then run study 4 (n=1500), then study 5 (RF).
cd "/Users/nicolas/Desktop/Cambrdige RP/Project/publications-code/DoubleCrossFit/Python" || exit 1
PY=".venv-pinned/bin/python"
# wait for 3c
while pgrep -f "out study3c_results" >/dev/null; do sleep 120; done
echo "$(date '+%F %T') 3c finished; launching study 4" >> chain_45.log
caffeinate -ims "$PY" -W ignore -u e0_studies.py --start 1 --end 500 --ml sl --n 1500 --rf-jobs 2 --out study4_results_0001_0500.csv >> study4_chunkA.log 2>&1 &
caffeinate -ims "$PY" -W ignore -u e0_studies.py --start 501 --end 1000 --ml sl --n 1500 --rf-jobs 2 --out study4_results_0501_1000.csv >> study4_chunkB.log 2>&1 &
wait
echo "$(date '+%F %T') study 4 finished; launching study 5" >> chain_45.log
caffeinate -ims "$PY" -W ignore -u e0_studies.py --start 1 --end 500 --ml rf --rf-jobs 2 --out study5_results_0001_0500.csv >> study5_chunkA.log 2>&1 &
caffeinate -ims "$PY" -W ignore -u e0_studies.py --start 501 --end 1000 --ml rf --rf-jobs 2 --out study5_results_0501_1000.csv >> study5_chunkB.log 2>&1 &
wait
echo "$(date '+%F %T') study 5 finished; chain complete" >> chain_45.log
