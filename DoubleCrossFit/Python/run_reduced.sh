#!/bin/bash
# Launch (or resume) the reduced-design run, FULLY DETACHED so it survives the terminal /
# Claude session closing. Safe to run repeatedly: the driver skips datasets already in each
# output CSV, so re-running just continues from where it stopped.
#
#   Usage:  bash run_reduced.sh          # start / resume both chunks
#           bash check_reduced.sh        # see progress

cd "/Users/nicolas/Desktop/Cambrdige RP/Project/publications-code/DoubleCrossFit/Python" || exit 1
PY=".venv-pinned/bin/python"

launch () {  # $1=start $2=end $3=outfile $4=logfile
  if pgrep -f "reduced_design.py --start $1 --end $2" >/dev/null; then
    echo "chunk $1-$2 already running"; return
  fi
  # macOS-compatible detachment: nohup + background + disown -> orphaned to launchd,
  # survives the terminal / Claude session closing.
  # caffeinate flags: -i idle sleep, -m disk sleep, -s system sleep (while on AC power).
  # (-i alone is NOT enough: the machine slept mid-run on 2026-07-23 and killed the jobs.)
  nohup caffeinate -ims "$PY" -W ignore -u reduced_design.py \
      --start "$1" --end "$2" --k 10 --rf-jobs 2 --out "$3" >> "$4" 2>&1 < /dev/null &
  disown
  echo "launched chunk $1-$2 (detached, pid $!)"
}

launch 1   500  reduced_results_0001_0500.csv reduced_chunkA.log
launch 501 1000 reduced_results_0501_1000.csv reduced_chunkB.log
sleep 2
echo "running processes:"; pgrep -fl "reduced_design.py" | grep -o "start [0-9]* --end [0-9]*"
