#!/bin/bash
# Show progress of the reduced-design run.
cd "/Users/nicolas/Desktop/Cambrdige RP/Project/publications-code/DoubleCrossFit/Python" || exit 1
echo "=== processes ==="
pgrep -fl "reduced_design.py" | grep -o "start [0-9]* --end [0-9]*" || echo "  none running"
echo "=== progress ==="
for f in reduced_results_0001_0500.csv reduced_results_0501_1000.csv; do
  [ -f "$f" ] && printf "  %s: %d datasets done\n" "$f" "$(( $(wc -l < "$f") - 1 ))"
done
