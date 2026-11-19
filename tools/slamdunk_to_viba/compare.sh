#!/bin/bash
# Run VIBA on a SLAM Dunk run of a TUM-VI sequence and evaluate against SLAM Dunk's global BA.
#
#   compare.sh <seq> <slam dunk run tag> <work dir> [extra ark_vi_ba args...]
#
# Environment: SLAMDUNK (default /galway/data/slam_dunk), TUMVI (default /galway/data/tumvi),
# VIBA_BIN (default build/interfaces/ark/ark_vi_ba), EVAL (eval_traj.py, required).
set -euo pipefail
seq=$1; tag=$2; work=$3; shift 3
here=$(cd "$(dirname "$0")" && pwd)
SLAMDUNK=${SLAMDUNK:-/galway/data/slam_dunk}
TUMVI=${TUMVI:-/galway/data/tumvi}
VIBA_BIN=${VIBA_BIN:-$here/../../build/interfaces/ark/ark_vi_ba}
run=$SLAMDUNK/results/$seq/$tag
rec=$SLAMDUNK/rec/$seq
tumvi=$TUMVI/dataset-${seq}_512_16
out=$work/$seq
mkdir -p "$out"

python3 "$here/export_trajectories.py" slamdunk-init "$run" "$rec" "$out/sd_init.txt"
python3 "$here/export_trajectories.py" slamdunk-ba "$run" "$rec" "$out/sd_ba.txt"
for init in input final; do
  python3 "$here/slamdunk_to_viba.py" --run "$run" --rec "$rec" --tumvi "$tumvi" \
    --out "$out/in_$init" --init "$init"
  /usr/bin/time -f "wall %e s, maxrss %M KB" "$VIBA_BIN" -i "$out/in_$init" -o "$out/out_$init" "$@" \
    > "$out/viba_$init.log" 2>&1
  tail -n 1 "$out/viba_$init.log"
  python3 "$here/export_trajectories.py" viba "$out/out_$init" "$out/viba_from_$init.txt"
done
python3 "$EVAL" "$tumvi/mav0" "$out/sd_init.txt" "$out/viba_from_input.txt" "$out/sd_ba.txt" \
  "$out/viba_from_final.txt" | tee "$out/eval.txt"
