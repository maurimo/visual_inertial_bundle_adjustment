#!/bin/bash
# Prepare, run and evaluate VIBA on EuRoC / TUM-VI from several sources and camera models.
#
#   run_experiments.sh prepare [regex]   write the VIBA input folders (cheap, no VIBA run), and check them
#   run_experiments.sh run [regex]       run ark_vi_ba on the prepared inputs (sequential unless JOBS > 1)
#   run_experiments.sh eval [regex]      ATE of inputs and outputs, table in $WORK/summary.txt
#   run_experiments.sh list [regex]      list the experiments
#
# Experiments (name = <source>_<sequence>_<variant>_<camera model>):
#   sd_<seq>_{input,final}_{kb,ds,eucm}   SLAM Dunk's global BA, TUM-VI (8 sequences), from its input or
#                                         its result; kb = Kalibr Kannala-Brandt, ds / eucm = Basalt's
#                                         Double Sphere / EUCM calibrations
#   trk_<seq>_basalt_{kb,ds,eucm}         KLT tracks (sliding_marg/real anc_st), TUM-VI (12 sequences),
#                                         initialized from Basalt's VIO
#   trk_<V1_0x>_ours_{radtan,ds,eucm}     KLT tracks, EuRoC V1_01 / V1_02, initialized from our VIO
#   colmap_<name>_<align>_<model>         COLMAP models listed in $COLMAP_MODELS (see below)
#
# Environment (defaults in brackets):
#   WORK          output root [$PWD/viba_runs]
#   VIBA_BIN      [build/interfaces/ark/ark_vi_ba]          VIBA_ARGS  extra ark_vi_ba arguments
#   JOBS          parallel VIBA runs [1] (keep 1 for timings)
#   KF_DT         keyframe spacing of the track inputs [s] [0.2]
#   SLAMDUNK      [/galway/data/slam_dunk]    TUMVI [/galway/data/tumvi]    BASALT [~/vio_ref/basalt]
#   REAL          sliding_marg/real folder (tracks, inits, eval_traj.py)
#   COLMAP_MODELS space separated <name>:<model dir>:<ASL dataset dir>[:<align>] (align: imu | none)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
stage=${1:-list}
filter=${2:-.}
WORK=${WORK:-$PWD/viba_runs}
VIBA_BIN=${VIBA_BIN:-$repo/build/interfaces/ark/ark_vi_ba}
VIBA_ARGS=${VIBA_ARGS:-}
JOBS=${JOBS:-1}
KF_DT=${KF_DT:-0.2}
SLAMDUNK=${SLAMDUNK:-/galway/data/slam_dunk}
TUMVI=${TUMVI:-/galway/data/tumvi}
BASALT=${BASALT:-$HOME/vio_ref/basalt}
REAL=${REAL:-$HOME/git4/experiments/msckf_curvature_cpp/causal/sliding_marg/real}
EVAL=${EVAL:-$REAL/eval_traj.py}
COLMAP_MODELS=${COLMAP_MODELS:-}
PY=${PY:-python3}

declare -A SD_TAG=([room1]=run2 [corridor1]=full [slides1]=full2 [magistrale1]=full
                   [outdoors5]=full [outdoors6]=full [outdoors7]=full [outdoors8]=full)
TUMVI_SEQS="room1 corridor1 slides1 magistrale1 outdoors1 outdoors2 outdoors3 outdoors4 outdoors5 outdoors6 outdoors7 outdoors8"
EUROC_SEQS="V1_01_easy V1_02_medium"
declare -A EUROC_INIT=([V1_01_easy]=v101/lift_w6.txt [V1_02_medium]=v102/lift_w6.txt)
declare -A EUROC_TRACKS=([V1_01_easy]=v101_st.txt [V1_02_medium]=v102_st.txt)

tumvi_calib() {  # <model>: --calib argument for a TUM-VI camera model
  case $1 in kb) ;; ds) echo "--calib $BASALT/data/tumvi_512_ds_calib.json" ;;
    eucm) echo "--calib $BASALT/data/tumvi_512_eucm_calib.json" ;; esac
}
euroc_calib() {
  case $1 in radtan) ;; ds) echo "--calib $BASALT/data/euroc_ds_calib.json" ;;
    eucm) echo "--calib $BASALT/data/euroc_eucm_calib.json" ;; esac
}

# one line per experiment: <name>|<mav0 for evaluation>|<prepare command>|<reference trajectories>
experiments() {
  local seq tag init m d
  for seq in room1 corridor1 slides1 magistrale1 outdoors5 outdoors6 outdoors7 outdoors8; do
    tag=${SD_TAG[$seq]}
    d=$TUMVI/dataset-${seq}_512_16
    for init in input final; do
      for m in kb ds eucm; do
        echo "sd_${seq}_${init}_${m}|$d/mav0|$PY $repo/tools/slamdunk_to_viba/slamdunk_to_viba.py --run $SLAMDUNK/results/$seq/$tag --rec $SLAMDUNK/rec/$seq --dataset $d --init $init $(tumvi_calib $m) --out|sd:$SLAMDUNK/results/$seq/$tag:$SLAMDUNK/rec/$seq"
      done
    done
  done
  for seq in $TUMVI_SEQS; do
    d=$TUMVI/dataset-${seq}_512_16
    for m in kb ds eucm; do
      echo "trk_${seq}_basalt_${m}|$d/mav0|$PY $here/asl_tracks_to_viba.py --dataset $d --tracks $REAL/data/tumvi/${seq}anc_st.txt --traj $REAL/out/cmp/$seq/ref_basalt.txt --kf-dt $KF_DT $(tumvi_calib $m) --out|traj:$REAL/out/cmp/$seq/ref_basalt.txt"
    done
  done
  for seq in $EUROC_SEQS; do
    d=$REAL/data/$seq
    for m in radtan ds eucm; do
      echo "trk_${seq}_ours_${m}|$d/mav0|$PY $here/asl_tracks_to_viba.py --dataset $d --tracks $REAL/data/${EUROC_TRACKS[$seq]} --traj $REAL/out/${EUROC_INIT[$seq]} --kf-dt $KF_DT $(euroc_calib $m) --out|traj:$REAL/out/${EUROC_INIT[$seq]}"
    done
  done
  local entry name model ds align
  for entry in $COLMAP_MODELS; do
    IFS=: read -r name model ds align <<< "$entry"
    align=${align:-imu}
    echo "colmap_${name}_${align}_colmap|$ds/mav0|$PY $here/colmap_to_viba.py --model $model --dataset $ds --align $align --out|"
  done
}

selected() { experiments | grep -E "^[^|]*($filter)[^|]*\|" || true; }

prepare_one() {
  local name=$1 cmd=$2 dir=$WORK/$1
  mkdir -p "$dir"
  if $cmd "$dir/input" > "$dir/prepare.log" 2>&1 && \
     $PY "$here/check_viba_input.py" "$dir/input" --points 200 --max-span-kf 10 > "$dir/check.txt" 2>&1; then
    echo "$name: $(tail -n +2 "$dir/check.txt" | sed 's/reprojection error //; s/^ *//' | tr '\n' ' ')"
  else
    echo "$name: FAILED, see $dir/prepare.log"
  fi
}

run_one() {
  local name=$1 dir=$WORK/$1
  [[ -d $dir/input ]] || { echo "$name: not prepared"; return; }
  /usr/bin/time -f "wall %e s, maxrss %M KB" "$VIBA_BIN" -i "$dir/input" -o "$dir/output" $VIBA_ARGS \
    > "$dir/viba.log" 2>&1 || { echo "$name: VIBA FAILED, see $dir/viba.log"; return; }
  $PY "$repo/tools/slamdunk_to_viba/export_trajectories.py" viba "$dir/output" "$dir/viba.txt"
  echo "$name: $(tail -n 1 "$dir/viba.log")"
}

ate() {  # <mav0> <trajectory>: ATE se3 [m] as printed by eval_traj.py
  local v
  v=$($PY "$EVAL" "$1" "$2" 2>/dev/null | sed -n 's/.*ATE se3 \([0-9.]*\) m.*/\1/p' | head -1)
  echo "${v:--}"
}

eval_one() {
  local name=$1 mav0=$2 ref=$3 dir=$WORK/$1 init_ate ref_ate viba_ate
  [[ -f $dir/viba.txt ]] || return 0
  if [[ $ref == sd:* ]]; then
    IFS=: read -r _ run rec <<< "$ref"
    local which=slamdunk-init; [[ $name == *_final_* ]] && which=slamdunk-ba
    $PY "$repo/tools/slamdunk_to_viba/export_trajectories.py" $which "$run" "$rec" "$dir/init.txt"
    $PY "$repo/tools/slamdunk_to_viba/export_trajectories.py" slamdunk-ba "$run" "$rec" "$dir/sd_ba.txt"
    init_ate=$(ate "$mav0" "$dir/init.txt"); ref_ate=$(ate "$mav0" "$dir/sd_ba.txt")
  elif [[ $ref == traj:* ]]; then
    init_ate=$(ate "$mav0" "${ref#traj:}"); ref_ate=-
  else
    init_ate=-; ref_ate=-
  fi
  viba_ate=$(ate "$mav0" "$dir/viba.txt")
  printf "%-36s %10s %10s %10s\n" "$name" "$init_ate" "$viba_ate" "$ref_ate"
}

case $stage in
  list) selected | cut -d'|' -f1 ;;
  prepare)
    mkdir -p "$WORK"
    selected | while IFS='|' read -r name mav0 cmd ref; do prepare_one "$name" "$cmd"; done ;;
  run)
    [[ -x $VIBA_BIN ]] || { echo "no VIBA binary at $VIBA_BIN"; exit 1; }
    export -f run_one; export WORK VIBA_BIN VIBA_ARGS PY repo
    selected | cut -d'|' -f1 | xargs -P "$JOBS" -I{} bash -c 'run_one {}' ;;
  eval)
    { printf "%-36s %10s %10s %10s\n" experiment "init ATE" "VIBA ATE" "SD BA ATE"
      selected | while IFS='|' read -r name mav0 cmd ref; do eval_one "$name" "$mav0" "$ref"; done
    } | tee "$WORK/summary.txt" ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
