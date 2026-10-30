#!/usr/bin/env bash
# Train and analyse every reported run locally, then compare each experiment across seeds.
#
# Start it with `just bg sweep`. Safe to rerun: a run whose weights exist is skipped, and an
# unfinished run's folder is moved to runs/failed/ before it is trained again.
#
# Two queues run side by side: the small sweeps (intro, polygons, phase) on the CPU, one
# thread each, and the dimensionality sweeps on the GPU, a few at a time (about 1.2 GB each).
# CPU runs get no GPU at all: PyTorch's Adam otherwise opens a CUDA context (about 100 MB)
# per process even for CPU tensors.
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"

CPU_JOBS=${CPU_JOBS:-12}
GPU_JOBS=${GPU_JOBS:-3}
SEEDS=${SEEDS:-'0 1 2 3 4'}
mkdir -p runs/logs runs/failed

run_one() {
  local experiment=$1 model=$2 seed=$3 device=$4
  local activation=$model; [ "$model" = linear ] && activation=none
  local name=$experiment-$model-seed$seed dir=runs/$experiment-$model-seed$seed
  if [ ! -f "$dir/model.safetensors" ]; then
    if [ -e "$dir" ]; then
      mv "$dir" "runs/failed/$name-$(date +%Y%m%d-%H%M%S)"
    fi
    if [ "$device" = cpu ]; then
      CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run train.py --experiment "$experiment" \
        --model.activation "$activation" --seed "$seed" --device cpu --run-name "$name" > "runs/logs/$name.log" 2>&1
    else
      uv run train.py --experiment "$experiment" --model.activation "$activation" --seed "$seed" \
        --device cuda --run-name "$name" > "runs/logs/$name.log" 2>&1
    fi || { echo "FAILED $name (runs/logs/$name.log)"; return 0; }
  fi
  if [ ! -e "$dir/analysis" ]; then
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run analyze.py --run "$dir" >> "runs/logs/$name.log" 2>&1 \
      || { echo "FAILED analysis of $name"; return 0; }
  fi
  echo "done $name"
}
export -f run_one

jobs() {
  local device=$1; shift
  for seed in $SEEDS; do for model in relu linear; do for experiment in "$@"; do
    echo "$experiment $model $seed $device"
  done; done; done
}

echo "sweep started $(date)"
jobs cpu intro polygons phase | xargs -P "$CPU_JOBS" -L 1 bash -c 'run_one "$@"' _ &
jobs cuda dimensionality | xargs -P "$GPU_JOBS" -L 1 bash -c 'run_one "$@"' _ &
wait

for experiment in intro polygons phase dimensionality; do
  if [ ! -e "runs/compare-$experiment" ]; then
    uv run compare.py --experiment "$experiment" --seeds $SEEDS || echo "FAILED compare $experiment"
  fi
done
echo "sweep finished $(date)"
