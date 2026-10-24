#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_id="${GPU_ID:-0}"
num_workers="${NUM_WORKERS:-8}"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd} CHECKPOINT [state_dict|checkpoint]\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) dataset_root="$data_root/LEVIR_CD"; image_a="A"; image_b="B"; label_dir="label" ;;
  levircdplus) dataset_root="$data_root/LEVIR_CD_PLUS/cut_data"; image_a="A"; image_b="B"; label_dir="label" ;;
  cdd) dataset_root="$data_root/CDD"; image_a="A"; image_b="B"; label_dir="OUT" ;;
  whucd) dataset_root="$data_root/WHUCD"; image_a="A"; image_b="B"; label_dir="label" ;;
  sysucd) dataset_root="$data_root/SYSU_CD"; image_a="time1"; image_b="time2"; label_dir="label" ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

checkpoint="$(realpath "$2")"
checkpoint_format="${3:-state_dict}"
if [[ ! -f "$checkpoint" ]]; then
  printf 'Checkpoint not found: %s\n' "$checkpoint" >&2
  exit 1
fi
if [[ "$checkpoint_format" != state_dict && "$checkpoint_format" != checkpoint ]]; then
  printf 'Checkpoint format must be state_dict or checkpoint.\n' >&2
  exit 2
fi
for split in train test; do
  for folder in "$image_a" "$image_b" "$label_dir"; do
    if [[ ! -d "$dataset_root/$split/$folder" ]]; then
      printf 'Dataset directory missing: %s\n' "$dataset_root/$split/$folder" >&2
      exit 1
    fi
  done
done

timestamp="$(date +%Y%m%d_%H%M%S)"
work_dir="$repo_root/FHRB-Net_work_dirs/$1/inference_$timestamp"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu_id"
python inference.py \
  --dataset-root "$dataset_root" \
  --image-dir-a "$image_a" \
  --image-dir-b "$image_b" \
  --label-dir "$label_dir" \
  --checkpoint "$checkpoint" \
  --checkpoint-format "$checkpoint_format" \
  --work-dir "$work_dir" \
  --num-workers "$num_workers"
