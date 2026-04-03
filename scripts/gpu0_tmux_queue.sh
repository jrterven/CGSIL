#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/juan/dev/CGSIL"
LOG_DIR="$ROOT/logs/tmux"
mkdir -p "$LOG_DIR"
GPU=0
run_train() {
  local name="$1"
  shift
  local metrics_file="$ROOT/outputs/$name/metrics.csv"
  if [[ -f "$metrics_file" ]]; then
    local line_count
    line_count=$(wc -l < "$metrics_file")
    if [[ "$line_count" -ge 201 ]]; then
      echo "[$(date -Iseconds)] SKIP $name (metrics.csv already complete with $line_count lines)"
      return 0
    fi
  fi
  echo "[$(date -Iseconds)] START $name"
  CUDA_VISIBLE_DEVICES="$GPU" conda run -n cgsil python "$ROOT/train_cifar_lt.py" \
    --data-root "$ROOT/data" \
    --output-dir "$ROOT/outputs" \
    --epochs 200 \
    --batch-size 128 \
    --num-workers 4 \
    --download \
    --device cuda \
    --experiment-name "$name" \
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date -Iseconds)] DONE $name"
}
run_train "cifar10_cgsilv2_legacy_ir10_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 10 --tail-quantile 0.3 --beta-start 0.8 --beta-end 0.9 --cgsil-on-top-of ce --surgery-scope all --warmup-epochs 0 --min-tail-samples 1 --min-tail-classes 1 --conflict-threshold 0.0
run_train "cifar10_cgsilv2_main_ir10_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 10 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_base_ce_ir10_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 10 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_base_focal_ir10_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 10 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of focal --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_scope_all_ir10_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 10 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope all --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_nowarmup_ir10_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 10 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 0 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_loose_tail_ir10_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 10 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 1 --min-tail-classes 1 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_no_threshold_ir10_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 10 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold 0.0
run_train "cifar10_cgsilv2_legacy_ir50_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 50 --tail-quantile 0.3 --beta-start 0.8 --beta-end 0.9 --cgsil-on-top-of ce --surgery-scope all --warmup-epochs 0 --min-tail-samples 1 --min-tail-classes 1 --conflict-threshold 0.0
run_train "cifar10_cgsilv2_main_ir50_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 50 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_base_ce_ir50_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 50 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_base_focal_ir50_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 50 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of focal --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_scope_all_ir50_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 50 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope all --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_nowarmup_ir50_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 50 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 0 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_loose_tail_ir50_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 50 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 1 --min-tail-classes 1 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_no_threshold_ir50_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 50 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold 0.0
run_train "cifar10_cgsilv2_legacy_ir100_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 100 --tail-quantile 0.3 --beta-start 0.8 --beta-end 0.9 --cgsil-on-top-of ce --surgery-scope all --warmup-epochs 0 --min-tail-samples 1 --min-tail-classes 1 --conflict-threshold 0.0
run_train "cifar10_cgsilv2_main_ir100_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 100 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_base_ce_ir100_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 100 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_base_focal_ir100_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 100 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of focal --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_scope_all_ir100_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 100 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope all --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_nowarmup_ir100_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 100 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 0 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_loose_tail_ir100_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 100 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 1 --min-tail-classes 1 --conflict-threshold -0.05
run_train "cifar10_cgsilv2_ablate_no_threshold_ir100_seed42" --dataset cifar10 --method cgsil --imbalance-ratio 100 --tail-quantile 0.3 --beta-start 0.6 --beta-end 0.8 --cgsil-on-top-of weighted_ce --surgery-scope fc --warmup-epochs 50 --min-tail-samples 8 --min-tail-classes 2 --conflict-threshold 0.0
