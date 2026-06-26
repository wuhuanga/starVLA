#!/usr/bin/env bash
# RIDE-VLA Experiment Scripts  (2 datasets × 4 variants = 8 train jobs)
#
# ┌─────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
# │                 │  Base (none) │  Lang only   │ Visual only  │  Full (both) │
# ├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
# │ LIBERO          │ libero_1     │ libero_2     │ libero_3     │ libero_4     │
# │ SimplerEnv      │ simpleenv_1  │ simpleenv_2  │ simpleenv_3  │ simpleenv_4  │
# └─────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
#
# TRAINING (run from repo root):
#   bash RIDE_VLA/train/libero_1_base.sh
#   bash RIDE_VLA/train/libero_2_lang_only.sh
#   bash RIDE_VLA/train/libero_3_visual_only.sh
#   bash RIDE_VLA/train/libero_4_full.sh
#   bash RIDE_VLA/train/simpleenv_1_base.sh
#   bash RIDE_VLA/train/simpleenv_2_lang_only.sh
#   bash RIDE_VLA/train/simpleenv_3_visual_only.sh
#   bash RIDE_VLA/train/simpleenv_4_full.sh
#
# EVALUATION (pass checkpoint path as $1):
#   bash RIDE_VLA/eval/eval_libero_standard.sh        <ckpt>          # Spatial/Object/Goal/Long
#   bash RIDE_VLA/eval/eval_libero_robustness.sh      <ckpt> [suite]  # Clean/Lang/Visual/Both splits
#   bash RIDE_VLA/eval/eval_simpleenv_clean.sh        <ckpt>          # 4 WidowX tasks
#   bash RIDE_VLA/eval/eval_simpleenv_perturbed.sh    <ckpt>          # Lang/Visual/Both splits
#   bash RIDE_VLA/eval/eval_consistency_diagnostic.sh <ckpt> [n]      # Rep.Dist + Action.Dist

echo "Reference file only — not meant to be executed."
