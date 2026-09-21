#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
python_bin="$project_dir/.venv/bin/python"
fresh=false

if [[ "${1:-}" == "--fresh" ]]; then
  fresh=true
  shift
fi

train_rounds="${1:-1000}"
eval_rounds="${2:-100}"
train_seed=2026
eval_seeds=(11 22 33 44 55)
agents=(tabular_q_agent tabular_sarsa_agent)

cd "$project_dir"

# Pygame is imported by the framework even in --no-gui mode. A dummy SDL video
# driver keeps headless runs portable across macOS terminals and CI machines.
export SDL_VIDEODRIVER=dummy

if [[ "$fresh" == true ]]; then
  rm -f \
    agent_code/tabular_q_agent/q_table.pkl \
    agent_code/tabular_sarsa_agent/q_table.pkl
fi

for agent in "${agents[@]}"; do
  echo "Training $agent for $train_rounds rounds..."
  "$python_bin" main.py play \
    --agents "$agent" \
    --train 1 \
    --scenario coin-heaven \
    --no-gui \
    --n-rounds "$train_rounds" \
    --seed "$train_seed"
done

for agent in "${agents[@]}"; do
  for seed in "${eval_seeds[@]}"; do
    echo "Evaluating $agent with seed $seed..."
    "$python_bin" main.py play \
      --agents "$agent" \
      --scenario coin-heaven \
      --no-gui \
      --n-rounds "$eval_rounds" \
      --seed "$seed" \
      --save-stats "results/stage1_${agent}_${seed}.json"
  done
done

"$python_bin" scripts/summarize_stage1.py \
  --rounds-per-seed "$eval_rounds" \
  --seeds "${eval_seeds[@]}"
