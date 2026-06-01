#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/BEND"

docker build -t bend_env .

mkdir -p logs

models=(dnabert2-curriculum-512-genus)
tasks=(chromatin_accessibility)

for t in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        echo "▶️  $model  /  $t"
        docker run --rm --gpus all --shm-size=100g \
        -v "$(pwd)":/app -w /app \
        bend_env \
        bash -lc "\
              /root/.local/bin/poetry install --no-root && \
              /root/.local/bin/poetry run python scripts/train_on_task.py \
                --config-name ${t} embedder=${model} params.load_checkpoint=True \
        > /app/logs/train_${model}_${t}_$(hostname).out 2>&1"
    done
    echo "✅  All training for task $t completed."
done