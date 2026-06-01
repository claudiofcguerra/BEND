#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/BEND"

docker build -t bend_env .

mkdir -p logs

models=(dnabert2-no-overlap-curriculum)
tasks=(gene_finding enhancer_annotation histone_modification chromatin_accessibility cpg_methylation)

for model in "${models[@]}"; do
    for t in "${tasks[@]}"; do
        echo "▶️  $model  /  $t"
        docker run --rm --gpus all \
            -v "$(pwd)":/app -w /app \
            -e HYDRA_FULL_ERROR=1 \
            bend_env \
            bash -lc "\
          /root/.local/bin/poetry install --no-root && \
          /root/.local/bin/poetry run python scripts/precompute_embeddings.py \
            model=${model} task=${t} \
          > /app/logs/${model}_${t}_$(hostname).out 2>&1"
    done
    echo "✅  All embeddings for $model completed."
done
