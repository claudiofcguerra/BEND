#!/usr/bin/env bash
set -euo pipefail

RANK=${1:?rank missing}
WORLD=${2:?world size missing}

DISPLAY_RANK=$(( RANK + 1 ))
echo "🏁  Node $(hostname)  rank=$DISPLAY_RANK / $WORLD  started at $(date)"

cd "$HOME/BEND"

docker build -t bend_env .

mkdir -p logs results_variant_effects

EXPERIMENTS=(
  "variant_effects:dnabert2-curriculum-512-class:expression:dnabert2:./fine_tuned_models/dnabert_2_curriculum_patience_5_batch_size_60_max_length_512/curriculum_stage_class/merged_model:256:"
  "variant_effects:dnabert2-curriculum-512-class:disease:dnabert2:./fine_tuned_models/dnabert_2_curriculum_patience_5_batch_size_60_max_length_512/curriculum_stage_class/merged_model:256:"
)

TOTAL=${#EXPERIMENTS[@]}
echo "📊  Total experiments: $TOTAL, distributed across $WORLD nodes"
echo

idx=0
local_count=0
local_success=0
local_fail=0

for entry in "${EXPERIMENTS[@]}"; do
  if (( idx % WORLD == RANK )); then

    IFS=':' read -r phase model task model_type checkpoint embedding_idx extra_args <<< "$entry"

    echo "▶️  [$idx] ($phase) $model / $task"

    log_file=""
    ret=0

    case "$phase" in
      embedding)
        log_file="logs/${model}_${task}_$(hostname).out"
        embed_dir="data/${task}/${model}"
        echo "  🧹 Cleaning existing embedding chunks in ${embed_dir} (inside container)"
        set +e
        docker run --rm --gpus all --shm-size=100g \
          -v "$(pwd)":/app -w /app \
          bend_env \
          bash -lc "\
            rm -f /app/${embed_dir}/train_*.tar.gz /app/${embed_dir}/test_*.tar.gz /app/${embed_dir}/valid_*.tar.gz && \
            /root/.local/bin/poetry install --no-root && \
            /root/.local/bin/poetry run python scripts/precompute_embeddings.py \
              model=${model} task=${task} \
            > /app/${log_file} 2>&1"
        ret=$?
        set -e
        ;;

      training)
        log_file="logs/train_${model}_${task}_$(date +%Y%m%d_%H%M%S).log"
        training_dir="downstream_tasks/${task}/${model}"
        echo "  🧹 Cleaning existing training artifacts in ${training_dir} (inside container)"
        set +e
        docker run --rm --gpus all --shm-size=100g \
          -v "$(pwd)":/app -w /app \
          bend_env \
          bash -lc "\
            rm -rf /app/${training_dir} && \
            /root/.local/bin/poetry install --no-root && \
            /root/.local/bin/poetry run python scripts/train_on_task.py \
              --config-name ${task} embedder=${model} params.load_checkpoint=False \
            > /app/${log_file} 2>&1"
        ret=$?
        set -e
        ;;

      variant_effects)
        variant_file="data/variant_effects/variant_effects_${task}.bed"
        output_file="results_variant_effects/variant_effects_${task}_${model}.csv"
        log_file="logs/variant_effects_${model}_${task}_$(hostname).out"
        echo "  🧹 Cleaning existing variant output ${output_file} (inside container)"

        set +e
        docker run --rm --gpus all --shm-size=100g \
          -v "$(pwd)":/app -w /app \
          bend_env \
          bash -lc "\
            rm -f /app/${output_file} && \
            /root/.local/bin/poetry install --no-root && \
            /root/.local/bin/poetry run python scripts/predict_variant_effects.py \
              '${variant_file}' '${output_file}' \
              ${model_type} '${checkpoint}' \
              data/genomes/GRCh38.primary_assembly.genome.fa \
              --embedding_idx ${embedding_idx} ${extra_args} \
            > /app/${log_file} 2>&1"
        ret=$?
        set -e
        ;;
    esac

    if (( ret == 0 )); then
      echo "  ✅  $model / $task succeeded (log: $log_file)"
      local_success=$(( local_success + 1 ))
    else
      echo "  ⚠️  $model / $task FAILED (exit $ret; log: $log_file)" >&2
      local_fail=$(( local_fail + 1 ))
    fi

    local_count=$(( local_count + 1 ))
  fi
  idx=$(( idx + 1 ))
done

echo
echo "✅  Rank $RANK finished at $(date)"
echo "    Ran $local_count experiments: $local_success succeeded, $local_fail failed"
