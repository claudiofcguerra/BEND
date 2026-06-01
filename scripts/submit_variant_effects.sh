#!/usr/bin/env bash
# submit_variant_effects.sh
# -----------------------------
# Run variant effects prediction on all models locally

set -euo pipefail

cd "$HOME/BEND"

docker build -t bend_env .

mkdir -p logs results_variant_effects

tasks=("expression" "disease")

for task in "${tasks[@]}"; do
  variant_file="data/variant_effects/variant_effects_${task}.bed"
  
#   # DNABERT2-curriculum-512-phylum
#   echo "▶️  dnabert2-curriculum-512-phylum  /  $task"
#   docker run --rm --gpus all --shm-size=100g \
#     -v "$(pwd)":/app -w /app \
#     bend_env \
#     bash -lc "\
#       /root/.local/bin/poetry install --no-root && \
#       /root/.local/bin/poetry run python scripts/predict_variant_effects.py \
#         ${variant_file} results_variant_effects/variant_effects_${task}_dnabert2-curriculum-512-phylum.csv \
#         dnabert2 fine_tuned_models/dnabert_2_curriculum_patience_5_batch_size_60_max_length_512/curriculum_stage_phylum/merged_model \
#         data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256" \
#     > "logs/variant_effects_dnabert2-curriculum-512-phylum_${task}_$(hostname).out" 2>&1
  
#   # DNABERT2-curriculum-512-class
#   echo "▶️  dnabert2-curriculum-512-class  /  $task"
#   docker run --rm --gpus all --shm-size=100g \
#     -v "$(pwd)":/app -w /app \
#     bend_env \
#     bash -lc "\
#       /root/.local/bin/poetry install --no-root && \
#       /root/.local/bin/poetry run python scripts/predict_variant_effects.py \
#         ${variant_file} results_variant_effects/variant_effects_${task}_dnabert2-curriculum-512-class.csv \
#         dnabert2 fine_tuned_models/dnabert_2_curriculum_patience_5_batch_size_60_max_length_512/curriculum_stage_class/merged_model \
#         data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256" \
#     > "logs/variant_effects_dnabert2-curriculum-512-class_${task}_$(hostname).out" 2>&1
  
#   # DNABERT2-curriculum-512-order
#   echo "▶️  dnabert2-curriculum-512-order  /  $task"
#   docker run --rm --gpus all --shm-size=100g \
#     -v "$(pwd)":/app -w /app \
#     bend_env \
#     bash -lc "\
#       /root/.local/bin/poetry install --no-root && \
#       /root/.local/bin/poetry run python scripts/predict_variant_effects.py \
#         ${variant_file} results_variant_effects/variant_effects_${task}_dnabert2-curriculum-512-order.csv \
#         dnabert2 fine_tuned_models/dnabert_2_curriculum_patience_5_batch_size_60_max_length_512/curriculum_stage_order/merged_model \
#         data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256" \
#     > "logs/variant_effects_dnabert2-curriculum-512-order_${task}_$(hostname).out" 2>&1
  
#   # DNABERT2-curriculum-512-family
#   echo "▶️  dnabert2-curriculum-512-family  /  $task"
#   docker run --rm --gpus all --shm-size=100g \
#     -v "$(pwd)":/app -w /app \
#     bend_env \
#     bash -lc "\
#       /root/.local/bin/poetry install --no-root && \
#       /root/.local/bin/poetry run python scripts/predict_variant_effects.py \
#         ${variant_file} results_variant_effects/variant_effects_${task}_dnabert2-curriculum-512-family.csv \
#         dnabert2 fine_tuned_models/dnabert_2_curriculum_patience_5_batch_size_60_max_length_512/curriculum_stage_family/merged_model \
#         data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256" \
#     > "logs/variant_effects_dnabert2-curriculum-512-family_${task}_$(hostname).out" 2>&1
  
#   # DNABERT2-curriculum-512-genus
#   echo "▶️  dnabert2-curriculum-512-genus  /  $task"
#   docker run --rm --gpus all --shm-size=100g \
#     -v "$(pwd)":/app -w /app \
#     bend_env \
#     bash -lc "\
#       /root/.local/bin/poetry install --no-root && \
#       /root/.local/bin/poetry run python scripts/predict_variant_effects.py \
#         ${variant_file} results_variant_effects/variant_effects_${task}_dnabert2-curriculum-512-genus.csv \
#         dnabert2 fine_tuned_models/dnabert_2_curriculum_patience_5_batch_size_60_max_length_512/curriculum_stage_genus/merged_model \
#         data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256" \
#     > "logs/variant_effects_dnabert2-curriculum-512-genus_${task}_$(hostname).out" 2>&1
  
  # DNABERT2-no-overlap-curriculum
  echo "▶️  dnabert2-no-overlap-curriculum  /  $task"
  docker run --rm --gpus all --shm-size=100g \
    -v "$(pwd)":/app -w /app \
    bend_env \
    bash -lc "\
      /root/.local/bin/poetry install --no-root && \
      /root/.local/bin/poetry run python scripts/predict_variant_effects.py \
        ${variant_file} results_variant_effects/variant_effects_${task}_dnabert2-no-overlap-curriculum.csv \
        dnabert2 fine_tuned_models/dnabert2-no-overlap-curriculum/merged_model \
        data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256" \
    > "logs/variant_effects_dnabert2-no-overlap-curriculum_${task}_$(hostname).out" 2>&1

#   # HyenaDNA-no-overlap
#   echo "▶️  hyenadna-no-overlap  /  $task"
#   docker run --rm --gpus all --shm-size=100g \
#     -v "$(pwd)":/app -w /app \
#     bend_env \
#     bash -lc "\
#       /root/.local/bin/poetry install --no-root && \
#       /root/.local/bin/poetry run python scripts/predict_variant_effects.py \
#         ${variant_file} results_variant_effects/variant_effects_${task}_hyenadna-no-overlap.csv \
#         hyenadna-no-overlap fine_tuned_models/hyenadna_no_overlap data/genomes/GRCh38.primary_assembly.genome.fa \
#         --extra_context 512 --embedding_idx 511" \
#     > "logs/variant_effects_hyenadna-no-overlap_${task}_$(hostname).out" 2>&1
done

echo "✅  All variant effects predictions for all models completed."