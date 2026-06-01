#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/tese-monorepo/BEND"

docker build -t monorepo ..
docker run --rm --gpus all --shm-size=100g -it \
  -v "$(pwd)":/app -w /app \
  monorepo \
  bash -lc "\
    poetry install --no-root && \
    poetry run python scripts/fine_tune_curriculum_mlm.py \
      --dataset_dir ../metagenomic-language-models/curriculum_datasets/samples_per_rank \
      --output_base_dir dnabert_curriculum_samples_per_rank"