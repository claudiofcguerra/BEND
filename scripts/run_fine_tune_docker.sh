#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${HOME}/BEND"

if [ ! -d "$REPO_DIR" ]; then
  echo "ERROR: Directory $REPO_DIR not found."
  echo "Make sure you have the repository checked out at $REPO_DIR"
  exit 1
fi

echo "Building docker image 'bend_env' from $REPO_DIR"
cd "$REPO_DIR"
docker build -t bend_env .
 
echo "Running container and starting training..."
docker run -e HYDRA_FULL_ERROR=1 --gpus all --shm-size=100g --rm \
  -v "$(pwd)":/app -w /app bend_env bash -lc "\
    cd scripts && \
    /root/.local/bin/poetry install --no-root && \
    /root/.local/bin/poetry run python fine_tune_curriculum_mlm_minimal.py \
      --model hyenadna \
      --batch_size 128 \
      --patience 5 \
      --epochs_per_stage 100 \
      --max_length 1000 \
      --final_model_dir hyenadna_curriculum_clm \
      2>&1 | tee hyenadna_training.log"