#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/BEND"

if [[ -z "${OAR_NODEFILE:-}" ]]; then
  echo "❌  OAR_NODEFILE not set. You must run this inside an OAR allocation:"
  echo "    oarsub -I -p \"host in ('alakazam-01.grid5000.fr', ...)\" -l nodes=4,walltime=72:00:00"
  exit 1
fi

mapfile -t HOSTS < <(sort -u "$OAR_NODEFILE")
WORLD=${#HOSTS[@]}

echo "🌐  World size = $WORLD node(s):"
printf '  - %s\n' "${HOSTS[@]}"
echo
echo "📊  Launching experiment registry from scripts/rerun_missing_on_node.sh"
echo

echo "🐳  Building Docker image..."
docker build -t bend_env .
echo

mkdir -p ~/.ssh
chmod 700 ~/.ssh
> ~/.ssh/known_hosts
for host in "${HOSTS[@]}"; do
  ssh-keyscan -H "$host" 2>/dev/null >> ~/.ssh/known_hosts
done

mkdir -p logs results_variant_effects

for rank in "${!HOSTS[@]}"; do
  host=${HOSTS[$rank]}
  echo "🚀  Launching rank=$rank on $host"

  if [[ "$host" == "$(hostname)" || "$host" == "$(hostname -f)" ]]; then
    bash scripts/rerun_missing_on_node.sh "$rank" "$WORLD" \
      > "logs/rerun_missing_rank${rank}_${host}.out" 2>&1 &
  else
    oarsh "$host" "cd $HOME/BEND && bash scripts/rerun_missing_on_node.sh $rank $WORLD" \
      > "logs/rerun_missing_rank${rank}_${host}.out" 2>&1 &
  fi
done

echo
echo "⏳  All ranks launched. Waiting for completion..."
echo "    Monitor with: tail -f logs/rerun_missing_rank*.out"
wait

echo
echo "✅  All ranks finished at $(date)"
echo "📁  Logs: logs/rerun_missing_rank*.out"
echo "📁  Results: results_variant_effects/"
