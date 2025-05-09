#!/usr/bin/env bash
#
# batch_run.sh
#

# activate python env
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

# arg lists:
LINKS=(5 10)
FLOWS=(2 5)
RATE_PATTERNS=(low medium random)
METHODS=(ExactEnum GreedyApprox RandomSample)

# iterate over all combination
for L in "${LINKS[@]}"; do
  for F in "${FLOWS[@]}"; do
    for RP in "${RATE_PATTERNS[@]}"; do
      for M in "${METHODS[@]}"; do
        echo "Running: links=$L flows=$F rate_pattern=$RP method=$M"
        python3 run_experiment.py \
          --links "$L" \
          --flows "$F" \
          --rate_pattern "$RP" \
          --method "$M"
      done
    done
  done
done

echo "All experiments completed."
