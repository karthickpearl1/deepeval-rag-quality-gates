#!/usr/bin/env bash
# Replays the README demo: the good bot passes the gate, the regressed bot fails it.
# Recorded with: asciinema rec -c "bash scripts/demo.sh" docs/demo.cast
set -u
export DEEPEVAL_TELEMETRY_OPT_OUT=1

type_cmd() {
  printf '\033[1;32m$\033[0m '
  local text="$1"
  for ((i = 0; i < ${#text}; i++)); do
    printf '%s' "${text:i:1}"
    sleep 0.02
  done
  printf '\n'
  sleep 0.4
}

run() {
  type_cmd "$1"
  eval "$1"
  sleep 1.5
}

run "pytest tests/test_quality_gate.py -q | tail -n 1"
run "BOT_VARIANT=regressed pytest tests/test_quality_gate.py -q --tb=no > /dev/null; echo \"exit code: \$?\""
run "python -m evals.show_results regressed"
sleep 1
