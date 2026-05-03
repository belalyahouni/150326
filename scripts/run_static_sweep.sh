#!/bin/bash
set -u
SEED=1
ROOT=/home/belal/150326
LOGS=$ROOT/logs
RESULTS=$ROOT/results
VLLM=$ROOT/venv-phase-2/bin/vllm
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
PROMPTS=$ROOT/prompts/alternating_prompts.jsonl

mkdir -p "$LOGS" "$RESULTS"

start_static() {
  local gpu=$1 port=$2 cache=$3
  CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM" serve "$MODEL" \
    --port $port \
    --expert-offload --expert-cache-size $cache \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 --gpu-memory-utilization 0.3105 \
    > "$LOGS/sweep_static_cache${cache}_seed${SEED}_g${gpu}.log" 2>&1 &
  echo $!
}

wait_for_server() {
  local port=$1 pid=$2
  for i in $(seq 1 240); do
    if curl -sf -m 2 http://127.0.0.1:$port/v1/models >/dev/null 2>&1; then
      echo "[ready] port=$port after ${i}*2s"; return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[dead] server pid=$pid port=$port"; return 1
    fi
    sleep 2
  done
  echo "[timeout] port=$port"; return 1
}

run_workload_A() {
  local port=$1 cache=$2
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $port \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --disable-shuffle --skip-chat-template \
    --custom-output-len 20 --num-prompts 5 \
    --max-concurrency 1 --num-warmups 1 --seed $SEED \
    --result-filename "$RESULTS/sweep_static_cache${cache}_1A_seed${SEED}.json" \
    --save-result --trust-remote-code
}

run_workload_B() {
  local port=$1 cache=$2
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $port \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name random \
    --random-input-len 256 --random-output-len 80 \
    --random-range-ratio 0 --random-prefix-len 0 \
    --num-prompts 8 \
    --max-concurrency 1 --num-warmups 1 --seed $SEED \
    --result-filename "$RESULTS/sweep_static_cache${cache}_1B_seed${SEED}.json" \
    --save-result --trust-remote-code
}

shutdown() {
  local pid=$1
  if [ -z "${pid:-}" ]; then return; fi
  kill -TERM -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    if ! kill -0 $pid 2>/dev/null; then return; fi
    sleep 1
  done
  kill -KILL -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
}

run_cell() {
  local cache=$1 gpu=$2 port=$3
  local PID
  PID=$(start_static $gpu $port $cache)
  wait_for_server $port $PID || { shutdown $PID; return 1; }
  run_workload_A $port $cache > "$LOGS/bench_sweep_static_cache${cache}_1A_g${gpu}.log" 2>&1
  echo "[cache=$cache g$gpu] 1A exit: $?"
  run_workload_B $port $cache > "$LOGS/bench_sweep_static_cache${cache}_1B_g${gpu}.log" 2>&1
  echo "[cache=$cache g$gpu] 1B exit: $?"
  shutdown $PID
}

# Pair cache sizes across the 2 GPUs.
PAIRS="8,16 24,32 40,48 56,64"
for pair in $PAIRS; do
  C0=${pair%,*}
  C1=${pair#*,}
  echo "=== Sweep round: cache=$C0 (GPU0) + cache=$C1 (GPU1) ==="
  run_cell $C0 0 8000 &
  run_cell $C1 1 8001 &
  wait
  echo "=== Sweep round done ==="
done

echo "=== Static sweep complete ==="
