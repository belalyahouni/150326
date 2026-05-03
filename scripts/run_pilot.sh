#!/bin/bash
set -u
SEED=1
ROOT=/home/belal/150326
LOGS=$ROOT/logs
RESULTS=$ROOT/results
VLLM=$ROOT/venv-phase-2/bin/vllm
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
PROMPTS=$ROOT/prompts/alternating_prompts.jsonl
UNIFIED_BLOCKS="${UNIFIED_BLOCKS:-68}"
TRACE_LEVEL="${TRACE_LEVEL:-1}"

mkdir -p "$LOGS" "$RESULTS"

start_static() {
  local gpu=$1 port=$2 cache=$3 cell=$4
  CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM" serve "$MODEL" \
    --port $port \
    --expert-offload --expert-cache-size $cache \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 --gpu-memory-utilization 0.3105 \
    > "$LOGS/${cell}_seed${SEED}_g${gpu}.log" 2>&1 &
  echo $!
}

start_unified() {
  local gpu=$1 port=$2 cache=$3 cell=$4 trace=$5
  local logsuffix=""
  local env_prefix=""
  if [ "$trace" = "1" ]; then
    env_prefix="VLLM_UNIFIED_POOL_TRACE=$TRACE_LEVEL"
    logsuffix="_trace"
  fi
  env $env_prefix CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM" serve "$MODEL" \
    --port $port \
    --expert-offload --expert-unified-pool --expert-cache-size $cache \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 --num-gpu-blocks-override $UNIFIED_BLOCKS \
    > "$LOGS/${cell}_seed${SEED}_g${gpu}${logsuffix}.log" 2>&1 &
  echo $!
}

wait_for_server() {
  local port=$1
  local pid=$2
  for i in $(seq 1 240); do
    if curl -sf -m 2 http://127.0.0.1:$port/v1/models >/dev/null 2>&1; then
      echo "[ready] port=$port after ${i}*2s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[dead] server pid=$pid (port=$port) exited during boot"
      return 1
    fi
    sleep 2
  done
  echo "[timeout] port=$port did not become ready"
  return 1
}

run_bench_1A() {
  local port=$1 outname=$2
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $port \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --disable-shuffle --skip-chat-template \
    --custom-output-len 20 --num-prompts 20 \
    --max-concurrency 1 --num-warmups 1 --seed $SEED \
    --result-filename "$RESULTS/$outname" --save-result --trust-remote-code
}

shutdown() {
  local pid=$1
  if [ -z "${pid:-}" ]; then return; fi
  # SIGTERM the whole process group started by setsid
  kill -TERM -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    if ! kill -0 $pid 2>/dev/null; then return; fi
    sleep 1
  done
  kill -KILL -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
}

ROUND="${1:-all}"

if [ "$ROUND" = "round1" ] || [ "$ROUND" = "all" ]; then
  echo "=== Round 1: 1A-static-bad (GPU0) + 1A-static-good (GPU1) ==="
  PID0=$(start_static 0 8000 64 test1A_static_bad)
  PID1=$(start_static 1 8001 20 test1A_static_good)
  wait_for_server 8000 $PID0 || { shutdown $PID0; shutdown $PID1; exit 1; }
  wait_for_server 8001 $PID1 || { shutdown $PID0; shutdown $PID1; exit 1; }
  run_bench_1A 8000 test1A_static_bad_seed${SEED}.json > "$LOGS/bench_test1A_static_bad_seed${SEED}.log" 2>&1 &
  B0=$!
  run_bench_1A 8001 test1A_static_good_seed${SEED}.json > "$LOGS/bench_test1A_static_good_seed${SEED}.log" 2>&1 &
  B1=$!
  wait $B0; B0_EX=$?
  wait $B1; B1_EX=$?
  echo "[round1] bench exits: g0=$B0_EX g1=$B1_EX"
  shutdown $PID0
  shutdown $PID1
  echo "=== Round 1 done ==="
fi

if [ "$ROUND" = "round2" ] || [ "$ROUND" = "all" ]; then
  echo "=== Round 2: 1A-unified-from-bad latency (GPU0) + trace (GPU1) ==="
  PID0=$(start_unified 0 8000 64 test1A_unified_from_bad 0)
  PID1=$(start_unified 1 8001 64 test1A_unified_from_bad 1)
  wait_for_server 8000 $PID0 || { shutdown $PID0; shutdown $PID1; exit 1; }
  wait_for_server 8001 $PID1 || { shutdown $PID0; shutdown $PID1; exit 1; }
  run_bench_1A 8000 test1A_unified_from_bad_seed${SEED}.json > "$LOGS/bench_test1A_unified_latency_seed${SEED}.log" 2>&1 &
  B0=$!
  run_bench_1A 8001 _discard_trace_seed${SEED}.json > "$LOGS/bench_test1A_unified_trace_seed${SEED}.log" 2>&1 &
  B1=$!
  wait $B0; B0_EX=$?
  wait $B1; B1_EX=$?
  echo "[round2] bench exits: latency=$B0_EX trace=$B1_EX"
  shutdown $PID0
  shutdown $PID1
  rm -f "$RESULTS/_discard_trace_seed${SEED}.json"
  echo "=== Round 2 done ==="
fi

if [ "$ROUND" = "round3" ] || [ "$ROUND" = "all" ]; then
  echo "=== Round 3: Test 0 output consistency ==="
  PID0=$(start_static 0 8000 64 test0_static)
  PID1=$(start_unified 1 8001 40 test0_unified 0)
  wait_for_server 8000 $PID0 || { shutdown $PID0; shutdown $PID1; exit 1; }
  wait_for_server 8001 $PID1 || { shutdown $PID0; shutdown $PID1; exit 1; }

  declare -a PROMPTS_ARR=(
    "The capital of France is"
    $'def fibonacci(n):\n    '
  )
  PROMPT3=$(/home/belal/150326/venv-phase-2/bin/python -c '
import json
with open("/home/belal/150326/alternating_prompts.jsonl") as f:
    line=f.readline()
print(json.loads(line)["prompt"][:400])
')
  PROMPTS_ARR+=("$PROMPT3")

  for backend in static unified; do
    if [ "$backend" = "static" ]; then port=8000; else port=8001; fi
    out="$RESULTS/test0_${backend}_outputs.txt"
    : > "$out"
    for i in 0 1 2; do
      p="${PROMPTS_ARR[$i]}"
      echo "=== prompt $i (backend=$backend) ===" >> "$out"
      printf "PROMPT: %q\n" "$p" >> "$out"
      curl -s http://127.0.0.1:$port/v1/completions \
        -H "Content-Type: application/json" \
        -d "$(/home/belal/150326/venv-phase-2/bin/python -c "
import json,sys
print(json.dumps({'model':'$MODEL','prompt':sys.argv[1],'max_tokens':32,'temperature':0,'seed':1}))
" "$p")" \
        | tee -a "$out" > /dev/null
      echo >> "$out"
    done
  done
  shutdown $PID0
  shutdown $PID1
  echo "=== Round 3 done ==="
fi

echo "=== Pilot complete ==="
