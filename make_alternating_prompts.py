"""Generate alternating-prefix prompts JSONL for Scenario A benchmark.

Two random prefixes (~PREFIX_LEN tokens each) alternated across NUM_PROMPTS
requests. Each request gets a fresh suffix so we measure prefix-only cache
behaviour, not full-prompt deduplication.

Run once before the benchmark:
    python make_alternating_prompts.py
"""

import json
import random

from transformers import AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
# Random-token prompts re-encode to ~5% more tokens after decode/re-encode,
# so we target shorter to land safely below max_model_len=4096 (incl. output).
PREFIX_LEN = 3700
SUFFIX_LEN = 60
NUM_PROMPTS = 10
OUTPUT_LEN = 20
SEED = 1
OUT = "alternating_prompts.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
random.seed(SEED)
vocab = tok.vocab_size


def gen_text(n_tokens):
    ids = [random.randint(0, vocab - 1) for _ in range(n_tokens)]
    return tok.decode(ids, skip_special_tokens=True)


prefixes = [gen_text(PREFIX_LEN), gen_text(PREFIX_LEN)]

with open(OUT, "w") as f:
    for i in range(NUM_PROMPTS):
        prompt = prefixes[i % 2] + " " + gen_text(SUFFIX_LEN)
        json.dump({"prompt": prompt, "output_tokens": OUTPUT_LEN}, f)
        f.write("\n")

print(f"Wrote {NUM_PROMPTS} alternating prompts to {OUT}")
