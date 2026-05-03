"""Generate alternating-prefix prompts JSONL for Scenario A benchmark.

Two repeated-character prefixes (PREFIX_TOKENS tokens each) alternated across
NUM_PROMPTS requests. **No random suffix** — each prompt is just the prefix.
This minimises expert-routing variety so the workload exercises the unified
pool's prefix-vs-expert tradeoff cleanly: per the variety probe in
`expert_variety_test.md`, a pure single-character prefix touches only ~11-20/64
experts per layer, leaving the rest cold and cleanly evictable.

Each character has very different BPE behaviour with the OLMoE tokenizer (a
single "a" * N tokenizes to far fewer tokens than "b" * N), so we oversize the
character repeat then truncate at the token level to land both prefixes at
exactly PREFIX_TOKENS.

PREFIX_TOKENS is **deliberately one token past 2 × block_size (= 3072)**: with
exactly 3072 tokens the prompt's *last* block is full block 2, and vLLM does
not promote a prompt's terminal block to the prefix cache (only block 1 ends
up cacheable). Empirically that capped the prefix-cache hit rate at ~50% and
forced 1,536 tokens of recompute per request. PREFIX_TOKENS = 3073 makes
block 3 the terminal block (1 token, partial) so block 2 is no longer the
last block and gets hashed → both prefix blocks cache → ~100% hit rate.

Run once before the benchmark:
    python make_alternating_prompts.py
"""

import json

from transformers import AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
PREFIX_TOKENS = 3073
NUM_PROMPTS = 20
OUTPUT_LEN = 20
OUT = "alternating_prompts.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def make_char_prefix(ch, n_tokens):
    """Build a string of repeated `ch` that tokenizes to exactly n_tokens.

    Oversize first (50k chars is enough for any single-char repeat to exceed
    n_tokens with the OLMoE tokenizer), then truncate the token list.
    """
    raw = ch * 50000
    ids = tok.encode(raw, add_special_tokens=False)
    assert len(ids) >= n_tokens, (
        f"50000×'{ch}' tokenized to only {len(ids)} tokens; bump the multiplier"
    )
    return tok.decode(ids[:n_tokens], skip_special_tokens=True)


prefixes = [
    make_char_prefix("a", PREFIX_TOKENS),
    make_char_prefix("b", PREFIX_TOKENS),
]

with open(OUT, "w") as f:
    for i in range(NUM_PROMPTS):
        prompt = prefixes[i % 2]
        json.dump({"prompt": prompt, "output_tokens": OUTPUT_LEN}, f)
        f.write("\n")

print(f"Wrote {NUM_PROMPTS} alternating prompts to {OUT}")
