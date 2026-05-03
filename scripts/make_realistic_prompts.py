"""Generate realistic system-prompt + user-query JSONL.

Each prompt = [1,536-token system prompt — fixed] + [variable user query].
The system prompt fills exactly 1 block at block_size=1536, so it caches
cleanly. User queries cycle through 8 distinct natural-language questions
across NUM_PROMPTS = 10 requests (so some queries repeat — tests multi-block
prefix-cache behaviour beyond just the system block).

Run once before the benchmark:
    python make_realistic_prompts.py
"""

import json

from transformers import AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
SYSTEM_TOKENS = 1536  # exactly one block at block_size=1536
NUM_PROMPTS = 10
OUT = "/home/belal/150326/prompts/realistic_prompts.jsonl"
OUTPUT_LEN = 50

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# A plausible instruction template. We oversize then truncate to exactly
# SYSTEM_TOKENS tokens so the system block is full and hashable.
SYSTEM_TEMPLATE = (
    "You are a helpful, harmless, and concise assistant. "
    "Answer the user's question using clear, accurate information. "
    "Always cite uncertainty when relevant. Avoid filler. "
    "Use plain language. Provide examples where they help. "
    "If the user asks for code, return runnable code. "
    "If the user asks for an explanation, structure it logically. "
    "Stay on topic. Be polite and professional. "
)
# Repeat the template enough times to reliably exceed SYSTEM_TOKENS, then
# truncate at the token level.
oversized = (SYSTEM_TEMPLATE * 200)
ids = tok.encode(oversized, add_special_tokens=False)
assert len(ids) >= SYSTEM_TOKENS, f"template too short, got {len(ids)} tokens"
SYSTEM_PROMPT = tok.decode(ids[:SYSTEM_TOKENS], skip_special_tokens=True)
# Verify exact length
actual_len = len(tok.encode(SYSTEM_PROMPT, add_special_tokens=False))
assert actual_len == SYSTEM_TOKENS, (
    f"system prompt re-tokenises to {actual_len}, expected {SYSTEM_TOKENS}"
)

# 8 distinct user queries of varied lengths (50-200 tokens each).
USER_QUERIES = [
    "What is the capital of France, and can you tell me a few interesting "
    "historical facts about the city's founding and early development?",
    "Explain the concept of recursion in programming with a simple Python "
    "example. Include the base case and the recursive case clearly.",
    "Could you describe the difference between supervised and unsupervised "
    "machine learning, with one practical example of each used in industry?",
    "What were the main causes of the French Revolution? Please provide "
    "a structured overview covering political, economic, and social factors.",
    "Write a short Python function that takes a list of integers and "
    "returns the second largest unique value. Handle edge cases properly.",
    "How does a transformer neural network process input sequences? Focus "
    "on attention and feed-forward layers in your explanation.",
    "Summarise the plot of Hamlet in three paragraphs, focusing on the "
    "main conflict and its resolution. Avoid spoilers about minor characters.",
    "Explain why the sky appears blue during the day but reddish at sunset. "
    "Include the role of Rayleigh scattering and atmospheric path length.",
]

with open(OUT, "w") as f:
    for i in range(NUM_PROMPTS):
        query = USER_QUERIES[i % len(USER_QUERIES)]
        prompt = SYSTEM_PROMPT + "\n\n" + query
        json.dump({"prompt": prompt, "output_tokens": OUTPUT_LEN}, f)
        f.write("\n")

# Quick sanity print
print(f"Wrote {NUM_PROMPTS} prompts to {OUT}")
print(f"  System prompt: {SYSTEM_TOKENS} tokens (fixed)")
print(f"  User queries: {len(USER_QUERIES)} unique, cycled across {NUM_PROMPTS} prompts")
sample_lens = []
for line in open(OUT):
    p = json.loads(line)["prompt"]
    sample_lens.append(len(tok.encode(p, add_special_tokens=False)))
print(f"  Total prompt lengths: min={min(sample_lens)}, max={max(sample_lens)}, mean={sum(sample_lens)/len(sample_lens):.0f}")
