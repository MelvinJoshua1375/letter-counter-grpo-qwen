"""
CPU smoke training run.

Stands in for the full GPU run when developing locally without CUDA. It
proves the whole GRPO + LoRA + reward-funcs loop is wired up correctly,
and produces a real (small) trend line for the training plot.

Differences from the GPU/T4 submission run (notebook):
  - base model: Qwen2.5-0.5B-Instruct (instead of 3B; 3B will not fit in 5.7 GB CPU RAM)
  - rollouts:    transformers generation (instead of vLLM)
  - precision:   fp32 (instead of 4-bit quant)
  - steps:       configurable via MAX_STEPS env var (default 3)
  - batch:       per_device_train_batch_size=1, num_generations=2 (TRL minimum)

The reward functions, system prompt, dataset schema, and LoRA target
modules are IDENTICAL to the notebook.
"""

import os
import re
import json
import time
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(int(os.environ.get("CPU_THREADS", "4")))

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "3"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs_smoke")

SYSTEM_PROMPT = """You are a careful letter-counting assistant. When asked to count how many times a letter appears in a word, think step by step:

1. Spell the word out letter by letter, numbered starting at 1.
2. At each step, check whether the letter is the target letter.
3. Keep a running total of matches.
4. State the final count.

Respond in EXACTLY this XML format and nothing else:
<reasoning>
Counting the number of [letter]'s in the word [word]
1. [first letter] - [count so far] so far
2. [second letter] - [count so far] so far
...
</reasoning>
<answer>
[final integer count]
</answer>

Worked example.

User: How many of the letter "o" are there in the word "room"
Assistant:
<reasoning>
Counting the number of o's in the word room
1. r - 0 so far
2. o - 1 so far
3. o - 2 so far
4. m - 2 so far
</reasoning>
<answer>
2
</answer>
"""


def extract_letter_numbering(response):
    matches = re.findall(r"\n(\d+). [a-z]", response)
    return [int(m) for m in matches] if matches else []


def extract_spelling(response):
    matches = re.findall(r"\n\d+. ([a-z])", response, flags=re.IGNORECASE)
    return "".join(matches) if matches else ""


def get_resp_letters_and_counts(response):
    matches = re.findall(r"\n(\d+)\. ([a-z])\D*(\d+)", response, flags=re.IGNORECASE)
    return [(L, c) for _, L, c in matches] if matches else []


def extract_xml_answer(text):
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def numbering_reward_func(completions, words, **kwargs):
    responses = [c[0]["content"] for c in completions]
    out = []
    for r, w in zip(responses, words):
        reward = 0.0
        for ix, n in enumerate(extract_letter_numbering(r)):
            line = ix + 1
            reward += 0.5 if n == line else -0.5
            if line > len(w):
                reward -= 1.0
        out.append(reward / max(len(w), 1))
    return out


def spelling_reward_func(completions, words, **kwargs):
    responses = [c[0]["content"] for c in completions]
    out = []
    for w, r in zip(words, responses):
        reward = 0.0
        spelled = extract_spelling(r).lower()
        target = w.lower()
        if spelled == target:
            reward += 2.0
        reward -= 0.5 * abs(len(spelled) - len(target))
        cs, ct = Counter(spelled), Counter(target)
        for ch, n in cs.items():
            extra = n - ct.get(ch, 0)
            if extra > 0:
                reward -= 1.0 * extra
        for ch, n in ct.items():
            missing = n - cs.get(ch, 0)
            if missing > 0:
                reward -= 0.5 * missing
        out.append(reward)
    return out


def counting_reward_func(completions, letters, **kwargs):
    responses = [c[0]["content"] for c in completions]
    out = []
    for letter, r in zip(letters, responses):
        reward = 0.0
        lc = get_resp_letters_and_counts(r)
        if not lc:
            out.append(-1.0)
            continue
        actual = 0
        for rl, rc in lc:
            if letter == rl:
                actual += 1
            reward += 1.0 if int(rc) == actual else -1.0
        out.append(reward / len(lc))
    return out


def format_reward_func(completions, **kwargs):
    pat = r"\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    out = []
    for c in completions:
        r = c[0]["content"]
        reward = 0.0
        if re.match(pat, r, flags=re.MULTILINE | re.DOTALL):
            reward += 0.5
        else:
            reward -= 0.5    # penalty for a missing / broken XML envelope
        if extract_xml_answer(r).isdigit():
            reward += 0.5
        else:
            reward -= 0.5    # penalty for a non-numeric (or missing) answer
        out.append(reward)
    return out


def correct_answer_reward_func(prompts, completions, counts, **kwargs):
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    print(f"[reward log] q={prompts[0][-1]['content'][:60]!r} gold={counts[0]} got={extracted[0]!r}")
    return [2.0 if str(r) == str(a) else -1.0 for r, a in zip(extracted, counts)]


REWARD_FUNCS = [
    numbering_reward_func,
    spelling_reward_func,
    counting_reward_func,
    format_reward_func,
    correct_answer_reward_func,
]


RAW = [
    ("strawberry", "r", 3),
    ("banana",     "a", 3),
    ("mississippi", "s", 4),
    ("goal",       "o", 1),
    ("effectiveness", "e", 3),
    ("balloon",    "l", 2),
    ("committee",  "m", 2),
    ("apple",      "p", 2),
]


def to_row(word, letter, count):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f'How many of the letter "{letter}" are there in the word "{word}"'},
        ],
        "words":   word,
        "letters": letter,
        "counts":  count,
    }


def main():
    t0 = time.time()
    ds = Dataset.from_list([to_row(*x) for x in RAW])
    print(f"[data] {len(ds)} rows, columns={ds.column_names}", flush=True)

    print(f"[model] loading {MODEL_ID} on CPU (fp32)…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    print(f"[model] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        beta=1e-4,
        per_device_train_batch_size=2,
        num_generations=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        max_completion_length=96,
        max_steps=MAX_STEPS,
        save_steps=10_000,
        report_to="none",
        use_vllm=False,
        bf16=False,
        fp16=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=ds,
        peft_config=lora_config,
    )

    print(f"[train] starting {MAX_STEPS}-step GRPO smoke run…", flush=True)
    trainer_res = trainer.train()
    print(f"[train] done in {time.time()-t0:.1f}s", flush=True)
    print(f"[train] metrics: {trainer_res.metrics}", flush=True)

    # Dump log_history so the notebook can pick up the per-step rewards.
    log = trainer.state.log_history
    out_path = os.path.join(OUTPUT_DIR, "log_history.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[train] wrote {out_path} ({len(log)} entries)", flush=True)


if __name__ == "__main__":
    main()
