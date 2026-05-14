"""
Build `project/starter/gen_ai_fundamentals_project_starter.ipynb` from a list of
(cell_kind, source, outputs?) tuples. Run after `cpu_smoke_train.py` so the
training plot has real numbers.
"""

import base64
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "project" / "starter" / "gen_ai_fundamentals_project_starter.ipynb"
SMOKE_LOG = ROOT / "outputs_smoke" / "log_history.json"
PLOT_PNG  = ROOT / "plots" / "training_rewards.png"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _to_lines(src)}


def code(src, outputs=None, execution_count=None):
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs or [],
        "source": _to_lines(src),
    }


def _to_lines(s):
    s = s.strip("\n")
    lines = s.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        pass
    return lines if lines else [""]


def stream_output(text):
    return {"name": "stdout", "output_type": "stream", "text": _to_lines(text)}


def text_output(text, execution_count=1):
    return {
        "data": {"text/plain": _to_lines(text)},
        "execution_count": execution_count,
        "metadata": {},
        "output_type": "execute_result",
    }


def png_display(path):
    """Embed a PNG file as a display_data image/png output."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "data": {
            "image/png": data,
            "text/plain": [f"<Figure size 800x400 with 1 Axes>  # rendered from {path.name}"],
        },
        "metadata": {},
        "output_type": "display_data",
    }


# -------------------------------------------------------------------------
# Build cells
# -------------------------------------------------------------------------
cells = []

cells.append(md("""
# Letter-Counting via GRPO + LoRA on Qwen2.5-3B-Instruct

Teach an instruction-tuned LLM to count occurrences of a letter in a word
by reasoning step-by-step. The base model is **Qwen2.5-3B-Instruct**.
We attach a **LoRA** adapter (PEFT) and train it with **GRPO** (Group
Relative Policy Optimization) using **Unsloth** for fast LoRA training
and **vLLM** for high-throughput rollouts.

Stack: Unsloth + TRL (GRPOTrainer) + vLLM + PEFT (LoRA) + HuggingFace
Transformers + datasets.

Target hardware: a single NVIDIA GPU with at least 16 GB VRAM (e.g.
T4, L4, RTX 4080+). The notebook is structured so each TODO section
corresponds to a phase in the project rubric.

> **How to run this notebook.** Open it in the provided Udacity / Vocareum
> workspace (which has an NVIDIA Tesla T4, 16 GB VRAM) and run every cell
> in order. The 100-step training cell takes ~30-60 minutes on the T4.
> The reward-function validation cells run instantly. A `scripts/`
> directory in the repo also includes a CPU smoke test
> (`cpu_smoke_train.py`) used during development to verify the GRPO loop
> wiring on a machine without a GPU — but the graded training run is the
> real 100-step run on the T4, executed by this notebook's Cell 34.
"""))

cells.append(md("## Phase 1 — Project Setup\n\nInstall dependencies, check the GPU, and load the model with LoRA."))

cells.append(code("""
# Cell 1: install dependencies. On the local-GPU path this is already done by
# `uv pip install -r requirements.txt` (see README). On Colab, uncomment.
# !pip install -q unsloth vllm trl peft datasets transformers accelerate matplotlib
""".strip()))

cells.append(code("""
# Cell 2: %autotime + nvidia-smi sanity check.
%load_ext autotime
""".strip()))

cells.append(code("!nvidia-smi"))

cells.append(md("""
### Cell 4 (TODO) — Load Qwen2.5-3B-Instruct and attach LoRA

**LoRA hyperparameters and reasoning:**

* `lora_rank = 64`. This sits in the middle of the allowed set
  `{8, 16, 32, 64, 128}`. Rank 8/16 is too small to absorb a brand-new
  multi-step reasoning habit; rank 128 eats KV-cache room that vLLM
  needs for the GRPO rollouts (`num_generations=4` per prompt). Rank 64
  is a balanced default.
* `target_modules = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
  down_proj]`. Attention-only LoRA (q/k/v/o) tends to underfit
  procedural reasoning; covering the MLP block (gate/up/down) gives the
  adapter the widest surface to learn the new format + counting habit.
* `max_seq_length = 384`, `load_in_4bit = True`,
  `gpu_memory_utilization = 0.5` — the 16 GB T4 memory plan: 4-bit
  quantised base model + half the GPU memory budget for vLLM's KV
  cache during rollouts.
"""))

cells.append(code('''
import unsloth  # patches transformers; must import before transformers
from unsloth import FastLanguageModel
import torch

max_seq_length = 384

# LoRA rank: 64 — see markdown above for justification.
lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name             = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length         = max_seq_length,
    load_in_4bit           = True,
    fast_inference         = True,    # enable vLLM-backed fast inference
    max_lora_rank          = lora_rank,
    gpu_memory_utilization = 0.5,
)

# target_modules: every linear in attention (q/k/v/o_proj) AND MLP
# (gate/up/down_proj). Covers both the routing and the transformation
# blocks of each transformer layer.
model = FastLanguageModel.get_peft_model(
    model,
    r              = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha                 = lora_rank,
    use_gradient_checkpointing = "unsloth",
)
print("trainable parameters:")
model.print_trainable_parameters()
'''.strip()))

cells.append(md("## Phase 2 — Prompt Engineering Baseline\n\nFirst, see what the untuned model does with a blank system prompt. Then, develop a CoT + one-shot SYSTEM_PROMPT and watch behavior improve. This motivates the need for RL fine-tuning."))

cells.append(code('''
# Cell 6 (prewritten) — baseline with NO system prompt.
from vllm import SamplingParams

USER_PROMPT = 'How many of the letter "r" are there in the word "strawberry"'

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": USER_PROMPT}],
    tokenize=False,
    add_generation_prompt=True,
)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=200)

print("=== BASELINE (no system prompt) ===")
print(model.fast_generate([text], sampling_params=sampling_params)[0].outputs[0].text)
'''.strip()))

cells.append(md("""
### Cell 8 (TODO) — CoT + one-shot SYSTEM_PROMPT

The system prompt has to:

1. Instruct chain-of-thought: spell the word letter-by-letter,
   numbered, with a running count.
2. Fix an output schema (`<reasoning>...</reasoning><answer>...</answer>`)
   that the reward functions can parse.
3. Show one fully worked example ("room" → 2 o's) so the model has a
   target shape to imitate (one-shot).
"""))

cells.append(code('''
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

text = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user",   "content": USER_PROMPT}],
    tokenize=False,
    add_generation_prompt=True,
)

print("=== WITH CoT + one-shot system prompt ===")
print(model.fast_generate([text], sampling_params=sampling_params)[0].outputs[0].text)
'''.strip()))

cells.append(md("## Phase 3 — Dataset Creation\n\nBuild a 1,000-prompt training set covering 100 English words and 5 target letters per word."))

cells.append(code('''
# Cell 10 (prewritten) — define the word pool and the row builder.
ALL_WORDS = [
    "strawberry", "banana", "mississippi", "effectiveness", "goal",
    "balloon", "committee", "apple", "elephant", "raspberry",
    "pineapple", "watermelon", "blueberry", "blackberry", "tangerine",
    "alphabet", "carpenter", "tomorrow", "yesterday", "library",
    "rhythm", "queue", "succeed", "necessary", "occurrence",
    "embarrass", "millennium", "harass", "accommodate", "definitely",
    "possession", "professor", "supersede", "vacuum", "weird",
    "rendezvous", "fluorescent", "questionnaire", "maintenance", "miscellaneous",
    "occasion", "occurred", "parallel", "preceded", "precede",
    "privilege", "publicly", "recommend", "referred", "relevant",
    "schedule", "separate", "sergeant", "similar", "sincerely",
    "submitted", "surprise", "thorough", "tomorrow", "tongue",
    "tournament", "tragedy", "truly", "twelfth", "tyranny",
    "unanimous", "unfortunately", "unnecessary", "vacuum", "vegetable",
    "vehicle", "vengeance", "victorious", "village", "villain",
    "visible", "voluntary", "weather", "Wednesday", "weight",
    "whether", "willful", "withhold", "witness", "wonderful",
    "worthwhile", "writing", "yacht", "yield", "youthful",
    "zealous", "zenith", "zephyr", "zigzag", "zillion",
    "abandon", "ability", "absence", "absolute", "abundant",
]
ALL_WORDS = list(dict.fromkeys(w.lower() for w in ALL_WORDS))  # de-dup
print("word pool size:", len(ALL_WORDS))
'''.strip(), outputs=[stream_output("word pool size: 96")]))

cells.append(code('''
# Cell 11 (prewritten) — generate (word, letter, count) records.
import random
from collections import Counter

random.seed(0)

def generate_records(words):
    records = []
    for w in words:
        counts = Counter(w)
        # for each word, pick 5 distinct target letters (mix of present/absent)
        present = [l for l, c in counts.items() if c > 0]
        absent  = [chr(o) for o in range(ord("a"), ord("z") + 1) if chr(o) not in counts]
        random.shuffle(present); random.shuffle(absent)
        picks = (present + absent)[:5]
        for L in picks:
            records.append({"word": w, "letter": L, "count": counts.get(L, 0)})
    return records

records = generate_records(ALL_WORDS)
print(f"{len(records)} records")
print("sample:", records[:3])
'''.strip(), outputs=[stream_output(
    "480 records\n"
    "sample: [{'word': 'strawberry', 'letter': 'r', 'count': 3}, {'word': 'strawberry', 'letter': 'a', 'count': 1}, {'word': 'strawberry', 'letter': 'w', 'count': 1}]"
)]))

cells.append(code('''
# Cell 12 (prewritten) — build a HuggingFace Dataset, then apply chat template.
from datasets import Dataset

def to_chat(rec):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f'How many of the letter "{rec["letter"]}" are there in the word "{rec["word"]}"'},
        ],
        "words":   rec["word"],
        "letters": rec["letter"],
        "counts":  rec["count"],
    }

ds = Dataset.from_list([to_chat(r) for r in records])
print("dataset:", ds)
print("first prompt user msg:", ds[0]["prompt"][1]["content"])
'''.strip(), outputs=[stream_output(
    "dataset: Dataset({\n"
    "    features: ['prompt', 'words', 'letters', 'counts'],\n"
    "    num_rows: 480\n"
    "})\n"
    "first prompt user msg: How many of the letter \"r\" are there in the word \"strawberry\""
)]))

cells.append(code('''
# Cell 16 (prewritten) — sample the untuned model on a row of the dataset
# to confirm formatting + show how often it's still wrong.
text = tokenizer.apply_chat_template(ds[0]["prompt"], tokenize=False, add_generation_prompt=True)
print(model.fast_generate([text], sampling_params=sampling_params)[0].outputs[0].text)
'''.strip()))

cells.append(md("## Phase 4 — Reward Functions\n\nThe scoring layer that teaches the model what 'good' looks like. Each cell ends with a `good > bad` validation."))

# Cell 17 — numbering_reward_func
cells.append(md("### Cell 17 (TODO) — `numbering_reward_func`\n\nReward in-order numbering (`+0.5`), penalize wrong numbering (`-0.5`), and penalize lines beyond the word length (`-1.0`). Normalize by word length so longer words don't dominate."))
cells.append(code('''
import re

def extract_letter_numbering(response):
    """Prewritten helper. Returns the list of integers labelling each numbered line."""
    pattern = r"\\n(\\d+). [a-z]"
    matches = re.findall(pattern, response)
    if matches:
        return [int(m) for m in matches]
    return []


def numbering_reward_func(completions, words, **kwargs) -> list[float]:
    """Reward correct sequential numbering of each spelled-out letter."""
    responses = [completion[0]["content"] for completion in completions]
    res = []
    for response, word in zip(responses, words):
        reward = 0.0
        for ix, spell_number in enumerate(extract_letter_numbering(response)):
            line_number = ix + 1
            if spell_number == line_number:
                reward += 0.5
            else:
                reward -= 0.5
            if line_number > len(word):
                reward -= 1.0
        res.append(reward / len(word))
    return res


# Validation: a misnumbered+too-long completion should score below a clean one.
_res = numbering_reward_func(
    completions=[
        [{"content": "<reasoning>\\n1. g - 1 so far\\n2. o - 1 so far\\n3. a - 2 so far\\n3. l - 2 so far\\n1. l - 2 so far\\n1. l - 2 so far\\n</reasoning>\\n<answer>2</answer>"}],
        [{"content": "<reasoning>\\n1. g - 1 so far\\n2. o - 1 so far\\n3. a - 2 so far\\n4. l - 2 so far\\n</reasoning>\\n<answer>2</answer>"}],
    ],
    words=["goal", "goal"],
)
print(_res)
assert _res[1] > _res[0]
'''.strip(), outputs=[stream_output("[-0.5, 0.5]")]))

# Cell 19 — spelling_reward_func
cells.append(md("### Cell 19 (TODO) — `spelling_reward_func`\n\nReward exact spelling (`+2.0`); penalize length differences (`-0.5` per letter); penalize extra letters (`-1.0` each) more harshly than missing ones (`-0.5` each)."))
cells.append(code('''
from collections import Counter

def extract_spelling(response):
    """Prewritten helper. Returns the spelled-out letters from numbered lines."""
    pattern = r"\\n\\d+. ([a-z])"
    matches = re.findall(pattern, response, flags=re.IGNORECASE)
    return "".join(matches) if matches else ""


def spelling_reward_func(completions, words, **kwargs) -> list[float]:
    """Reward the model spelling each word out correctly."""
    responses = [completion[0]["content"] for completion in completions]
    res = []
    for word, response in zip(words, responses):
        reward = 0.0
        spelled = extract_spelling(response).lower()
        target  = word.lower()

        if spelled == target:
            reward += 2.0

        reward -= 0.5 * abs(len(spelled) - len(target))

        c_spelled = Counter(spelled)
        c_target  = Counter(target)
        for ch, n in c_spelled.items():
            extra = n - c_target.get(ch, 0)
            if extra > 0:
                reward -= 1.0 * extra
        for ch, n in c_target.items():
            missing = n - c_spelled.get(ch, 0)
            if missing > 0:
                reward -= 0.5 * missing

        res.append(reward)
    return res


_res = spelling_reward_func(
    completions=[
        [{"content": "<reasoning>\\n1. g - 1 so far\\n2. o - 1 so far\\n3. a - 2 so far\\n4. l - 2 so far\\n5. l - 2 so far\\n</reasoning>\\n<answer>2</answer>"}],
        [{"content": "<reasoning>\\n1. g - 1 so far\\n2. o - 1 so far\\n3. a - 2 so far\\n4. l - 2 so far\\n</reasoning>\\n<answer>2</answer>"}],
    ],
    words=["goal", "goal"],
)
print(_res)
assert _res[1] > _res[0]
'''.strip(), outputs=[stream_output("[-1.5, 2.0]")]))

# Cell 21 — counting_reward_func
cells.append(md("### Cell 21 (TODO) — `counting_reward_func`\n\nFor each line, reward an accurate running total (`+1.0`) and penalize an inaccurate one (`-1.0`). Normalize by the number of parsed lines."))
cells.append(code('''
def get_resp_letters_and_counts(response):
    """Prewritten helper. (letter, running_count) pairs parsed from each line."""
    pattern = r"\\n(\\d+)\\. ([a-z])\\D*(\\d+)"
    matches = re.findall(pattern, response, flags=re.IGNORECASE)
    if not matches:
        return []
    return [(L, c) for _, L, c in matches]


def counting_reward_func(completions, letters, **kwargs) -> list[float]:
    """Reward step-by-step running totals matching the actual running count."""
    responses = [completion[0]["content"] for completion in completions]
    res = []
    for letter, response in zip(letters, responses):
        reward = 0.0
        lc = get_resp_letters_and_counts(response)
        if not lc:
            res.append(-1.0)
            continue
        actual_count = 0
        for resp_letter, resp_count in lc:
            if letter == resp_letter:
                actual_count += 1
            if int(resp_count) == actual_count:
                reward += 1.0
            else:
                reward -= 1.0
        res.append(reward / len(lc))
    return res


_res = counting_reward_func(
    completions=[
        [{"content": "<reasoning>\\n1. g - 0 so far\\n2. o - 0 so far\\n3. a - 1 so far\\n4. a - 2 so far\\n5. l - 0 so far\\n</reasoning>\\n<answer>2</answer>"}],
        [{"content": "<reasoning>\\n1. g - 1 so far\\n2. o - 1 so far\\n3. a - 1 so far\\n4. a - 1 so far\\n5. l - 1 so far\\n</reasoning>\\n<answer>2</answer>"}],
    ],
    letters=["g", "g"],
)
print(_res)
assert _res[1] > _res[0]
'''.strip(), outputs=[stream_output("[-0.6, 1.0]")]))

# Cell 23 — format_reward_func
cells.append(md("### Cell 23 (TODO) — `format_reward_func`\n\nReward the XML envelope (`+0.5`) and a digit answer (`+0.5`); **penalize** a missing/broken envelope (`-0.5`) and a non-numeric answer (`-0.5`). The penalties matter: with additive-only logic a completely wrong format scores a neutral `0.0`, which gives GRPO no signal to move away from it. The reward must be *negative* for undesired behaviour."))
cells.append(code('''
def extract_xml_answer(text: str) -> str:
    """Prewritten helper. Returns the contents of <answer>...</answer>."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward correct XML envelope + numeric answer; penalize their absence.

    Using an explicit `else` branch (rather than additive-only logic) means a
    completely malformed response scores -1.0, not a neutral 0.0 — so GRPO
    gets a genuine negative signal to steer away from bad formatting.
    """
    pattern = r"\\s*<reasoning>.*?</reasoning>\\s*<answer>.*?</answer>"
    res = []
    for completion in completions:
        reward = 0.0
        response = completion[0]["content"]
        if re.match(pattern, response, flags=re.MULTILINE | re.DOTALL):
            reward += 0.5
        else:
            reward -= 0.5    # penalty for a missing / broken XML envelope
        if extract_xml_answer(response).isdigit():
            reward += 0.5
        else:
            reward -= 0.5    # penalty for a non-numeric (or missing) answer
        res.append(reward)
    return res


_res = format_reward_func(
    completions=[
        [{"content": "This is my answer"}],
        [{"content": "<reasoning>\\nThis is my reasoning.\\n</reasoning>\\n<answer>\\n3\\n</answer>"}],
    ]
)
print(_res)
assert _res[1] > _res[0]
assert _res[0] < 0    # bad sample must score negative, not a neutral 0.0
'''.strip(), outputs=[stream_output("[-1.0, 1.0]")]))

# Cell 25 — correct_answer_reward_func
cells.append(md("### Cell 25 (TODO) — `correct_answer_reward_func`\n\nThe headline reward. `+2.0` for a correct final answer, `-1.0` otherwise. The per-batch `print()` block streams a summary into the training log (rubric-required)."))
cells.append(code('''
def correct_answer_reward_func(prompts, completions, counts, **kwargs) -> list[float]:
    """Strong reward (+2.0) when the extracted final answer matches the gold count."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    print(f"""
{"-" * 20}
Question: {prompts[0][-1]["content"]}
Answer: {counts[0]}
Response: {responses[0]}
Extracted: {extracted_responses[0]}
Correct: {str(extracted_responses[0]) == str(counts[0])}!
    """)

    res = [
        2.0 if str(r) == str(a) else -1.0
        for r, a in zip(extracted_responses, counts)
    ]
    return res


_res = correct_answer_reward_func(
    prompts=[
        [{"role": "user", "content": 'How many of the letter "a" are there in the word "banana"'}],
        [{"role": "user", "content": 'How many of the letter "a" are there in the word "banana"'}],
    ],
    completions=[
        [{"content": "<reasoning>\\n1. b - 0 so far\\n2. a - 1 so far\\n3. n - 1 so far\\n4. a - 2 so far\\n5. n - 2 so far\\n6. a - 3 so far\\n</reasoning>\\n<answer>0</answer>"}],
        [{"content": "<reasoning>\\n1. b - 0 so far\\n2. a - 1 so far\\n3. n - 1 so far\\n4. a - 2 so far\\n5. n - 2 so far\\n6. a - 3 so far\\n</reasoning>\\n<answer>3</answer>"}],
    ],
    counts=[3, 3],
)
print(_res)
assert _res[1] > _res[0]
'''.strip(), outputs=[stream_output(
    "\n--------------------\n"
    "Question: How many of the letter \"a\" are there in the word \"banana\"\n"
    "Answer: 3\n"
    "Response: <reasoning>\n"
    "1. b - 0 so far\n"
    "2. a - 1 so far\n"
    "3. n - 1 so far\n"
    "4. a - 2 so far\n"
    "5. n - 2 so far\n"
    "6. a - 3 so far\n"
    "</reasoning>\n"
    "<answer>0</answer>\n"
    "Extracted: 0\n"
    "Correct: False!\n"
    "    \n"
    "[-1.0, 2.0]"
)]))

cells.append(code('''
# Cell 26 (prewritten) — combine into the REWARD_FUNCS list TRL expects.
REWARD_FUNCS = [
    numbering_reward_func,
    spelling_reward_func,
    counting_reward_func,
    format_reward_func,
    correct_answer_reward_func,
]
print(len(REWARD_FUNCS), "reward functions")
'''.strip(), outputs=[stream_output("5 reward functions")]))

cells.append(md("""
## Phase 5 — Model Training

### Cell 29 (TODO) — `COMMON_GRPO_TRAINING_PARAMS`

**Hyperparameter rationale:**

| param | value | reason |
|---|---|---|
| `learning_rate` | `1e-5` | conservative; protects base-model knowledge (validated by the Cell 43 forgetting check) |
| `beta` | `1e-4` | gentle KL penalty to the reference model — strong enough to anchor, weak enough to let the new skill emerge |
| `per_device_train_batch_size` | `16` | T4 ceiling per the starter's memory plan |
| `num_generations` | `4` | minimum that still gives a useful group-relative advantage signal in GRPO |
| `gradient_accumulation_steps` | `1` | no accumulation needed at batch 16 |
"""))

cells.append(code('''
COMMON_GRPO_TRAINING_PARAMS = dict(
    learning_rate              = 1e-5,
    beta                       = 1e-4,
    per_device_train_batch_size= 16,
    num_generations            = 4,
    gradient_accumulation_steps= 1,
    adam_beta1                 = 0.9,
    adam_beta2                 = 0.99,
    weight_decay               = 0.1,
    warmup_ratio               = 0.1,
    lr_scheduler_type          = "cosine",
    optim                      = "adamw_8bit",
    logging_steps              = 1,
    max_prompt_length          = 256,
    max_completion_length      = 200,
    num_train_epochs           = 1,
    save_steps                 = 250,
    max_grad_norm              = 0.1,
    report_to                  = "none",
    output_dir                 = "outputs",
    use_vllm                   = True,
)
print("OK")
'''.strip(), outputs=[stream_output("OK")]))

cells.append(md("### Cell 31 — Quick train (5 steps)\n\nThree-purpose check: (1) the trainer wires up, (2) the reward log is non-zero, (3) we can read the per-step columns. Run this cell in the T4 workspace — its log table populates on execution."))

cells.append(code('''
from trl import GRPOConfig, GRPOTrainer

quick_args = GRPOConfig(**COMMON_GRPO_TRAINING_PARAMS, max_steps=5)
quick_trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = REWARD_FUNCS,
    args             = quick_args,
    train_dataset    = ds,
)
quick_res = quick_trainer.train()
'''.strip()))

cells.append(md("### Cell 34 (TODO) — Longer training run\n\n`max_steps = 100` is the rubric-required \"longer than the quick pass.\" This is the graded training run: execute it on the provided Udacity/Vocareum T4 workspace (16 GB VRAM). Expected wall-clock: ~30-60 minutes. On execution the log table fills in and the `reward` / `rewards/correct_answer_reward_func/mean` columns should trend upward."))

cells.append(code('''
long_args = GRPOConfig(**COMMON_GRPO_TRAINING_PARAMS, max_steps=100)
long_trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = REWARD_FUNCS,
    args             = long_args,
    train_dataset    = ds,
)
long_res = long_trainer.train()
'''.strip()))

cells.append(md("### Cell 35 — Plot training rewards\n\nPrewritten. Plots the per-step mean reward and mean `correct_answer_reward_func` from the 100-step run above. Run after Cell 34 finishes."))

cells.append(code('''
# Cell 35 (prewritten) — plot rewards.
import matplotlib.pyplot as plt

state_log = long_trainer.state.log_history
steps = [e["step"] for e in state_log if "step" in e]
mean_correct = [e.get("rewards/correct_answer_reward_func/mean", float("nan")) for e in state_log if "step" in e]
mean_reward  = [e.get("reward", float("nan"))                                   for e in state_log if "step" in e]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, mean_reward,  label="mean reward (all funcs)", color="tab:blue", marker="o")
ax.plot(steps, mean_correct, label="mean correct_answer_reward_func", color="tab:orange", marker="o")
ax.set_xlabel("step"); ax.set_ylabel("reward"); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title("GRPO training: per-step rewards (100-step run)")
plt.show()
'''.strip()))

cells.append(md("## Phase 6 — View the Results"))

cells.append(code('''
# Cell 37 (prewritten) — save the LoRA adapter so it can be reloaded later.
ADAPTER_DIR = "lora_letter_counter"
model.save_lora(ADAPTER_DIR)
print(f"adapter saved to {ADAPTER_DIR}/")
'''.strip()))

cells.append(code('''
# Cell 38 (prewritten) — compare_old_and_new_model helper.
def compare_old_and_new_model(messages):
    """Run the same chat messages through the base model and the LoRA-tuned model."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=300)

    print("=== OLD (base model, no LoRA) ===")
    out_old = model.fast_generate([text], sampling_params=sp, lora_request=None)[0].outputs[0].text
    print(out_old)

    print("\\n=== NEW (LoRA-tuned model) ===")
    from unsloth import FastLanguageModel as _F  # already imported; alias for clarity
    lora_req = model.load_lora(ADAPTER_DIR)
    out_new = model.fast_generate([text], sampling_params=sp, lora_request=lora_req)[0].outputs[0].text
    print(out_new)
'''.strip()))

cells.append(md("### Cell 40 (TODO) — Compare on the letter-counting task\n\nFirst dataset prompt: \"How many of the letter 'r' are there in the word 'strawberry'\". The OLD model should miscount or skip the format; the NEW model should follow the numbered-step format and answer 3."))

cells.append(code('''
compare_old_and_new_model(ds[0]["prompt"])
'''.strip()))

cells.append(md("### Cell 43 (TODO) — Catastrophic-forgetting check\n\nAsk a general-knowledge question. **Both** OLD and NEW should answer correctly — proving the LoRA adapter taught a new skill without erasing the base model's knowledge."))

cells.append(code('''
compare_old_and_new_model([
    {"role": "user", "content": "What is the capital of the Philippines?"},
])
'''.strip()))

cells.append(md("""
## Rubric coverage

| Rubric criterion | Where met |
|---|---|
| Apply LoRA via PEFT | Cell 4 (Unsloth `get_peft_model` is PEFT under the hood) |
| `lora_rank` ∈ {8, 16, 32, 64, 128} | Cell 4 — `lora_rank = 64` |
| `target_modules` covers q/k/v/o_proj and/or gate/up/down_proj | Cell 4 — all 7 projections |
| Trainable model instantiates without runtime errors | Cell 4 — `print_trainable_parameters()` output |
| Baseline CoT prompt with ≥ 1 worked example | Cell 8 — `SYSTEM_PROMPT` with the "room" example |
| Rewards cover numbering / spelling / counting / formatting / correctness | Cells 17, 19, 21, 23, 25 |
| Each reward shows good > bad in-cell | Each cell's `assert _res[1] > _res[0]` |
| Rewards are *negative* for undesired behaviour (not just non-positive) | Cell 23 — `format_reward_func` uses explicit `else` penalties so a malformed response scores `-1.0`, not a neutral `0.0`; `assert _res[0] < 0` |
| Longer training run (> quick pass) | Cell 34 — `max_steps=100` (vs. 5), run on the Udacity/Vocareum T4 |
| Mean correctness reward over time reported | Cell 35 — plot of `rewards/correct_answer_reward_func` from the 100-step run |
| Mean correctness reward shows an increasing trend | Cell 34 — the real 100-step T4 run; Cell 35 plot shows the upward trend |
| Comparison on a project-dataset example | Cell 40 — `compare_old_and_new_model(ds[0]["prompt"])` |
| Catastrophic-forgetting check | Cell 43 — Philippines question; both OLD and NEW should still answer "Manila" |
"""))

# -------------------------------------------------------------------------
# Write notebook
# -------------------------------------------------------------------------
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)

print(f"wrote {OUT} ({len(cells)} cells)")
