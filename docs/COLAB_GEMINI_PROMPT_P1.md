# Prompt for Gemini in Google Colab — GenAI Project 1 (Letter-Counting GRPO + LoRA)

## HOW TO USE THIS FILE IN COLAB

Gemini in Colab is a chat panel — it does not "open" a `.md` file like an
IDE agent does. You feed it the text. Two ways:

**Option A — paste (simplest, recommended).**
1. Open this `.md` file in any text editor, select everything **below the
   `=== PROMPT STARTS HERE ===` line**, and copy it.
2. In Colab, open the Gemini panel (the ✨ / "Gemini" button, top-right, or
   the chat icon in the left sidebar).
3. Paste the copied text into the Gemini chat box and send.

**Option B — upload the file and point Gemini at it.**
1. In Colab's file browser (folder icon, left sidebar), click **Upload**
   and upload this `.md` file into the session.
2. In the Gemini chat, type: *"Read the uploaded file
   `COLAB_GEMINI_PROMPT_P1.md` and follow its instructions exactly."*
3. (You can also `print(open('COLAB_GEMINI_PROMPT_P1.md').read())` in a
   cell so the content is visible, then reference it.)

Either way: also make sure the **44-cell notebook**
`gen_ai_fundamentals_project_starter.ipynb` is the file open in Colab — see
the first instruction in the prompt.

---

=== PROMPT STARTS HERE ===

You are Gemini, working inside a Google Colab notebook. This is the
**Udacity GenAI Fundamentals — Project 1: Letter-Counting via GRPO + LoRA
on Qwen2.5-3B-Instruct**. Your task is narrow: **statically verify and fix
the notebook's code**, then stop. The user will Run All themselves.

## STEP 0 — CONFIRM YOU HAVE THE RIGHT NOTEBOOK (do this first)

The correct notebook is `gen_ai_fundamentals_project_starter.ipynb` and it
has **~44 cells**, including 11 TODO cells, 5 reward functions, a dataset
section, and a real GRPO training cell.

**If the open notebook has only ~11-12 cells, STOP immediately.** That is a
stripped-down stub, not the project. Tell the user: "This is the wrong
file — you need the 44-cell `gen_ai_fundamentals_project_starter.ipynb`
from the repo `MelvinJoshua1375/letter-counter-grpo-qwen`, path
`project/starter/`." Do not try to rebuild the project from the stub. Do
not write a new notebook.

Only proceed past Step 0 once you have confirmed the notebook has ~44 cells.

## CONTEXT — THIS IS A RE-SUBMISSION

Project 1 was submitted once and returned **Requires Changes**. Three
rubric items passed, two failed. Do not regress the passing ones.

| Rubric item | Verdict |
|---|---|
| Model Setup (LoRA config) | PASS — keep as is |
| Baseline Prompting (CoT) | PASS — keep as is |
| Reward Design & Validation | **REQUIRES CHANGES** — reviewer note 1 |
| Training & Monitoring | **REQUIRES CHANGES** — reviewer note 2 |
| Final Comparison | PASS — keep as is |

### Reviewer note 1 — Reward Design & Validation (REQUIRES CHANGES)

> "The current `format_reward_func` only uses additive logic
> (`reward += 0.5`). If the model provides a completely incorrect format,
> it receives a `0.0` (neutral) rather than a negative penalty."
>
> TO PASS: add an `else` block with a negative penalty:
> ```
> if match:
>     reward += 0.5
> else:
>     reward -= 0.5  # negative penalty for missing tags
> if extracted_answer.strip().isdigit():
>     reward += 0.5
> else:
>     reward -= 0.5  # negative penalty for non-numeric answers
> ```
> Rubric: "rewards that are positive for correct/desired behavior and
> **negative** for incorrect/undesired behavior."

The corrected Cell 23 code is below — its validation prints `[-1.0, 1.0]`
and asserts the bad sample is negative.

### Reviewer note 2 — Training & Monitoring (REQUIRES CHANGES)

> "The core objective of this project is to actually teach an LLM to
> reason. A smoke test on a smaller model cannot achieve this... you must
> execute the actual training run (taking 100 steps) on a capable GPU so
> the model has enough time to learn and the reward plots show a
> meaningful upward trend."

So Cell 34 must contain a real `GRPOTrainer(...).train()` call with
`max_steps=100`. It must NOT be a markdown "training complete" summary, a
smoke test, or a smaller model. (The user runs it — you only verify the
code is correct.)

## SCOPE — VERIFY THE CODE ONLY, DO NOT RUN ANYTHING

- **Do NOT execute the notebook, the model, the trainer, or any cell.**
- Do NOT run `trainer.train()`. Do NOT start the 30-60 minute training.
- You may read the notebook and edit cell source. You may do a cheap
  static syntax check (e.g. `ast.parse` on each cell's source). Nothing
  that loads the model or hits the GPU.
- The user will press **Run All** themselves after you confirm the code.

## COLAB ENVIRONMENT NOTES

- Set the runtime to a **T4 GPU**: `Runtime → Change runtime type → T4 GPU`.
  (This is for the user's later Run All — not something you execute.)
- On Colab the dependencies are **not** pre-installed. **Cell 1** has the
  install line, commented out. It must be **uncommented** so the user's
  Run All installs everything:
  ```python
  !pip install -q unsloth vllm trl peft datasets transformers accelerate matplotlib
  ```
  Make sure Cell 1 has that line uncommented. (Unlike a restricted
  workspace, Colab exposes the CUDA toolchain, so this install works.)
- If `%load_ext autotime` appears and would error, it is safe to delete
  that single line — it is not graded.

## THE 11 TODO CELLS — EXACT CODE EACH MUST CONTAIN

Compare each TODO cell's current source to the solution below. If a cell
has a `**********` placeholder, stale code, or any difference, edit the
cell so it matches exactly. TODO cells are: 4, 8, 17, 19, 21, 23, 25, 29,
34, 40, 43.

### Cell 4 — Load Qwen2.5-3B-Instruct + attach LoRA

```python
import unsloth  # patches transformers; must import before transformers
from unsloth import FastLanguageModel
import torch

max_seq_length = 384

# LoRA rank: 64 — mid-range of the allowed {8,16,32,64,128}. Rank 8/16
# underfits a new multi-step reasoning habit; rank 128 eats KV-cache room
# vLLM needs for num_generations=4 rollouts. 64 is the balanced default.
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
# (gate/up/down_proj). Attention-only LoRA underfits procedural reasoning;
# covering the MLP block gives the adapter the widest learning surface.
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
```

### Cell 8 — CoT + one-shot SYSTEM_PROMPT

```python
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
```

### Cell 17 — numbering_reward_func

```python
import re


def extract_letter_numbering(response):
    """Prewritten helper. Returns the list of integers labelling each numbered line."""
    pattern = r"\n(\d+). [a-z]"
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
        [{"content": "<reasoning>\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n3. l - 2 so far\n1. l - 2 so far\n1. l - 2 so far\n</reasoning>\n<answer>2</answer>"}],
        [{"content": "<reasoning>\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n4. l - 2 so far\n</reasoning>\n<answer>2</answer>"}],
    ],
    words=["goal", "goal"],
)
print(_res)
assert _res[1] > _res[0]
```

### Cell 19 — spelling_reward_func

```python
from collections import Counter


def extract_spelling(response):
    """Prewritten helper. Returns the spelled-out letters from numbered lines."""
    pattern = r"\n\d+. ([a-z])"
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
        [{"content": "<reasoning>\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n4. l - 2 so far\n5. l - 2 so far\n</reasoning>\n<answer>2</answer>"}],
        [{"content": "<reasoning>\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n4. l - 2 so far\n</reasoning>\n<answer>2</answer>"}],
    ],
    words=["goal", "goal"],
)
print(_res)
assert _res[1] > _res[0]
```

### Cell 21 — counting_reward_func

```python
def get_resp_letters_and_counts(response):
    """Prewritten helper. (letter, running_count) pairs parsed from each line."""
    pattern = r"\n(\d+)\. ([a-z])\D*(\d+)"
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
        [{"content": "<reasoning>\n1. g - 0 so far\n2. o - 0 so far\n3. a - 1 so far\n4. a - 2 so far\n5. l - 0 so far\n</reasoning>\n<answer>2</answer>"}],
        [{"content": "<reasoning>\n1. g - 1 so far\n2. o - 1 so far\n3. a - 1 so far\n4. a - 1 so far\n5. l - 1 so far\n</reasoning>\n<answer>2</answer>"}],
    ],
    letters=["g", "g"],
)
print(_res)
assert _res[1] > _res[0]
```

### Cell 23 — format_reward_func  ← THE REVIEWER-FLAGGED FIX

This cell MUST use explicit `else` branches so a malformed response scores
**negative**, not a neutral `0.0`. The validation must include
`assert _res[0] < 0`. If the current cell only adds (`reward += 0.5`
without `else`) and prints `[0.0, 1.0]`, that is exactly the
reviewer-flagged bug — replace it with this:

```python
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
    pattern = r"\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
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
        [{"content": "<reasoning>\nThis is my reasoning.\n</reasoning>\n<answer>\n3\n</answer>"}],
    ]
)
print(_res)
assert _res[1] > _res[0]
assert _res[0] < 0    # bad sample must score negative, not a neutral 0.0
```

### Cell 25 — correct_answer_reward_func

```python
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
        [{"content": "<reasoning>\n1. b - 0 so far\n2. a - 1 so far\n3. n - 1 so far\n4. a - 2 so far\n5. n - 2 so far\n6. a - 3 so far\n</reasoning>\n<answer>0</answer>"}],
        [{"content": "<reasoning>\n1. b - 0 so far\n2. a - 1 so far\n3. n - 1 so far\n4. a - 2 so far\n5. n - 2 so far\n6. a - 3 so far\n</reasoning>\n<answer>3</answer>"}],
    ],
    counts=[3, 3],
)
print(_res)
assert _res[1] > _res[0]
```

### Cell 29 — COMMON_GRPO_TRAINING_PARAMS

```python
COMMON_GRPO_TRAINING_PARAMS = dict(
    learning_rate              = 1e-5,    # conservative — protects base knowledge
    beta                       = 1e-4,    # gentle KL penalty to the reference model
    per_device_train_batch_size= 16,      # T4 ceiling per the starter's memory plan
    num_generations            = 4,       # min for a useful group-relative advantage
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
```

### Cell 34 — Longer training run  ← THE OTHER REVIEWER-FLAGGED ITEM

The cell's code MUST be a real `trainer.train()` call for `max_steps=100`
on the 3B model — exactly as below. Not a markdown "training complete"
summary, not a smoke test, not a smaller model. (You verify the code; the
user runs it — ~30-60 min on the T4.)

```python
long_args = GRPOConfig(**COMMON_GRPO_TRAINING_PARAMS, max_steps=100)
long_trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = REWARD_FUNCS,
    args             = long_args,
    train_dataset    = ds,
)
long_res = long_trainer.train()
```

### Cell 40 — Compare on the letter-counting task

```python
compare_old_and_new_model(ds[0]["prompt"])
```

### Cell 43 — Catastrophic-forgetting check

```python
compare_old_and_new_model([
    {"role": "user", "content": "What is the capital of the Philippines?"},
])
```

## PREWRITTEN "JUST RUN" CELLS — do not edit, just confirm present

- **Cell 31** — quick train (`max_steps=5`), wires up `GRPOConfig` /
  `GRPOTrainer`.
- **Cell 35** — `matplotlib` plot of per-step `reward` and
  `rewards/correct_answer_reward_func/mean` from the Cell 34 run.
- **Cell 37** — `model.save_lora("lora_letter_counter")`.
- **Cell 38** — defines `compare_old_and_new_model(messages)` (toggles the
  LoRA adapter via `lora_request`).
- Dataset cells (≈10-16) — build `ALL_WORDS`, `generate_records`, the `ds`
  HuggingFace Dataset, and a sample untuned generation.

## YOUR TASK — STEP BY STEP (verification only)

1. Confirm Step 0 — the notebook has ~44 cells. If not, stop and tell the
   user it is the wrong file.
2. Confirm Cell 1's `!pip install` line is **uncommented** (Colab needs it).
3. Go through the 11 TODO cells (4, 8, 17, 19, 21, 23, 25, 29, 34, 40, 43).
   Compare each to the solution above. Edit any cell with a `**********`
   placeholder, stale code, or a difference so it matches exactly.
4. Confirm no cell was deleted or replaced with a markdown "summary" — in
   particular Cell 34 must contain the real `GRPOTrainer(...).train()`
   call, not a markdown claim.
5. Confirm the prewritten cells above are still present and intact.
6. Optional cheap check: `ast.parse` each code cell's source for syntax
   errors. Do NOT run any cell.
7. Save the notebook. **Stop.** Report a short summary: notebook cell
   count, which TODO cells were already correct, which you fixed, and
   explicit confirmation that (a) Cell 23 has the `else` negative-penalty
   branches + `assert _res[0] < 0`, and (b) Cell 34 has the real
   `max_steps=100` `trainer.train()` call.

Then the user sets the T4 runtime and presses Run All themselves.

## STATIC VERIFICATION CHECKLIST (no execution)

- [ ] Notebook has ~44 cells (NOT ~12 — if 12, it is the wrong file).
- [ ] Cell 1 — `!pip install` line is uncommented.
- [ ] Cell 4 — `lora_rank = 64`; `target_modules` lists all 7 projections;
      ends with `model.print_trainable_parameters()`.
- [ ] Cell 8 — `SYSTEM_PROMPT` has CoT instructions + the one-shot "room"
      example.
- [ ] Cells 17/19/21/23/25 — each reward function defined; each ends with a
      validation block that `assert`s good > bad.
- [ ] **Cell 23** — has BOTH `else: reward -= 0.5` branches; expected
      printed result `[-1.0, 1.0]`; line `assert _res[0] < 0` present.
- [ ] Cell 29 — `COMMON_GRPO_TRAINING_PARAMS` with `use_vllm=True`,
      `num_generations=4`, `per_device_train_batch_size=16`.
- [ ] Cell 34 — `GRPOConfig(**COMMON_GRPO_TRAINING_PARAMS, max_steps=100)`
      and a real `long_trainer.train()` call. NOT a markdown cell.
- [ ] Cells 40 & 43 — call `compare_old_and_new_model(...)` as shown.
- [ ] Prewritten cells (31, 35, 37, 38, dataset cells) still present.
- [ ] Every code cell's source parses without syntax errors.

## DO NOT

- **Do not run the notebook, the model, the trainer, or any cell.** The
  user runs Run All — keep your session short.
- **Do not create a new notebook or a "demo" notebook.** Work only on the
  existing 44-cell `gen_ai_fundamentals_project_starter.ipynb`.
- Do not fabricate training output or replace Cell 34 with a markdown
  "training complete" claim with invented numbers.
- Do not swap in a smaller model or fewer steps. It is
  `Qwen/Qwen2.5-3B-Instruct` and `max_steps=100`.
- Do not change cells for the 3 already-passing rubric items beyond
  matching the code above.

When the checklist is all true, tell the user the code is verified and the
notebook is ready for them to set the T4 runtime and Run All.
