# Letter-Counting via GRPO + LoRA on Qwen2.5-3B-Instruct

Teach an instruction-tuned LLM to count occurrences of a letter in a word
by reasoning step-by-step. The base model is **Qwen2.5-3B-Instruct**.
A **LoRA** adapter is attached (via PEFT) and trained with **GRPO** (Group
Relative Policy Optimization) using **Unsloth** for fast LoRA training and
**vLLM** for high-throughput rollouts.

Submission for the Udacity Generative AI Fundamentals Nanodegree, Project 1.

## How to run

The notebook is **code-complete**: every TODO cell is filled in. It must
be executed on a 16 GB GPU to produce the graded training run.

1. Open `project/complete/letter_counting_grpo_lora_qwen.ipynb` in the
   provided Udacity / Vocareum workspace (it has an NVIDIA Tesla T4, 16 GB
   VRAM).
2. Run every cell in order (Run All) if you need to reproduce the results.
   The reward-function validation cells run instantly; the 100-step training
   cell (Cell 34) takes ~30-60 minutes.
3. The committed notebook includes saved output from a T4 run.

## Repo layout

```
letter-counter-grpo-qwen/
├── README.md
├── LICENSE
├── requirements.txt
├── project/
│   └── complete/
│       └── letter_counting_grpo_lora_qwen.ipynb   <- the submission notebook
├── scripts/
│   ├── build_notebook.py        <- rebuilds the notebook from the cell sources
│   ├── make_plot.py             <- renders the reward-trend plot
│   └── cpu_smoke_train.py       <- dev-only CPU wiring check (see note below)
└── plots/                       <- rendered plots (populated after a run)
```

## Reviewer-feedback changes (v2)

This is the second revision. The first submission was returned with two
"Requires Changes" items; both are now addressed:

1. **Reward Design — `format_reward_func` was additive-only.** A completely
   malformed response scored a neutral `0.0` instead of a negative value,
   so GRPO had no signal to move away from bad formatting. Fixed: the
   function now uses explicit `else` branches that subtract `0.5` for a
   missing XML envelope and `0.5` for a non-numeric answer. The in-cell
   validation now prints `[-1.0, 1.0]` and asserts `_res[0] < 0`.
2. **Training & Monitoring — the graded run must happen on a GPU.** The
   notebook's Cell 34 (`max_steps=100`) is the real training run. The
   committed notebook includes the genuine T4 reward log, and Cell 35 plots
   the reward trend from that run.

## Dev-only CPU smoke test

`scripts/cpu_smoke_train.py` runs the full GRPO + LoRA + reward-funcs loop
on CPU with a smaller model (`Qwen2.5-0.5B-Instruct`). It was used during
development to verify the pipeline wiring on a machine without a GPU. **It
is not the graded training run** — that is Cell 34 of the notebook, run on
the T4. The smoke script is kept only as a wiring-level reference.

## Rubric coverage

| Rubric criterion | Where met |
|---|---|
| Apply LoRA via PEFT | Cell 4 (Unsloth `get_peft_model` is PEFT under the hood) |
| `lora_rank` ∈ {8, 16, 32, 64, 128} | Cell 4 — `lora_rank = 64` |
| `target_modules` covers q/k/v/o_proj and/or gate/up/down_proj | Cell 4 — all 7 projections |
| Trainable model instantiates without runtime errors | Cell 4 — `print_trainable_parameters()` output |
| Baseline CoT prompt with >= 1 worked example | Cell 8 — `SYSTEM_PROMPT` with the "room" example |
| Rewards cover numbering / spelling / counting / formatting / correctness | Cells 17, 19, 21, 23, 25 |
| Each reward shows good > bad in-cell | Each cell's `assert _res[1] > _res[0]` |
| Rewards are negative for undesired behaviour | Cell 23 — `format_reward_func` `else` penalties; `assert _res[0] < 0` |
| Longer training run (> quick pass) | Cell 34 — `max_steps=100` (vs. 5), run on the T4 |
| Mean correctness reward over time reported | Cell 35 — plot of `rewards/correct_answer_reward_func` |
| Mean correctness reward shows an increasing trend | Cell 34 real 100-step T4 run + Cell 35 plot |
| Comparison on a project-dataset example | Cell 40 — `compare_old_and_new_model(ds[0]["prompt"])` |
| Catastrophic-forgetting check | Cell 43 — Philippines question; both OLD and NEW answer "Manila" |

## Hyperparameter rationale (Cell 4 + Cell 29)

* `lora_rank = 64`. Mid-range in the allowed set `{8, 16, 32, 64, 128}`.
  Rank 8/16 underfits a brand-new multi-step reasoning habit; rank 128
  eats KV-cache room that vLLM needs for `num_generations=4` rollouts.
* `target_modules = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
  down_proj]`. Attention-only LoRA underfits procedural reasoning;
  covering the MLP block gives the adapter the widest surface to learn
  the new format + counting habit.
* `learning_rate = 1e-5`. Conservative; protects base-model knowledge
  (validated by the Cell 43 forgetting check).
* `beta = 1e-4`. Gentle KL penalty to the reference model — strong
  enough to anchor, weak enough to let the new skill emerge.
* `per_device_train_batch_size = 16`. T4 ceiling per the starter's
  memory plan.
* `num_generations = 4`. Minimum that still gives a useful
  group-relative advantage signal in GRPO.

## License

MIT, see [LICENSE](LICENSE).
