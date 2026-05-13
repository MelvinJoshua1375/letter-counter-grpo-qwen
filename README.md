# Letter-Counting via GRPO + LoRA on Qwen2.5-3B-Instruct

Teach an instruction-tuned LLM to count occurrences of a letter in a word
by reasoning step-by-step. The base model is **Qwen2.5-3B-Instruct**.
A **LoRA** adapter is attached (via PEFT) and trained with **GRPO** (Group
Relative Policy Optimization) using **Unsloth** for fast LoRA training and
**vLLM** for high-throughput rollouts.

Submission for the Udacity Generative AI Fundamentals Nanodegree, Project 1.
Reworked for **local execution** on a single NVIDIA GPU (>= 16 GB VRAM)
instead of the Vocareum container.

## Repo layout

```
letter-counter-grpo-qwen/
├── README.md
├── LICENSE
├── requirements.txt
├── project/
│   └── starter/
│       └── gen_ai_fundamentals_project_starter.ipynb   <- the submission notebook
├── scripts/
│   ├── build_notebook.py        <- rebuilds the notebook from the cell sources
│   └── cpu_smoke_train.py       <- end-to-end pipeline check on CPU (no GPU needed)
├── outputs_smoke/               <- artefacts from the CPU smoke run
│   ├── log_history.json         <- per-step rewards from the 4-step CPU smoke training
│   └── checkpoint-*/            <- LoRA adapter from the smoke run
└── plots/
    └── training_rewards.png     <- rendered plot of the reward trend
```

## Setup (local NVIDIA GPU)

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create the env and install dependencies:

   ```sh
   uv venv .venv --python 3.12
   source .venv/bin/activate
   uv pip install -r requirements.txt
   python -m ipykernel install --user --name=letter-counter-grpo --display-name "Python (letter-counter-grpo)"
   ```

3. Launch Jupyter and run the notebook:

   ```sh
   jupyter lab project/starter/gen_ai_fundamentals_project_starter.ipynb
   ```

   Select the `Python (letter-counter-grpo)` kernel. Run every cell in order.

Hardware requirement: single NVIDIA GPU with >= 16 GB VRAM (e.g. T4, L4,
RTX 4080+) and CUDA 12.x. Full 100-step training takes ~30–60 minutes on a T4.

## CPU smoke test (no GPU required)

The Unsloth + vLLM stack assumes CUDA, but the rest of the pipeline (GRPO
loop, reward functions, LoRA adapter, dataset, system prompt) is plain
PyTorch / TRL / PEFT and runs fine on CPU. `scripts/cpu_smoke_train.py`
swaps in a smaller base model so the whole loop fits in ~2 GB RAM:

| | GPU notebook | CPU smoke |
|---|---|---|
| base model | `Qwen/Qwen2.5-3B-Instruct` (4-bit) | `Qwen/Qwen2.5-0.5B-Instruct` (fp32) |
| inference engine | vLLM (`fast_generate`) | `transformers.generate` |
| `per_device_train_batch_size` | 16 | 1 |
| `num_generations` | 4 | 2 (TRL minimum) |
| `max_steps` | 100 | configurable, default 3 |
| reward funcs | same | same |
| LoRA target modules | same | same |

Run it:

```sh
MAX_STEPS=4 python scripts/cpu_smoke_train.py
```

It writes a real per-step `log_history.json` into `outputs_smoke/` that
the notebook can quote.

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
| Longer training run (> quick pass) | Cell 34 — `max_steps=100` (vs. 5) |
| Mean correctness reward over time reported | Cell 35 — plot of `rewards/correct_answer_reward_func` |
| Mean correctness reward increasing trend, OR documented anomaly + remedial config | Cell 34 — local CPU machine documented; remediation: T4 + same `COMMON_GRPO_TRAINING_PARAMS` |
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
