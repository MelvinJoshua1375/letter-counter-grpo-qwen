"""Render outputs_smoke/log_history.json -> plots/training_rewards.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "outputs_smoke" / "log_history.json"
OUT = ROOT / "plots" / "training_rewards.png"

with open(LOG) as f:
    log = json.load(f)

steps = [e["step"] for e in log if "step" in e]
mean_correct = [e.get("rewards/correct_answer_reward_func/mean", float("nan")) for e in log if "step" in e]
mean_reward = [e.get("reward", float("nan")) for e in log if "step" in e]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, mean_reward,  label="mean reward (all funcs)",        color="tab:blue",   marker="o")
ax.plot(steps, mean_correct, label="mean correct_answer_reward_func", color="tab:orange", marker="o")
ax.set_xlabel("step")
ax.set_ylabel("reward")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title(f"GRPO training (CPU smoke run, Qwen2.5-0.5B-Instruct, {max(steps)} steps)")
fig.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
