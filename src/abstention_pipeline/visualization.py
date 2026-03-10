"""Training curve visualization from HuggingFace Trainer state."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def load_trainer_state(path: Path) -> Dict:
    """Load trainer_state.json from a training output directory."""
    if path.is_dir():
        path = path / "trainer_state.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_training_curves(state: Dict) -> Dict[str, List]:
    """Extract loss, LR, grad_norm, and eval_loss per step from trainer state."""
    log_history = state.get("log_history", [])

    curves = {
        "step": [],
        "train_loss": [],
        "learning_rate": [],
        "grad_norm": [],
        "eval_step": [],
        "eval_loss": [],
    }

    for entry in log_history:
        if "loss" in entry:
            curves["step"].append(entry["step"])
            curves["train_loss"].append(entry["loss"])
            curves["learning_rate"].append(entry.get("learning_rate"))
            curves["grad_norm"].append(entry.get("grad_norm"))
        if "eval_loss" in entry:
            curves["eval_step"].append(entry["step"])
            curves["eval_loss"].append(entry["eval_loss"])

    return curves


def plot_training_curves(
    curves: Dict[str, List],
    title: str = "Training Curves",
    save_path: Optional[Path] = None,
) -> None:
    """Plot training loss and LR schedule (eval loss overlaid on loss plot if available)."""
    has_eval = len(curves.get("eval_loss", [])) > 0
    has_lr = any(v is not None for v in curves.get("learning_rate", []))

    n_plots = 1 + int(has_lr)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Train loss (with eval loss overlay if available)
    axes[idx].plot(curves["step"], curves["train_loss"], label="train_loss", color="steelblue")
    if has_eval:
        axes[idx].plot(curves["eval_step"], curves["eval_loss"], label="eval_loss", color="tomato", marker="o", markersize=4)
    axes[idx].set_xlabel("Step")
    axes[idx].set_ylabel("Loss")
    axes[idx].set_title("Loss")
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    # Learning rate
    if has_lr:
        lr_vals = [v for v in curves["learning_rate"] if v is not None]
        lr_steps = [s for s, v in zip(curves["step"], curves["learning_rate"]) if v is not None]
        axes[idx].plot(lr_steps, lr_vals, color="green")
        axes[idx].set_xlabel("Step")
        axes[idx].set_ylabel("Learning Rate")
        axes[idx].set_title("LR Schedule")
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    plt.show()
