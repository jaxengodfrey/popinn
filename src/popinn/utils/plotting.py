import matplotlib.pyplot as plt

def plot_training_history(history: dict, save_path: str = None):
    """Plot training loss curves from the history dict returned by train_model.

    Args:
        history:   Dict mapping component names to lists of scalar values.
                   Must contain a ``"total"`` key.
        save_path: If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(history["total"], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")

    ax = axes[1]
    for key, vals in history.items():
        if key == "total":
            continue
        if any(v > 0 for v in vals):
            ax.semilogy(vals, label=key)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Component Losses")
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
