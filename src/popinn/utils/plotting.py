import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot the training loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(history["total"], label="Total", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()

    ax = axes[1]
    for key in ["pde", "ic", "bc_left", "bc_right", "non_neg"]:
        vals = history[key]
        if any(v > 0 for v in vals):
            ax.semilogy(vals, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Component Losses")
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('training_history.png')