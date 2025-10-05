# visualisation_params.py
import matplotlib.pyplot as plt
import numpy as np

def plot_training_dynamics(loss_hist, grad_hist, wd_hist, log_interval=10, switch_step=24000):
    """
    Plot loss, gradient norm, and weight decay evolution during training.

    Args:
        loss_hist (list): Recorded training loss values.
        grad_hist (list): Recorded gradient norms.
        wd_hist (list): Recorded weight decay values per step.
        log_interval (int): Logging interval used in training.
        switch_step (int): Step at which weight decay changes.
    """
    steps = np.arange(0, len(loss_hist) * log_interval, log_interval)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(steps, loss_hist, label="Loss", color="pink", linewidth=1, solid_capstyle="round")
    ax1.plot(steps, grad_hist, label="Gradient Norm", color="orange", linewidth=1, solid_capstyle="round")
    ax2.plot(steps, wd_hist, label="Weight Decay", color="brown", linewidth=2, linestyle="--", solid_capstyle="round")

    ax1.axvline(x=switch_step, color="yellow", linestyle=":", label="Weight Decay Switch")

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss / Gradient Norm")
    ax2.set_ylabel("Weight Decay")

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")

    plt.title("Training Dynamics: Loss, Gradient Norm, and Weight Decay")
    plt.tight_layout()
    plt.savefig("Plots/training_dynamics.png")
    plt.show()
