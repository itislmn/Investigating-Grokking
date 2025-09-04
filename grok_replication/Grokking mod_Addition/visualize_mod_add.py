import numpy as np
import matplotlib.pyplot as plt

def smooth(values, box_pts=50):
    """
    Simple moving average smoothing.
    """
    if len(values) < box_pts:
        return values  # not enough points to smooth
    box = np.ones(box_pts) / box_pts
    return np.convolve(values, box, mode="same")

def plot_grokking(train_accuracies, val_accuracies, title="Grokking Phenomena: Modular Addition (p=113) (50% datatsplit)"):
    """
    Plot the training accuracy and validation accuracy on the same graph to show Grokking.

    Args:
    - train_accuracies (list): List of training accuracy values recorded per log step.
    - val_accuracies (list): List of validation accuracy values recorded per log step.
    """
    steps = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(10, 6))

    # Apply smoothing
    train_smooth = smooth(train_accuracies, box_pts=50)
    val_smooth   = smooth(val_accuracies, box_pts=50)

    # Plot both training and validation accuracy
    plt.plot(steps, train_smooth, label="Training Accuracy", color='blue', linewidth=1, solid_capstyle='round')
    plt.plot(steps, val_smooth, label="Validation Accuracy", color='green', linewidth=1, solid_capstyle='round')

    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)

    ax = plt.gca()
    ax.set_xscale('log')  # Use logarithmic scale for x-axis

    # Search for grokking point:
    stability_window = 10
    jump_threshold = 5.0  # percentage points increase to consider as grokking

    grokking_epoch = None
    for i in range(stability_window, len(val_smooth)):
        jump = val_smooth[i] - val_smooth[i - 1]
        if jump >= jump_threshold:
            grokking_epoch = i
            plt.axvline(x=grokking_epoch, color='yellow', linestyle=':',
                        label=f'Grokking Point (Step {grokking_epoch})')
            break

    if grokking_epoch is None:
        grokking_epoch = len(val_accuracies)  # fallback to end
        plt.plot(grokking_epoch, label='No Grokking', color='black', linestyle=':', linewidth=3)

    plt.legend()

    plt.savefig('Plots/grokking_mod_add.png')
    plt.show()
    plt.close()
