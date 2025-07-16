import numpy as np
import matplotlib.pyplot as plt

def plot_grokking(train_accuracies, val_accuracies):
    """
    Plot the training accuracy and validation accuracy on the same graph to show Grokking.

    Args:
    - train_accuracies (list): List of training accuracy values recorded per epoch.
    - val_accuracies (list): List of validation accuracy values recorded per epoch.
    - save_path (str, optional): Path to save the plot. If None, the plot is displayed but not saved.

    Returns:
    - None
    """
    epochs = range(1, len(train_accuracies) + 1)

    # Create a figure and axis for plotting
    plt.figure(figsize=(10, 6))

    # Plot both training and validation accuracy
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='blue', linestyle='.')
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='orange', linestyle='.')

    # Labeling the plot
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy (%)')

    # Find the grokking point dynamically:
    stability_window = 10  # Number of epochs to consider as "stable"
    threshold = 0.05  # Percentage increase considered significant for grokking

    stability_window = 10  # or whatever number of epochs to ignore at start
    jump_threshold = 10.0  # percentage points increase to consider as grokking

    grokking_epoch = None
    for i in range(stability_window, len(val_accuracies)):
        # Calculate the jump in validation accuracy compared to previous epoch
        jump = val_accuracies[i] - val_accuracies[i - 1]
        if jump >= jump_threshold:
            grokking_epoch = i
            break

    if grokking_epoch is None:
        grokking_epoch = len(val_accuracies)  # fallback to end

    if grokking_epoch is not None:
        plt.axvline(x=grokking_epoch, color='red', linestyle=':', label=f'Grokking Point (Epoch {grokking_epoch})')
    else:
        plt.plot([], [], label='No Grokking Point', color='red', linestyle=':')

    # Show the legend
    plt.legend()

    plt.grid()

    # Display or save the plot
    plt.savefig('Plots/grokking_xor.png')
    plt.show()

    # Close the plot to avoid memory issues during training
    plt.close()
