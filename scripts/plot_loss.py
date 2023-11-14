#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(training_loss, validation_loss, save_path="loss_plot.png"):
    """
    Plot training and validation loss and save the figure.

    Parameters:
    - training_loss (list or numpy array): Vector of training loss values.
    - validation_loss (list or numpy array): Vector of validation loss values.
    - save_path (str): Path to save the figure (default is "loss_plot.png").
    """
    epochs = range(1, len(training_loss) + 1)

    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.suptitle(f'Average last 10 val {np.mean(validation_loss[-10:])}. Average last 10 train {np.mean(training_loss[-10:])}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the figure
    plt.savefig(save_path)

    # Show the plot (optional)
    plt.show()
