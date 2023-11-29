#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(training_loss, validation_loss=None, save_path="loss_plot.png"):
    """
    Plot training and validation loss and save the figure.

    Parameters:
    - training_loss (list or numpy array): Vector of training loss values.
    - validation_loss (list or numpy array): Vector of validation loss values.
    - save_path (str): Path to save the figure (default is "loss_plot.png").
    """
    if validation_loss is not None:
        epochs = range(1, len(training_loss) + 1)

        plt.plot(epochs, training_loss, label='Training Loss')
        plt.plot(epochs, validation_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.suptitle(f'Min loss val {round(np.min(validation_loss), 4)}. Min loss train {round(np.min(training_loss), 4)}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    else:
        epochs = range(1, len(training_loss) + 1)

        plt.plot(epochs, training_loss, label='Training Loss')
        plt.title('Training Loss')
        plt.suptitle(f'Min loss train {round(np.min(training_loss), 4)}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    # Save the figure
    plt.savefig(save_path)

    
