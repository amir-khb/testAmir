import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_hsi(hsi_data, mask, title, bands=[10, 20, 30], colors=None, class_names=None):
    """
    Visualize hyperspectral image with false color composite
    Args:
        hsi_data: normalized hyperspectral image
        mask: ground truth mask
        title: title for the plot
        bands: which bands to use for RGB visualization (default: [10, 20, 30])
        colors: color list for classes
        class_names: dictionary of class names
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Create a false color composite
    rgb = np.zeros((hsi_data.shape[0], hsi_data.shape[1], 3))
    rgb[:, :, 0] = hsi_data[:, :, bands[0]]  # Red channel
    rgb[:, :, 1] = hsi_data[:, :, bands[1]]  # Green channel
    rgb[:, :, 2] = hsi_data[:, :, bands[2]]  # Blue channel

    # Normalize for visualization
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min)

    # Plot the false color composite
    axs[0].imshow(rgb)
    axs[0].set_title(f"{title} - False Color Composite (Bands {bands})")
    axs[0].axis('off')

    # Create colormap if not provided
    if colors is None:
        from src.config.config import colors
    cmap = ListedColormap(colors)

    # Plot the mask with the custom colormap
    masked_img = axs[1].imshow(mask, cmap=cmap, vmin=0, vmax=len(colors) - 1)
    axs[1].set_title(f"{title} - Ground Truth")
    axs[1].axis('off')

    # Add colorbar for the mask
    if class_names is not None:
        cbar = plt.colorbar(masked_img, ax=axs[1], ticks=range(len(class_names)))
        cbar.set_ticklabels([class_names[i] for i in range(len(class_names))])

    plt.tight_layout()
    return fig


def plot_training_history(history):
    """Plot training history metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training and validation loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss During Training')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Training and validation accuracy
    axes[0, 1].plot(history['train_acc'], label='Training Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy During Training')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Test metrics (if available)
    if 'test_acc' in history and len(history['test_acc']) > 0:
        test_epochs = range(10, len(history['test_acc']) * 10 + 1, 10)
        axes[1, 0].plot(test_epochs, history['test_acc'], 'ro-', label='Known Class Accuracy')
        axes[1, 0].plot(test_epochs, history['test_rej'], 'bo-', label='Unknown Rejection')
        axes[1, 0].set_title('Test Performance')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # HOS Score
        axes[1, 1].plot(test_epochs, history['hos'], 'go-', label='HOS Score')
        axes[1, 1].set_title('Harmonic Open Set (HOS) Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('HOS Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'No test metrics available',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'No HOS scores available',
                        ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks and label them
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    return fig


def print_pixel_distribution(labels, dataset_name, class_names):
    """Print pixel distribution for a dataset"""
    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = labels.size

    print(f"\n{dataset_name} - Pixel Distribution:")
    print("-" * 50)
    print(f"{'Class':<12} {'Name':<15} {'Pixels':<10} {'Percentage':<10}")
    print("-" * 50)

    for class_id in range(len(class_names)):
        if class_id in unique:
            idx = np.where(unique == class_id)[0][0]
            pixel_count = counts[idx]
            percentage = (pixel_count / total_pixels) * 100
        else:
            pixel_count = 0
            percentage = 0.0

        print(f"{class_id:<12} {class_names[class_id]:<15} {pixel_count:<10} {percentage:.2f}%")

    print("-" * 50)
    print(f"{'Total':<12} {'':<15} {total_pixels:<10} 100.00%")