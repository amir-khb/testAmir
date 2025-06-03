import os
import torch
import numpy as np
import scipy.io as sio


def debug_mat_files():
    """
    Debug function to check the structure of your .mat files
    """
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, "data", "DioniandLoukia")

    print(f"Looking for data in: {data_path}")

    files_to_check = [
        "Dioni.mat",
        "Dioni_gt_out68.mat",
        "Loukia.mat",
        "Loukia_gt_out68.mat"
    ]

    for filename in files_to_check:
        filepath = os.path.join(data_path, filename)
        print(f"\n=== Checking {filename} ===")

        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue

        try:
            data = sio.loadmat(filepath)
            print(f"✅ File loaded successfully")
            print(f"Keys in file: {[k for k in data.keys() if not k.startswith('__')]}")

            for key, value in data.items():
                if not key.startswith('__'):
                    print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
                    if len(value.shape) == 3:
                        print(
                            f"    → Likely hyperspectral data (H={value.shape[0]}, W={value.shape[1]}, C={value.shape[2]})")
                    elif len(value.shape) == 2:
                        print(f"    → Likely ground truth (H={value.shape[0]}, W={value.shape[1]})")
                        unique_vals = np.unique(value)
                        print(f"    → Unique values: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")

        except Exception as e:
            print(f"❌ Error loading file: {e}")

def save_model_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model checkpoint saved to {path}")


def load_model_checkpoint(model, optimizer, path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Model checkpoint loaded from {path}")
    print(f"Resuming from epoch {epoch} with loss {loss:.4f}")

    return epoch, loss


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'outputs',
        'outputs/models',
        'outputs/figures',
        'outputs/logs'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/exists: {directory}")