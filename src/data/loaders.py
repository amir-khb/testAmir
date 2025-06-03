import os
import numpy as np
import scipy.io as sio
import rasterio
from scipy.ndimage import zoom
from .preprocessing import (
    normalization, remap_houston2013, remap_houston2018,
    remap_paviaU, remap_paviaC, remap_ground_truth_dioni,
    remap_ground_truth_loukia, create_stratified_split_pavia_u,
    create_stratified_split_dioni
)


# Get the project root directory
def get_project_root():
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels: src/data -> src -> project_root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root


PROJECT_ROOT = get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def load_mat_file(filepath):
    """Load .mat file and return the data"""
    try:
        data = sio.loadmat(filepath)
        # Remove metadata keys that start with '__'
        data_keys = [key for key in data.keys() if not key.startswith('__')]
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def get_hyperspectral_array(hs_data):
    """Extract hyperspectral array from loaded data"""
    if hs_data is None:
        return None

    for key, value in hs_data.items():
        if not key.startswith('__') and isinstance(value, np.ndarray):
            # Return the array with most dimensions (likely the hyperspectral cube)
            if len(value.shape) == 3:
                return value
    return None


def get_ground_truth_array(gt_data):
    """Extract ground truth array from loaded data"""
    if gt_data is None:
        return None

    for key, value in gt_data.items():
        if not key.startswith('__') and isinstance(value, np.ndarray):
            # Return the 2D array (ground truth)
            if len(value.shape) == 2:
                return value
    return None


def load_dioni_loukia_datasets():
    """Load Dioni and Loukia datasets"""
    data_path = os.path.join(DATA_DIR, "DioniandLoukia")

    print("Loading Dioni and Loukia datasets...")
    print(f"Looking for data in: {data_path}")

    # Load Dioni dataset
    try:
        dioni_hs_path = os.path.join(data_path, "Dioni.mat")
        dioni_gt_path = os.path.join(data_path, "Dioni_gt_out68.mat")

        print(f"Loading Dioni HS from: {dioni_hs_path}")
        print(f"Loading Dioni GT from: {dioni_gt_path}")

        dioni_hs_data = load_mat_file(dioni_hs_path)
        dioni_gt_data = load_mat_file(dioni_gt_path)

        hs_dioni = get_hyperspectral_array(dioni_hs_data)
        gt_dioni = get_ground_truth_array(dioni_gt_data)

        if hs_dioni is None or gt_dioni is None:
            raise ValueError("Could not extract Dioni data from .mat files")

        print(f"Dioni loaded - HS shape: {hs_dioni.shape}, GT shape: {gt_dioni.shape}")

    except Exception as e:
        print(f"Error loading Dioni dataset: {e}")
        raise

    # Load Loukia dataset
    try:
        loukia_hs_path = os.path.join(data_path, "Loukia.mat")
        loukia_gt_path = os.path.join(data_path, "Loukia_gt_out68.mat")

        print(f"Loading Loukia HS from: {loukia_hs_path}")
        print(f"Loading Loukia GT from: {loukia_gt_path}")

        loukia_hs_data = load_mat_file(loukia_hs_path)
        loukia_gt_data = load_mat_file(loukia_gt_path)

        hs_loukia = get_hyperspectral_array(loukia_hs_data)
        gt_loukia = get_ground_truth_array(loukia_gt_data)

        if hs_loukia is None or gt_loukia is None:
            raise ValueError("Could not extract Loukia data from .mat files")

        print(f"Loukia loaded - HS shape: {hs_loukia.shape}, GT shape: {gt_loukia.shape}")

    except Exception as e:
        print(f"Error loading Loukia dataset: {e}")
        raise

    return gt_dioni, hs_dioni, gt_loukia, hs_loukia


def load_pavia_center():
    """Load Pavia Center dataset"""
    data_path = os.path.join(DATA_DIR, "pavia_C")

    print(f"Loading Pavia Center from: {data_path}")

    gt_path = os.path.join(data_path, "Pavia_gt.mat")
    hs_path = os.path.join(data_path, "Pavia.mat")

    print(f"GT file: {gt_path}")
    print(f"HS file: {hs_path}")

    # Check if files exist
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not os.path.exists(hs_path):
        raise FileNotFoundError(f"Hyperspectral file not found: {hs_path}")

    # Load the ground truth and hyperspectral data
    gt_data = sio.loadmat(gt_path)
    hs_data = sio.loadmat(hs_path)

    # Extract the arrays (key names may vary, so we'll find them)
    gt_keys = [k for k in gt_data.keys() if not k.startswith('__')]
    hs_keys = [k for k in hs_data.keys() if not k.startswith('__')]

    print(f"Ground truth keys: {gt_keys}")
    print(f"Hyperspectral data keys: {hs_keys}")

    # Usually the ground truth is stored in a key like 'pavia_gt' or similar
    gt = gt_data[gt_keys[0]]
    hs = hs_data[hs_keys[0]]

    return gt, hs


def load_pavia_university():
    """Load Pavia University dataset"""
    data_path = os.path.join(DATA_DIR, "pavia_U")

    print(f"Loading Pavia University from: {data_path}")

    gt_path = os.path.join(data_path, "PaviaU_gt.mat")
    hs_path = os.path.join(data_path, "PaviaU.mat")

    print(f"GT file: {gt_path}")
    print(f"HS file: {hs_path}")

    # Check if files exist
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not os.path.exists(hs_path):
        raise FileNotFoundError(f"Hyperspectral file not found: {hs_path}")

    # Load the ground truth and hyperspectral data
    gt_data = sio.loadmat(gt_path)
    hs_data = sio.loadmat(hs_path)

    # Extract the arrays
    gt_keys = [k for k in gt_data.keys() if not k.startswith('__')]
    hs_keys = [k for k in hs_data.keys() if not k.startswith('__')]

    print(f"Ground truth keys: {gt_keys}")
    print(f"Hyperspectral data keys: {hs_keys}")

    gt = gt_data[gt_keys[0]]
    hs = hs_data[hs_keys[0]]
    hs = hs[:, :, :102]

    return gt, hs


def load_and_preprocess_data():
    """Load and preprocess Houston data"""
    print("Loading and preprocessing Houston data...")

    houston2013_path = os.path.join(DATA_DIR, "Houston2013")
    houston2018_path = os.path.join(DATA_DIR, "Houston2018")

    print(f"Houston 2013 path: {houston2013_path}")
    print(f"Houston 2018 path: {houston2018_path}")

    # Load Houston 2013
    try:
        hs_lr_path = os.path.join(houston2013_path, "data_HS_LR.mat")
        train_gt_path = os.path.join(houston2013_path, "2013_IEEE_GRSS_DF_Contest_Samples_TR.tif")
        val_gt_path = os.path.join(houston2013_path, "2013_IEEE_GRSS_DF_Contest_Samples_VA.tif")

        print(f"Loading Houston 2013 HS from: {hs_lr_path}")

        hs_lr_data = sio.loadmat(hs_lr_path)
        hs_image_2013 = hs_lr_data['data_HS_LR']

        # Extract common bands (48 bands)
        common_band_indices = [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 39, 42, 45,
                               48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87,
                               90, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124,
                               127, 130, 133, 136, 139, 142]
        hs_image_2013_common = hs_image_2013[:, :, common_band_indices]

        # Load 2013 ground truth
        with rasterio.open(train_gt_path) as src:
            train_labels_2013 = src.read(1)
        with rasterio.open(val_gt_path) as src:
            val_labels_2013 = src.read(1)

        print(f"Houston 2013 loaded successfully")

    except Exception as e:
        print(f"Error loading Houston 2013 data: {e}")
        raise

    # Load Houston 2018
    try:
        hsi_path = os.path.join(houston2018_path, "HSI", "2018_IEEE_GRSS_DFC_HSI_TR")
        gt_path = os.path.join(houston2018_path, "GT", "2018_IEEE_GRSS_DFC_GT_TR.tif")

        print(f"Loading Houston 2018 HSI from: {hsi_path}")
        print(f"Loading Houston 2018 GT from: {gt_path}")

        with rasterio.open(hsi_path) as src:
            hs_image_2018 = src.read()[:48, :, :]
            hs_image_2018 = np.transpose(hs_image_2018, (1, 2, 0))
        with rasterio.open(gt_path) as src:
            labels_2018 = src.read(1)
            # Downsample to match resolution
            labels_2018_downsampled = zoom(labels_2018, 0.5, order=0)

        print(f"Houston 2018 loaded successfully")

    except Exception as e:
        print(f"Error loading Houston 2018 data: {e}")
        raise

    # Apply domain-agnostic normalization instead of the joint normalization
    hs_image_2013_norm, hs_image_2018_norm = normalization(
        hs_image_2013_common, hs_image_2018)

    # Verify normalization worked properly
    print("Normalization complete. Data statistics:")
    print(
        f"Houston 2013: min={hs_image_2013_norm.min():.4f}, max={hs_image_2013_norm.max():.4f}, mean={hs_image_2013_norm.mean():.4f}")
    print(
        f"Houston 2018: min={hs_image_2018_norm.min():.4f}, max={hs_image_2018_norm.max():.4f}, mean={hs_image_2018_norm.mean():.4f}")

    # Remap class labels
    print("Remapping class labels...")
    train_labels_2013_new = remap_houston2013(train_labels_2013)
    val_labels_2013_new = remap_houston2013(val_labels_2013)
    labels_2018_new = remap_houston2018(labels_2018)
    labels_2018_downsampled_new = remap_houston2018(labels_2018_downsampled)

    # Combine training and validation for HU13 ground truth
    gt_2013 = np.zeros_like(train_labels_2013_new)
    gt_2013[train_labels_2013_new > 0] = train_labels_2013_new[train_labels_2013_new > 0]
    gt_2013[val_labels_2013_new > 0] = val_labels_2013_new[val_labels_2013_new > 0]

    # Count classes
    from config.config import new_classes
    source_classes = np.unique(gt_2013, return_counts=True)
    target_classes = np.unique(labels_2018_downsampled_new, return_counts=True)

    print("Class distribution in source domain (Houston 2013):")
    for cls, count in zip(source_classes[0], source_classes[1]):
        if cls > 0:  # Skip background
            print(f"Class {cls} ({new_classes[cls]}): {count} samples")

    print("\nClass distribution in target domain (Houston 2018):")
    for cls, count in zip(target_classes[0], target_classes[1]):
        if cls > 0:  # Skip background
            print(f"Class {cls} ({new_classes[cls]}): {count} samples")

    return (hs_image_2013_norm, gt_2013,
            hs_image_2018_norm, labels_2018_downsampled_new,
            train_labels_2013_new, val_labels_2013_new)


def load_and_preprocess_pavia_data():
    """
    Load and preprocess Pavia University (source) and Pavia Center (target) data
    """
    print("Loading and preprocessing Pavia data...")

    # Load Pavia University and Center
    gt_pavia_u, hs_pavia_u = load_pavia_university()
    gt_pavia_c, hs_pavia_c = load_pavia_center()

    print(f"Pavia University - GT shape: {gt_pavia_u.shape}, HS shape: {hs_pavia_u.shape}")
    print(f"Pavia Center - GT shape: {gt_pavia_c.shape}, HS shape: {hs_pavia_c.shape}")

    # Apply remapping
    gt_pavia_u_remapped = remap_paviaU(gt_pavia_u)
    gt_pavia_c_remapped = remap_paviaC(gt_pavia_c)

    # Apply normalization
    hs_pavia_u_norm, hs_pavia_c_norm = normalization(hs_pavia_u, hs_pavia_c)

    # Create stratified split for Pavia University
    train_gt_pavia_u, val_gt_pavia_u = create_stratified_split_pavia_u(
        gt_pavia_u_remapped, test_size=0.2, random_state=42
    )

    # Print class distributions
    from config.config import new_classesPavia
    print("\nPavia University (Source) - Training set:")
    train_classes = np.unique(train_gt_pavia_u, return_counts=True)
    for cls, count in zip(train_classes[0], train_classes[1]):
        if cls > 0:
            print(f"Class {cls} ({new_classesPavia[cls]}): {count} samples")

    print("\nPavia Center (Target) - Test set:")
    target_classes = np.unique(gt_pavia_c_remapped, return_counts=True)
    for cls, count in zip(target_classes[0], target_classes[1]):
        if cls > 0:
            print(f"Class {cls} ({new_classesPavia[cls]}): {count} samples")

    return (hs_pavia_u_norm, train_gt_pavia_u, val_gt_pavia_u,
            hs_pavia_c_norm, gt_pavia_c_remapped)


def load_and_preprocess_dioni_loukia_data():
    """
    Load and preprocess Dioni (source) and Loukia (target) data
    """
    print("Loading and preprocessing Dioni and Loukia data...")

    # Load datasets
    gt_dioni, hs_dioni, gt_loukia, hs_loukia = load_dioni_loukia_datasets()

    print(f"Dioni - GT shape: {gt_dioni.shape}, HS shape: {hs_dioni.shape}")
    print(f"Loukia - GT shape: {gt_loukia.shape}, HS shape: {hs_loukia.shape}")

    # Apply remapping
    gt_dioni_remapped = remap_ground_truth_dioni(gt_dioni)
    gt_loukia_remapped = remap_ground_truth_loukia(gt_loukia)

    # Apply normalization
    hs_dioni_norm, hs_loukia_norm = normalization(hs_dioni, hs_loukia)

    # Create stratified split for Dioni
    train_gt_dioni, val_gt_dioni = create_stratified_split_dioni(
        gt_dioni_remapped, test_size=0.2, random_state=42
    )

    # Print class distributions
    from config.config import new_classesDioni
    print("\nDioni (Source) - Training set:")
    train_classes = np.unique(train_gt_dioni, return_counts=True)
    for cls, count in zip(train_classes[0], train_classes[1]):
        if cls > 0:
            print(f"Class {cls} ({new_classesDioni[cls]}): {count} samples")

    print("\nLoukia (Target) - Test set:")
    target_classes = np.unique(gt_loukia_remapped, return_counts=True)
    for cls, count in zip(target_classes[0], target_classes[1]):
        if cls > 0:
            print(f"Class {cls} ({new_classesDioni[cls] if cls <= 9 else 'Unknown'}): {count} samples")

    return (hs_dioni_norm, train_gt_dioni, val_gt_dioni,
            hs_loukia_norm, gt_loukia_remapped)