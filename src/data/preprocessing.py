import numpy as np
from sklearn.model_selection import train_test_split


def normalization(source_data, target_data):
    """
    Normalize source and target data separately using min-max normalization,
    following the same approach as the WGDT project.

    Args:
        source_data: Source domain hyperspectral image (H×W×C)
        target_data: Target domain hyperspectral image (H×W×C)

    Returns:
        Tuple of normalized source and target data, each in range [0,1]
    """
    # Convert to float32
    source_data = source_data.astype(np.float32)
    target_data = target_data.astype(np.float32)

    # Normalize source data independently
    source_min = np.min(source_data)
    source_max = np.max(source_data)
    source_norm = (source_data - source_min) / (source_max - source_min)

    # Normalize target data independently
    target_min = np.min(target_data)
    target_max = np.max(target_data)
    target_norm = (target_data - target_min) / (target_max - target_min)

    print("Project-style normalization complete. Data statistics:")
    print(f"Source: min={source_norm.min():.4f}, max={source_norm.max():.4f}, mean={source_norm.mean():.4f}")
    print(f"Target: min={target_norm.min():.4f}, max={target_norm.max():.4f}, mean={target_norm.mean():.4f}")

    return source_norm, target_norm


def remap_houston2013(labels):
    """Map Houston 2013 labels to new class scheme with only 7 known classes"""
    new_labels = np.zeros_like(labels, dtype=np.uint8)
    new_labels[labels == 0] = 0  # Background
    new_labels[labels == 1] = 1  # Grass healthy
    new_labels[labels == 2] = 2  # Grass stressed
    new_labels[labels == 4] = 3  # Trees
    new_labels[labels == 6] = 4  # Water
    new_labels[labels == 7] = 5  # Residential
    new_labels[labels == 8] = 6  # Commercial → Non-residential
    new_labels[labels == 9] = 7  # Road

    unknown_classes = [3, 5, 10, 11, 12, 13, 14,
                       15]  # Synthetic grass, Soil, Highway, Railway, Parking Lots, Tennis Court, Running Track
    for cls in unknown_classes:
        new_labels[labels == cls] = 8  # Unknown

    return new_labels


def remap_houston2018(labels):
    """Map Houston 2018 labels to new class scheme with only 7 known classes"""
    new_labels = np.zeros_like(labels, dtype=np.uint8)
    new_labels[labels == 0] = 0  # Background
    new_labels[labels == 1] = 1  # Grass healthy
    new_labels[labels == 2] = 2  # Grass stressed
    new_labels[np.isin(labels, [4, 5])] = 3  # Evergreen/Deciduous → Trees
    new_labels[labels == 7] = 4  # Water
    new_labels[labels == 8] = 5  # Residential buildings
    new_labels[labels == 9] = 6  # Non-residential buildings
    new_labels[labels == 10] = 7  # Roads

    # All other classes are unknown
    unknown_classes = [3, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                       20]  # Artificial turf, Bare earth, Sidewalks, Crosswalks,
    # Major thoroughfares, Highways, Railways, Paved/Unpaved parking lots,
    # Cars, Trains, Stadium seats
    for cls in unknown_classes:
        new_labels[labels == cls] = 8  # Unknown

    return new_labels


def remap_paviaU(labels):
    new_labels = np.zeros_like(labels, dtype=np.uint8)
    new_labels[labels == 0] = 0  # Background
    new_labels[labels == 4] = 1  # trees
    new_labels[labels == 1] = 2  # Asphalt
    new_labels[labels == 8] = 3  # Bricks
    new_labels[labels == 7] = 4  # Bitumen
    new_labels[labels == 2] = 5  # Meadows
    new_labels[labels == 9] = 6  # Shadow
    new_labels[labels == 6] = 7  # Bare Soil

    unknown_classes = [3, 5]  # gravel and painted metal sheets
    for cls in unknown_classes:
        new_labels[labels == cls] = 0  # Background

    return new_labels


def remap_paviaC(labels):
    new_labels = np.zeros_like(labels, dtype=np.uint8)
    new_labels[labels == 0] = 0  # Background
    new_labels[labels == 1] = 0  # Water
    new_labels[labels == 2] = 1  # trees
    new_labels[labels == 6] = 2  # Asphalt
    new_labels[labels == 4] = 3  # Bricks
    new_labels[labels == 7] = 4  # Bitumen
    new_labels[labels == 3] = 5  # Meadows
    new_labels[labels == 9] = 6  # Shadow
    new_labels[labels == 5] = 7  # Bare Soil

    unknown_classes = [8]  # Tiles
    for cls in unknown_classes:
        new_labels[labels == cls] = 8  # Unknown

    return new_labels


def remap_ground_truth_dioni(gt_array):
    """Remap Dioni ground truth to keep 9 classes (renumbered 1-9) and move rest to background"""
    remapped_gt = np.zeros_like(gt_array)

    # Mapping from original class numbers to new sequential numbers
    class_mapping = {
        1: 1,  # Dense Urban Fabric
        2: 2,  # Mineral Extraction Sites
        3: 3,  # Non Irrigated Arable Land
        4: 4,  # Fruit Trees
        5: 5,  # Olive Groves
        6: 6,  # Coniferous Forest
        8: 7,  # Sparse Sclerophyllous Vegetation
        10: 8,  # Rocks and Sand
        12: 9  # Coastal Water
    }

    # Apply mapping
    for original_cls, new_cls in class_mapping.items():
        remapped_gt[gt_array == original_cls] = new_cls

    # Classes 7, 9, 11 and any others remain as 0 (background)
    return remapped_gt


def remap_ground_truth_loukia(gt_array):
    """Remap Loukia ground truth to keep 9 classes (renumbered 1-9) and combine rest to Unknown (class 10)"""
    remapped_gt = np.zeros_like(gt_array)

    # Mapping from original class numbers to new sequential numbers
    class_mapping = {
        1: 1,  # Dense Urban Fabric
        2: 2,  # Mineral Extraction Sites
        3: 3,  # Non Irrigated Arable Land
        4: 4,  # Fruit Trees
        5: 5,  # Olive Groves
        6: 6,  # Coniferous Forest
        8: 7,  # Sparse Sclerophyllous Vegetation
        10: 8,  # Rocks and Sand
        12: 9  # Coastal Water
    }

    # Apply mapping for the 9 main classes
    for original_cls, new_cls in class_mapping.items():
        remapped_gt[gt_array == original_cls] = new_cls

    # Classes 7, 9, 11 become Unknown (class 10)
    classes_to_unknown = [7, 9, 11]
    for cls in classes_to_unknown:
        remapped_gt[gt_array == cls] = 10

    return remapped_gt


def create_stratified_split_pavia_u(gt_labels, test_size=0.2, random_state=42):
    """
    Create stratified train/validation split for Pavia University
    """
    from src.config.config import new_classesPavia

    # Get all labeled pixels (classes 1-7, excluding background 0)
    labeled_mask = (gt_labels >= 1) & (gt_labels <= 7)
    labeled_indices = np.where(labeled_mask)

    # Extract coordinates and labels
    coords = list(zip(labeled_indices[0], labeled_indices[1]))
    labels = gt_labels[labeled_indices]

    # Stratified split
    train_coords, val_coords, train_labels, val_labels = train_test_split(
        coords, labels, test_size=test_size, random_state=random_state,
        stratify=labels
    )

    # Create train and validation masks
    train_gt = np.zeros_like(gt_labels)
    val_gt = np.zeros_like(gt_labels)

    for (i, j), label in zip(train_coords, train_labels):
        train_gt[i, j] = label

    for (i, j), label in zip(val_coords, val_labels):
        val_gt[i, j] = label

    print(f"Training pixels: {len(train_coords)}")
    print(f"Validation pixels: {len(val_coords)}")

    # Print class distribution
    print("\nTraining class distribution:")
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    for cls, count in zip(unique_train, counts_train):
        print(f"Class {cls} ({new_classesPavia[cls]}): {count} samples")

    print("\nValidation class distribution:")
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    for cls, count in zip(unique_val, counts_val):
        print(f"Class {cls} ({new_classesPavia[cls]}): {count} samples")

    return train_gt, val_gt


def create_stratified_split_dioni(gt_labels, test_size=0.2, random_state=42):
    """
    Create stratified train/validation split for Dioni
    """
    from src.config.config import new_classesDioni

    # Get all labeled pixels (classes 1-9, excluding background 0)
    labeled_mask = (gt_labels >= 1) & (gt_labels <= 9)
    labeled_indices = np.where(labeled_mask)

    # Extract coordinates and labels
    coords = list(zip(labeled_indices[0], labeled_indices[1]))
    labels = gt_labels[labeled_indices]

    # Stratified split
    train_coords, val_coords, train_labels, val_labels = train_test_split(
        coords, labels, test_size=test_size, random_state=random_state,
        stratify=labels
    )

    # Create train and validation masks
    train_gt = np.zeros_like(gt_labels)
    val_gt = np.zeros_like(gt_labels)

    for (i, j), label in zip(train_coords, train_labels):
        train_gt[i, j] = label

    for (i, j), label in zip(val_coords, val_labels):
        val_gt[i, j] = label

    print(f"Training pixels: {len(train_coords)}")
    print(f"Validation pixels: {len(val_coords)}")

    # Print class distribution
    print("\nTraining class distribution:")
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    for cls, count in zip(unique_train, counts_train):
        print(f"Class {cls} ({new_classesDioni[cls]}): {count} samples")

    print("\nValidation class distribution:")
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    for cls, count in zip(unique_val, counts_val):
        print(f"Class {cls} ({new_classesDioni[cls]}): {count} samples")

    return train_gt, val_gt