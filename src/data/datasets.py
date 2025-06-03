import random
import numpy as np
import torch
from torch.utils.data import Dataset


class HSIDataset(Dataset):
    """Dataset class for hyperspectral images with improved augmentation"""

    def __init__(self, hsi_data, labels, patch_size=7, augment=False):
        self.hsi_data = hsi_data
        self.labels = labels
        self.patch_size = patch_size
        self.augment = augment
        self.pad_width = patch_size // 2

        # Improved padding with reflection
        self.hsi_padded = np.pad(
            hsi_data,
            ((self.pad_width, self.pad_width),
             (self.pad_width, self.pad_width),
             (0, 0)),
            mode='reflect'
        )

        # Extract foreground pixels, excluding unknown class by default
        self.indices = [(i, j) for i in range(self.labels.shape[0])
                        for j in range(self.labels.shape[1])
                        if 1 <= self.labels[i, j] <= 7]

        # Store class-specific indices for boundary sampling
        self.class_indices = {cls: [] for cls in range(1, 8)}  # Classes 1-7 only

        for idx, (i, j) in enumerate(self.indices):
            label = self.labels[i, j]
            if 1 <= label <= 7:
                self.class_indices[label].append(idx)

        print(f"Dataset initialized with {len(self.indices)} samples")
        # Print class distribution
        from src.config.config import new_classes
        for cls in range(1, 8):
            print(f"Class {cls} ({new_classes[cls]}): {len(self.class_indices[cls])} samples")

    def __len__(self):
        return len(self.indices)

    def augment_data(self, patch):
        """Improved data augmentation with multiple strategies"""
        if random.random() > 0.3:  # More frequent application
            noise_level = random.uniform(0.03, 0.08)  # Stronger noise
            noise = np.random.normal(0, noise_level, patch.shape)
            patch = patch + noise

        # Add channel-specific perturbations to simulate sensor differences
        if random.random() > 0.4:
            band_factors = np.random.uniform(0.8, 1.2, patch.shape[2])
            for b in range(patch.shape[2]):
                patch[:, :, b] *= band_factors[b]

        if random.random() > 0.5:
            # Flip horizontally
            patch = np.flip(patch, axis=0)

        if random.random() > 0.5:
            # Flip vertically
            patch = np.flip(patch, axis=1)

        if random.random() > 0.75:
            # Random band dropout (simulates sensor errors)
            num_bands = patch.shape[2]
            drop_bands = random.sample(range(num_bands), k=int(num_bands * 0.1))
            for band in drop_bands:
                patch[:, :, band] = 0

        # Ensure valid range
        patch = np.clip(patch, -3.0, 3.0)  # Allow standardized values

        return patch

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        i_pad, j_pad = i + self.pad_width, j + self.pad_width

        # Extract patch
        patch = self.hsi_padded[
                i_pad - self.pad_width:i_pad + self.pad_width + 1,
                j_pad - self.pad_width:j_pad + self.pad_width + 1,
                :
                ].copy()

        # Apply augmentation if needed
        if self.augment:
            patch = self.augment_data(patch)

        # Map labels: 1-7 -> 0-6
        label = self.labels[i, j]
        label = label - 1  # Shift 1-7 to 0-6

        # Convert to torch tensors
        patch = torch.FloatTensor(patch).permute(2, 0, 1)  # CxHxW format
        return patch, label

    def get_class_patch(self, class_idx):
        """Get a random patch from a specific class"""
        if len(self.class_indices[class_idx]) == 0:
            return None, None

        # Select a random sample from this class
        idx = random.choice(self.class_indices[class_idx])
        i, j = self.indices[idx]
        i_pad, j_pad = i + self.pad_width, j + self.pad_width

        # Extract patch
        patch = self.hsi_padded[
                i_pad - self.pad_width:i_pad + self.pad_width + 1,
                j_pad - self.pad_width:j_pad + self.pad_width + 1,
                :
                ].copy()

        if self.augment:
            patch = self.augment_data(patch)

        # Convert to tensor and return
        patch = torch.FloatTensor(patch).permute(2, 0, 1)
        return patch, class_idx - 1  # Return 0-indexed class


class HSITestDataset(Dataset):
    """Dataset class for evaluating on target domain"""

    def __init__(self, hsi_data, labels, patch_size=7):
        self.hsi_data = hsi_data
        self.labels = labels
        self.patch_size = patch_size
        self.pad_width = patch_size // 2

        # Pad the data
        self.hsi_padded = np.pad(
            hsi_data,
            ((self.pad_width, self.pad_width),
             (self.pad_width, self.pad_width),
             (0, 0)),
            mode='reflect'
        )

        # Include both known (1-7) and unknown (8) classes
        self.indices = [(i, j) for i in range(self.labels.shape[0])
                        for j in range(self.labels.shape[1])
                        if self.labels[i, j] >= 1]

        print(f"Test dataset initialized with {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        i_pad, j_pad = i + self.pad_width, j + self.pad_width

        # Extract patch
        patch = self.hsi_padded[
                i_pad - self.pad_width:i_pad + self.pad_width + 1,
                j_pad - self.pad_width:j_pad + self.pad_width + 1,
                :
                ].copy()

        # Map labels: 1-7 -> 0-6 (known), 8 -> 7 (unknown)
        label = self.labels[i, j]
        label = label - 1  # Shift 1-8 to 0-7

        patch = torch.FloatTensor(patch).permute(2, 0, 1)
        return patch, label


class HSIDatasetDioni(Dataset):
    """Dataset class for hyperspectral images with improved augmentation - for Dioni/Loukia (9 classes)"""

    def __init__(self, hsi_data, labels, patch_size=7, augment=False):
        self.hsi_data = hsi_data
        self.labels = labels
        self.patch_size = patch_size
        self.augment = augment
        self.pad_width = patch_size // 2

        # Improved padding with reflection
        self.hsi_padded = np.pad(
            hsi_data,
            ((self.pad_width, self.pad_width),
             (self.pad_width, self.pad_width),
             (0, 0)),
            mode='reflect'
        )

        # Extract foreground pixels, excluding unknown class by default (1-9 for Dioni)
        self.indices = [(i, j) for i in range(self.labels.shape[0])
                        for j in range(self.labels.shape[1])
                        if 1 <= self.labels[i, j] <= 9]

        # Store class-specific indices for boundary sampling
        self.class_indices = {cls: [] for cls in range(1, 10)}  # Classes 1-9 only

        for idx, (i, j) in enumerate(self.indices):
            label = self.labels[i, j]
            if 1 <= label <= 9:
                self.class_indices[label].append(idx)

        print(f"Dataset initialized with {len(self.indices)} samples")
        # Print class distribution
        from src.config.config import new_classesDioni
        for cls in range(1, 10):
            print(f"Class {cls} ({new_classesDioni[cls]}): {len(self.class_indices[cls])} samples")

    def __len__(self):
        return len(self.indices)

    def augment_data(self, patch):
        """Improved data augmentation with multiple strategies"""
        if random.random() > 0.3:  # More frequent application
            noise_level = random.uniform(0.03, 0.08)  # Stronger noise
            noise = np.random.normal(0, noise_level, patch.shape)
            patch = patch + noise

        # Add channel-specific perturbations to simulate sensor differences
        if random.random() > 0.4:
            band_factors = np.random.uniform(0.8, 1.2, patch.shape[2])
            for b in range(patch.shape[2]):
                patch[:, :, b] *= band_factors[b]

        if random.random() > 0.5:
            # Flip horizontally
            patch = np.flip(patch, axis=0)

        if random.random() > 0.5:
            # Flip vertically
            patch = np.flip(patch, axis=1)

        if random.random() > 0.75:
            # Random band dropout (simulates sensor errors)
            num_bands = patch.shape[2]
            drop_bands = random.sample(range(num_bands), k=int(num_bands * 0.1))
            for band in drop_bands:
                patch[:, :, band] = 0

        # Ensure valid range
        patch = np.clip(patch, -3.0, 3.0)  # Allow standardized values

        return patch

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        i_pad, j_pad = i + self.pad_width, j + self.pad_width

        # Extract patch
        patch = self.hsi_padded[
                i_pad - self.pad_width:i_pad + self.pad_width + 1,
                j_pad - self.pad_width:j_pad + self.pad_width + 1,
                :
                ].copy()

        # Apply augmentation if needed
        if self.augment:
            patch = self.augment_data(patch)

        # Map labels: 1-9 -> 0-8
        label = self.labels[i, j]
        label = label - 1  # Shift 1-9 to 0-8

        # Convert to torch tensors
        patch = torch.FloatTensor(patch).permute(2, 0, 1)  # CxHxW format
        return patch, label

    def get_class_patch(self, class_idx):
        """Get a random patch from a specific class"""
        if len(self.class_indices[class_idx]) == 0:
            return None, None

        # Select a random sample from this class
        idx = random.choice(self.class_indices[class_idx])
        i, j = self.indices[idx]
        i_pad, j_pad = i + self.pad_width, j + self.pad_width

        # Extract patch
        patch = self.hsi_padded[
                i_pad - self.pad_width:i_pad + self.pad_width + 1,
                j_pad - self.pad_width:j_pad + self.pad_width + 1,
                :
                ].copy()

        if self.augment:
            patch = self.augment_data(patch)

        # Convert to tensor and return
        patch = torch.FloatTensor(patch).permute(2, 0, 1)
        return patch, class_idx - 1  # Return 0-indexed class


class HSITestDatasetDioni(Dataset):
    """Dataset class for evaluating on target domain - for Dioni/Loukia"""

    def __init__(self, hsi_data, labels, patch_size=7):
        self.hsi_data = hsi_data
        self.labels = labels
        self.patch_size = patch_size
        self.pad_width = patch_size // 2

        # Pad the data
        self.hsi_padded = np.pad(
            hsi_data,
            ((self.pad_width, self.pad_width),
             (self.pad_width, self.pad_width),
             (0, 0)),
            mode='reflect'
        )

        # Include both known (1-9) and unknown (10) classes
        self.indices = [(i, j) for i in range(self.labels.shape[0])
                        for j in range(self.labels.shape[1])
                        if self.labels[i, j] >= 1]

        print(f"Test dataset initialized with {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        i_pad, j_pad = i + self.pad_width, j + self.pad_width

        # Extract patch
        patch = self.hsi_padded[
                i_pad - self.pad_width:i_pad + self.pad_width + 1,
                j_pad - self.pad_width:j_pad + self.pad_width + 1,
                :
                ].copy()

        # Map labels: 1-9 -> 0-8 (known), 10 -> 9 (unknown)
        label = self.labels[i, j]
        if label <= 9:
            label = label - 1  # Shift 1-9 to 0-8
        else:
            label = 9  # Map 10 to 9 (unknown)

        patch = torch.FloatTensor(patch).permute(2, 0, 1)
        return patch, label