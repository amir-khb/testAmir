import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = current_dir
project_root = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Now import our modules
from config.config import device, worker_init_fn, g, new_classes, new_classesDioni, new_classesPavia
from data.loaders import (
    load_and_preprocess_data,
    load_and_preprocess_pavia_data,
    load_and_preprocess_dioni_loukia_data
)
from data.datasets import HSIDataset, HSITestDataset, HSIDatasetDioni, HSITestDatasetDioni
from models.classifier import DCRN_SSUD_SIFD, HybridSSUDClassifierFixed
from training.trainer import train_one_epoch_hybrid, validate_hybrid
from training.evaluator import test_open_set_hybrid, test_open_set_hybrid_dioni
from utils.helpers import setup_directories


def train_and_evaluate_hybrid(epochs=75, batch_size=32, patch_size=7, lr=0.00001,
                              dataset_name="Houston",
                              # Fixed parameters
                              uncertainty_threshold=0.5,
                              decoupling_threshold=0.25,
                              spectral_weight=0.7):
    """
    Simplified training function with fixed parameters
    """
    print(f"Starting training with FIXED parameters on {dataset_name} dataset...")
    print(f"Fixed parameters:")
    print(f"  - Uncertainty threshold: {uncertainty_threshold}")
    print(f"  - Decoupling threshold: {decoupling_threshold}")
    print(f"  - Spectral weight: {spectral_weight}")

    # Create datasets - select the right dataset based on name
    print("Creating datasets...")

    if dataset_name.lower() == "houston":
        # Houston datasets
        (hs_image_2013_norm, gt_2013,
         hs_image_2018_norm, labels_2018_downsampled_new,
         train_labels_2013_new, val_labels_2013_new) = load_and_preprocess_data()

        source_train_dataset = HSIDataset(hs_image_2013_norm, train_labels_2013_new,
                                          patch_size=patch_size, augment=True)
        source_val_dataset = HSIDataset(hs_image_2013_norm, val_labels_2013_new,
                                        patch_size=patch_size, augment=False)
        target_test_dataset = HSITestDataset(hs_image_2018_norm, labels_2018_downsampled_new,
                                             patch_size=patch_size)
        input_channels = hs_image_2013_norm.shape[2]
    else:
        # Pavia datasets
        (hs_pavia_u_norm, train_gt_pavia_u, val_gt_pavia_u,
         hs_pavia_c_norm, gt_pavia_c_remapped) = load_and_preprocess_pavia_data()

        source_train_dataset = HSIDataset(hs_pavia_u_norm, train_gt_pavia_u,
                                          patch_size=patch_size, augment=True)
        source_val_dataset = HSIDataset(hs_pavia_u_norm, val_gt_pavia_u,
                                        patch_size=patch_size, augment=False)
        target_test_dataset = HSITestDataset(hs_pavia_c_norm, gt_pavia_c_remapped,
                                             patch_size=patch_size)
        input_channels = hs_pavia_u_norm.shape[2]

    # Create data loaders
    train_loader = DataLoader(source_train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(source_val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    test_loader = DataLoader(target_test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)

    # Create the enhanced model with SIFD
    num_classes = 7  # Known classes only
    model = DCRN_SSUD_SIFD(input_channels=input_channels, patch_size=patch_size, num_classes=num_classes).to(device)

    # Initialize classifier with FIXED parameters
    classifier = HybridSSUDClassifierFixed(
        model,
        num_classes=num_classes,
        uncertainty_threshold=uncertainty_threshold,
        decoupling_threshold=decoupling_threshold,
        spectral_weight=spectral_weight
    ).to(device)

    # Use Adam optimizer
    optimizer = optim.Adam([
        {'params': model.dcrn.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr},
        {'params': model.spectral_evidence.parameters(), 'lr': lr},
        {'params': model.spatial_evidence.parameters(), 'lr': lr},
        {'params': model.combined_evidence.parameters(), 'lr': lr},
        {'params': model.sifd.parameters(), 'lr': lr},
    ], weight_decay=1e-5)

    # Use CosineAnnealingLR for better performance
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc': [],
        'test_rej': [],
        'hos': []
    }

    # Best model tracking
    best_val_acc = 0
    best_hos = 0
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Train with unified loss function
        train_loss, train_acc = train_one_epoch_hybrid(
            model, classifier, train_loader, optimizer, epoch,
            alpha=0.3, beta=0.5, gamma=0.3, delta=0.2, clip_value=1.0)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validate
        val_loss, val_acc = validate_hybrid(model, classifier, val_loader, epoch)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Update learning rate
        scheduler.step()

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            print(f"New best model saved at epoch {epoch} with validation accuracy {val_acc:.2f}%")

    print(f"\n🎉 Training completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)

    print("\n🏁 Final evaluation with fixed parameters...")
    known_acc, unknown_rej, hos_score, confusion_mat = test_open_set_hybrid(model, classifier, test_loader)

    return {
        'model': model,
        'classifier': classifier,
        'history': history,
        'best_epoch': best_epoch,
        'final_metrics': {
            'known_acc': known_acc,
            'unknown_rej': unknown_rej,
            'hos_score': hos_score,
            'fixed_params': {
                'uncertainty_threshold': uncertainty_threshold,
                'decoupling_threshold': decoupling_threshold,
                'spectral_weight': spectral_weight
            }
        },
        'confusion_matrix': confusion_mat
    }


def train_and_evaluate_hybrid_dioni_loukia(epochs=75, batch_size=32, patch_size=7, lr=0.00001,
                                           # Fixed parameters
                                           uncertainty_threshold=0.5,
                                           decoupling_threshold=0.25,
                                           spectral_weight=0.7):
    """
    Simplified training function with fixed parameters for Dioni (source) and Loukia (target) datasets
    """
    print(f"Starting training with FIXED parameters on Dioni->Loukia...")
    print(f"Fixed parameters:")
    print(f"  - Uncertainty threshold: {uncertainty_threshold}")
    print(f"  - Decoupling threshold: {decoupling_threshold}")
    print(f"  - Spectral weight: {spectral_weight}")

    # Load data
    print("Loading datasets...")
    (hs_dioni_norm, train_gt_dioni, val_gt_dioni,
     hs_loukia_norm, gt_loukia_remapped) = load_and_preprocess_dioni_loukia_data()

    # Create datasets with 9 known classes (1-9)
    source_train_dataset = HSIDatasetDioni(hs_dioni_norm, train_gt_dioni,
                                           patch_size=patch_size, augment=True)
    source_val_dataset = HSIDatasetDioni(hs_dioni_norm, val_gt_dioni,
                                         patch_size=patch_size, augment=False)
    target_test_dataset = HSITestDatasetDioni(hs_loukia_norm, gt_loukia_remapped,
                                              patch_size=patch_size)

    input_channels = hs_dioni_norm.shape[2]  # Should be 176

    # Create data loaders
    train_loader = DataLoader(source_train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(source_val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    test_loader = DataLoader(target_test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)

    # Create the enhanced model with SIFD for 9 classes
    num_classes = 9  # Known classes only (1-9)
    model = DCRN_SSUD_SIFD(input_channels=input_channels, patch_size=patch_size, num_classes=num_classes).to(device)

    # Initialize classifier with FIXED parameters
    classifier = HybridSSUDClassifierFixed(
        model,
        num_classes=num_classes,
        uncertainty_threshold=uncertainty_threshold,
        decoupling_threshold=decoupling_threshold,
        spectral_weight=spectral_weight
    ).to(device)

    # Use Adam optimizer
    optimizer = optim.Adam([
        {'params': model.dcrn.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr},
        {'params': model.spectral_evidence.parameters(), 'lr': lr},
        {'params': model.spatial_evidence.parameters(), 'lr': lr},
        {'params': model.combined_evidence.parameters(), 'lr': lr},
        {'params': model.sifd.parameters(), 'lr': lr},
    ], weight_decay=1e-5)

    # Use CosineAnnealingLR for better performance
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc': [],
        'test_rej': [],
        'hos': []
    }

    # Best model tracking
    best_val_acc = 0
    best_hos = 0
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Train with unified loss function
        train_loss, train_acc = train_one_epoch_hybrid(
            model, classifier, train_loader, optimizer, epoch,
            alpha=0.3, beta=0.5, gamma=0.3, delta=0.2, clip_value=1.0)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validate
        val_loss, val_acc = validate_hybrid(model, classifier, val_loader, epoch)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Update learning rate
        scheduler.step()

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            print(f"New best model saved at epoch {epoch} with validation accuracy {val_acc:.2f}%")

    print(f"\n🎉 Training completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)

    print("\n🏁 Final evaluation with fixed parameters...")
    known_acc, unknown_rej, hos_score, confusion_mat = test_open_set_hybrid_dioni(model, classifier, test_loader)

    return {
        'model': model,
        'classifier': classifier,
        'history': history,
        'best_epoch': best_epoch,
        'final_metrics': {
            'known_acc': known_acc,
            'unknown_rej': unknown_rej,
            'hos_score': hos_score,
            'fixed_params': {
                'uncertainty_threshold': uncertainty_threshold,
                'decoupling_threshold': decoupling_threshold,
                'spectral_weight': spectral_weight
            }
        },
        'confusion_matrix': confusion_mat
    }


if __name__ == "__main__":
    # Setup directories
    setup_directories()

    print("Loading data...")

    # Set your fixed parameters here
    UNCERTAINTY_THRESHOLD = 0.4111  # Adjust this value
    DECOUPLING_THRESHOLD = 0.1  # Adjust this value
    SPECTRAL_WEIGHT = 0.7  # Adjust this value

    # Load Dioni/Loukia data and train
    # try:
    #     print("\n---------- DIONI/LOUKIA DATASET ----------")
    #     results_dioni = train_and_evaluate_hybrid_dioni_loukia(
    #         epochs=50,
    #         batch_size=32,
    #         patch_size=7,
    #         lr=0.00001,
    #         uncertainty_threshold=UNCERTAINTY_THRESHOLD,
    #         decoupling_threshold=DECOUPLING_THRESHOLD,
    #         spectral_weight=SPECTRAL_WEIGHT
    #     )
    #
    #     # Save the Dioni model
    #     torch.save({
    #         'model_state_dict': results_dioni['model'].state_dict(),
    #         'fixed_params': results_dioni['final_metrics']['fixed_params']
    #     }, 'outputs/models/best_hybrid_dioni_fixed_model.pth')
    #
    # except Exception as e:
    #     print(f"Error with Dioni/Loukia training: {e}")
    #
    # # Load Houston data and train
    # try:
    #     print("\n---------- HOUSTON DATASET ----------")
    #     results_houston = train_and_evaluate_hybrid(
    #         epochs=50,
    #         batch_size=32,
    #         patch_size=7,
    #         lr=0.00001,
    #         dataset_name="Houston",
    #         uncertainty_threshold=UNCERTAINTY_THRESHOLD,
    #         decoupling_threshold=DECOUPLING_THRESHOLD,
    #         spectral_weight=SPECTRAL_WEIGHT
    #     )
    #
    #     # Save the Houston model
    #     torch.save({
    #         'model_state_dict': results_houston['model'].state_dict(),
    #         'fixed_params': results_houston['final_metrics']['fixed_params']
    #     }, 'outputs/models/best_hybrid_houston_fixed_model.pth')
    #
    # except Exception as e:
    #     print(f"Error with Houston training: {e}")

    # Load Pavia data and train
    try:
        print("\n---------- PAVIA DATASET ----------")
        results_pavia = train_and_evaluate_hybrid(
            epochs=5,
            batch_size=32,
            patch_size=7,
            lr=0.00001,
            dataset_name="Pavia",
            uncertainty_threshold=UNCERTAINTY_THRESHOLD,
            decoupling_threshold=DECOUPLING_THRESHOLD,
            spectral_weight=SPECTRAL_WEIGHT
        )

        # Save the Pavia model
        torch.save({
            'model_state_dict': results_pavia['model'].state_dict(),
            'fixed_params': results_pavia['final_metrics']['fixed_params']
        }, 'outputs/models/best_hybrid_pavia_fixed_model.pth')

    except Exception as e:
        print(f"Error with Pavia training: {e}")