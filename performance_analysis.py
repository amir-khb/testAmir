import os
import sys
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
from tabulate import tabulate

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
project_root = current_dir
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Import project modules
from src.config.config import device, worker_init_fn, g
from src.data.loaders import (
    load_and_preprocess_data,
    load_and_preprocess_pavia_data,
    load_and_preprocess_dioni_loukia_data
)
from src.data.datasets import HSIDataset, HSITestDataset, HSIDatasetDioni, HSITestDatasetDioni
from src.models.classifier import DCRN_SSUD_SIFD, HybridSSUDClassifierFixed
from src.utils.helpers import setup_directories, count_parameters

try:
    from ptflops import get_model_complexity_info

    PTFLOPS_AVAILABLE = True
    print("✓ ptflops available")
except ImportError:
    try:
        from thop import profile, clever_format

        THOP_AVAILABLE = True
        PTFLOPS_AVAILABLE = False
        print("✓ thop available (ptflops not found)")
    except ImportError:
        print("⚠ Neither ptflops nor thop available. Install with: pip install ptflops thop")
        PTFLOPS_AVAILABLE = False
        THOP_AVAILABLE = False


def measure_flops(model, input_shape):
    """Measure FLOPS for the model using multiple methods"""
    print(f"  Measuring FLOPS for input shape: {input_shape}")

    if PTFLOPS_AVAILABLE:
        try:
            print("  Using ptflops...")
            model_copy = deepcopy(model)
            model_copy.eval()

            macs, params = get_model_complexity_info(
                model_copy,
                input_shape,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )

            flops = macs * 2  # MACs to FLOPS conversion
            result = format_flops(flops)
            print(f"  FLOPS measured: {result}")
            return result

        except Exception as e:
            print(f"  ptflops failed: {e}")

    if 'THOP_AVAILABLE' in globals() and THOP_AVAILABLE:
        try:
            print("  Using thop...")
            model_copy = deepcopy(model)
            model_copy.eval()

            # Create dummy input
            dummy_input = torch.randn(1, *input_shape).to(device)
            flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)

            result = format_flops(flops)
            print(f"  FLOPS measured: {result}")
            return result

        except Exception as e:
            print(f"  thop failed: {e}")

    # Manual FLOPS estimation as fallback
    try:
        print("  Using manual estimation...")
        result = estimate_flops_manual(model, input_shape)
        print(f"  FLOPS estimated: {result}")
        return result
    except Exception as e:
        print(f"  Manual estimation failed: {e}")
        return "N/A"


def format_flops(flops):
    """Format FLOPS in readable units"""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f}M"
    else:
        return f"{flops / 1e3:.2f}K"


def estimate_flops_manual(model, input_shape):
    """Manual FLOPS estimation based on model parameters"""
    total_params = count_parameters(model)
    # Very rough estimation: assume each parameter contributes to ~10 operations
    estimated_flops = total_params * 10
    return format_flops(estimated_flops)


def format_time(seconds):
    """Format time in readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_parameters(params):
    """Format parameter count in readable format"""
    if params >= 1e6:
        return f"{params / 1e6:.2f}M"
    elif params >= 1e3:
        return f"{params / 1e3:.2f}K"
    else:
        return str(params)


def measure_training_time_full_epoch(model, train_loader, optimizer):
    """Actually train for one full epoch to get accurate timing"""
    print(f"  Starting full epoch training with {len(train_loader)} batches...")

    model.train()
    total_batches = len(train_loader)

    # GPU synchronization before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)

        # Simple loss calculation
        import torch.nn.functional as F
        loss = F.cross_entropy(outputs['logits'], target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Progress indicator
        if (batch_idx + 1) % max(1, total_batches // 10) == 0:
            progress = (batch_idx + 1) / total_batches * 100
            print(f"    Progress: {progress:.1f}% ({batch_idx + 1}/{total_batches})")

    # GPU synchronization after finishing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.time() - start_time
    print(f"  Full epoch completed in {format_time(elapsed_time)}")

    return elapsed_time


def measure_inference_time_full(model, classifier, test_loader):
    """Run inference on full test set"""
    print(f"  Starting full inference with {len(test_loader)} batches...")

    model.eval()
    total_batches = len(test_loader)

    # GPU synchronization before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # Get predictions
            predictions = classifier.predict(data)

            # Progress indicator
            if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"    Progress: {progress:.1f}% ({batch_idx + 1}/{total_batches})")

    # GPU synchronization after finishing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.time() - start_time
    print(f"  Full inference completed in {format_time(elapsed_time)}")

    return elapsed_time


def analyze_dataset_performance(dataset_name):
    """Analyze performance for a specific dataset"""
    print(f"\n{'=' * 60}")
    print(f"ANALYZING {dataset_name.upper()} DATASET")
    print(f"{'=' * 60}")

    try:
        # Load data based on dataset
        print("Loading dataset...")
        if dataset_name == "Houston":
            (hs_source, gt_source, hs_target, gt_target, _, _) = load_and_preprocess_data()
            source_train_dataset = HSIDataset(hs_source, gt_source, patch_size=7, augment=True)
            target_test_dataset = HSITestDataset(hs_target, gt_target, patch_size=7)
            input_channels = hs_source.shape[2]
            num_classes = 7

        elif dataset_name == "Pavia":
            (hs_source, train_gt, val_gt, hs_target, gt_target) = load_and_preprocess_pavia_data()
            source_train_dataset = HSIDataset(hs_source, train_gt, patch_size=7, augment=True)
            target_test_dataset = HSITestDataset(hs_target, gt_target, patch_size=7)
            input_channels = hs_source.shape[2]
            num_classes = 7

        elif dataset_name == "Dioni-Loukia":
            (hs_source, train_gt, val_gt, hs_target, gt_target) = load_and_preprocess_dioni_loukia_data()
            source_train_dataset = HSIDatasetDioni(hs_source, train_gt, patch_size=7, augment=True)
            target_test_dataset = HSITestDatasetDioni(hs_target, gt_target, patch_size=7)
            input_channels = hs_source.shape[2]
            num_classes = 9

        print(f"✓ Dataset loaded successfully")
        print(f"  Input channels: {input_channels}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Training samples: {len(source_train_dataset)}")
        print(f"  Test samples: {len(target_test_dataset)}")

        # Create data loaders
        print("Creating data loaders...")
        batch_size = 32
        train_loader = DataLoader(source_train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
        test_loader = DataLoader(target_test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)

        print(f"✓ Data loaders created")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        if len(train_loader) == 0:
            raise ValueError("Training loader is empty!")
        if len(test_loader) == 0:
            raise ValueError("Test loader is empty!")

        # Create model
        print("Creating model...")
        model = DCRN_SSUD_SIFD(input_channels=input_channels, patch_size=7, num_classes=num_classes).to(device)
        classifier = HybridSSUDClassifierFixed(model, num_classes=num_classes).to(device)
        print(f"✓ Model created and moved to {device}")

        # 🔥 PARAMETER ANALYSIS SECTION - ADDED HERE 🔥
        print("\n" + "=" * 60)
        print("DETAILED PARAMETER ANALYSIS")
        print("=" * 60)

        # Count total and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

        # Memory usage estimate
        param_size_mb = (trainable_params * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"Estimated memory for parameters: {param_size_mb:.2f} MB")

        # Check if this matches your 56M
        if abs(trainable_params - 56_730_000) < 1000:
            print(f"🎯 MATCH FOUND! This model has ~56M trainable parameters")
            print(f"   Your reported: 56,730,000")
            print(f"   This model:   {trainable_params:,}")
            print(f"   Difference:   {abs(trainable_params - 56_730_000):,}")
        else:
            print(f"ℹ️  Different from your reported 56M:")
            print(f"   Your reported: 56,730,000")
            print(f"   This model:   {trainable_params:,}")
            print(f"   Difference:   {abs(trainable_params - 56_730_000):,}")

        # Check which parameters are trainable/frozen
        trainable_layers = []
        frozen_layers = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_layers.append(name)
            else:
                frozen_layers.append(name)

        if frozen_layers:
            print(f"\n🔒 FROZEN LAYERS ({len(frozen_layers)} layers):")
            for layer in frozen_layers[:5]:  # Show first 5
                print(f"   - {layer}")
            if len(frozen_layers) > 5:
                print(f"   ... and {len(frozen_layers) - 5} more frozen layers")
        else:
            print(f"\n✅ ALL LAYERS ARE TRAINABLE")

        print(f"\n🔓 TRAINABLE LAYERS: {len(trainable_layers)} layers")
        print("=" * 60)
        # 🔥 END OF PARAMETER ANALYSIS SECTION 🔥

        # Count parameters (your existing code - now for comparison)
        print("Counting parameters (legacy method)...")
        total_params_old = count_parameters(model)
        print(f"✓ Total parameters (old method): {format_parameters(total_params_old)}")

        # Verify consistency
        if total_params != total_params_old:
            print(f"⚠️  WARNING: Parameter count mismatch!")
            print(f"   New method: {total_params:,}")
            print(f"   Old method: {total_params_old:,}")
        else:
            print(f"✅ Parameter counts consistent between methods")

        # Measure FLOPS
        print("Measuring FLOPS...")
        flops = measure_flops(model, (input_channels, 7, 7))

        # Create optimizer
        print("Creating optimizer...")
        optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
        print("✓ Optimizer created")

        # Test one batch first to make sure everything works
        print("Testing one batch...")
        test_batch = next(iter(train_loader))
        data, target = test_batch[0].to(device), test_batch[1].to(device)
        outputs = model(data)
        print(f"✓ Forward pass successful, output shape: {outputs['logits'].shape}")

        # Measure training time (full epoch)
        print("\n" + "=" * 40)
        print("MEASURING TRAINING TIME (1 FULL EPOCH)")
        print("=" * 40)
        training_time = measure_training_time_full_epoch(model, train_loader, optimizer)

        # Measure inference time (full test set)
        print("\n" + "=" * 40)
        print("MEASURING INFERENCE TIME (FULL TEST SET)")
        print("=" * 40)
        inference_time = measure_inference_time_full(model, classifier, test_loader)

        # Updated result dictionary with detailed parameter info
        result = {
            'dataset': dataset_name,
            'training_time': training_time,
            'testing_time': inference_time,
            'flops': flops,
            'parameters': trainable_params,  # Now using trainable parameters
            'total_parameters': total_params,
            'frozen_parameters': frozen_params,
            'trainable_percentage': 100 * trainable_params / total_params,
            'matches_56M': abs(trainable_params - 56_730_000) < 1000,
            'input_channels': input_channels,
            'num_classes': num_classes,
            'train_samples': len(source_train_dataset),
            'test_samples': len(target_test_dataset),
            'train_batches': len(train_loader),
            'test_batches': len(test_loader)
        }

        print(f"\n✓ {dataset_name} analysis completed successfully!")
        return result

    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'training_time': "Error",
            'testing_time': "Error",
            'flops': "Error",
            'parameters': "Error",
            'total_parameters': "Error",
            'frozen_parameters': "Error",
            'trainable_percentage': "Error",
            'matches_56M': False,
            'input_channels': "Error",
            'num_classes': "Error",
            'train_samples': "Error",
            'test_samples': "Error",
            'train_batches': "Error",
            'test_batches': "Error"
        }


def main():
    """Main function to analyze all datasets"""
    print("🚀 PERFORMANCE ANALYSIS FOR HYPERSPECTRAL IMAGE CLASSIFICATION")
    print("=" * 80)
    print(f"Device: {device}")
    print("=" * 80)

    # Setup directories
    setup_directories()

    # Ask user which datasets to analyze
    print("\nAvailable datasets:")
    print("1. Houston")
    print("2. Pavia")
    print("3. Dioni-Loukia")
    print("4. All datasets")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        datasets = ["Houston"]
    elif choice == "2":
        datasets = ["Pavia"]
    elif choice == "3":
        datasets = ["Dioni-Loukia"]
    else:
        datasets = ["Houston", "Pavia", "Dioni-Loukia"]

    results = []

    for dataset_name in datasets:
        result = analyze_dataset_performance(dataset_name)
        results.append(result)

        # Print intermediate results with parameter details
        if isinstance(result['training_time'], (int, float)):
            print(f"\n📊 {dataset_name} Results Summary:")
            print(f"  Training time (1 epoch): {format_time(result['training_time'])}")
            print(f"  Testing time (full): {format_time(result['testing_time'])}")
            print(f"  FLOPS: {result['flops']}")
            print(f"  Total parameters: {format_parameters(result['total_parameters'])}")
            print(f"  Trainable parameters: {format_parameters(result['parameters'])}")
            print(f"  Frozen parameters: {format_parameters(result['frozen_parameters'])}")
            print(f"  Trainable percentage: {result['trainable_percentage']:.1f}%")
            print(f"  Matches your 56M? {'✅ YES' if result['matches_56M'] else '❌ NO'}")

    # Print final results table
    print(f"\n{'=' * 120}")
    print("FINAL PERFORMANCE ANALYSIS RESULTS")
    print(f"{'=' * 120}")

    # Enhanced performance table with parameter details
    table_data = []
    headers = ["Dataset", "Training Time", "Testing Time", "FLOPS", "Trainable Params", "Total Params", "Frozen Params",
               "56M Match?"]

    for result in results:
        table_data.append([
            result['dataset'],
            format_time(result['training_time']) if isinstance(result['training_time'], (int, float)) else result[
                'training_time'],
            format_time(result['testing_time']) if isinstance(result['testing_time'], (int, float)) else result[
                'testing_time'],
            result['flops'],
            format_parameters(result['parameters']) if isinstance(result['parameters'], (int, float)) else result[
                'parameters'],
            format_parameters(result['total_parameters']) if isinstance(result['total_parameters'], (int, float)) else
            result['total_parameters'],
            format_parameters(result['frozen_parameters']) if isinstance(result['frozen_parameters'], (int, float)) else
            result['frozen_parameters'],
            "✅ YES" if result['matches_56M'] else "❌ NO"
        ])

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Detailed information table
    print(f"\n{'=' * 100}")
    print("DETAILED DATASET INFORMATION")
    print(f"{'=' * 100}")

    detailed_headers = ["Dataset", "Input Channels", "Classes", "Train Samples", "Test Samples", "Train Batches",
                        "Test Batches"]
    detailed_data = []

    for result in results:
        detailed_data.append([
            result['dataset'],
            result['input_channels'],
            result['num_classes'],
            result['train_samples'],
            result['test_samples'],
            result['train_batches'],
            result['test_batches']
        ])

    print(tabulate(detailed_data, headers=detailed_headers, tablefmt="grid"))

    # Save results to file with enhanced information
    with open('outputs/performance_analysis.txt', 'w') as f:
        f.write("Performance Analysis Results with Parameter Details\n")
        f.write("=" * 60 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
        f.write("\n\nDetailed Information\n")
        f.write("=" * 30 + "\n\n")
        f.write(tabulate(detailed_data, headers=detailed_headers, tablefmt="grid"))

        f.write("\n\nParameter Analysis Summary:\n")
        for result in results:
            if isinstance(result['parameters'], (int, float)):
                f.write(f"\n{result['dataset']}:\n")
                f.write(f"  Total parameters: {result['total_parameters']:,}\n")
                f.write(f"  Trainable parameters: {result['parameters']:,}\n")
                f.write(f"  Frozen parameters: {result['frozen_parameters']:,}\n")
                f.write(f"  Trainable percentage: {result['trainable_percentage']:.2f}%\n")
                f.write(f"  Matches 56M target: {result['matches_56M']}\n")

        f.write("\n\nRaw Results:\n")
        for result in results:
            f.write(f"\n{result['dataset']}:\n")
            for key, value in result.items():
                f.write(f"  {key}: {value}\n")

    print(f"\n💾 Results saved to: outputs/performance_analysis.txt")

    # Print enhanced notes
    print(f"\n{'=' * 120}")
    print("📝 NOTES:")
    print("- Training time is for 1 COMPLETE epoch (all training batches)")
    print("- Testing time is for COMPLETE inference on test set (all test batches)")
    print("- Parameters column shows TRAINABLE parameters (what affects your 56M count)")
    print("- Total parameters includes both trainable and frozen parameters")
    print("- Frozen parameters are not updated during training")
    print("- 56M Match indicates whether trainable parameters ≈ 56,730,000")
    print("- Measurements include real training/inference, not estimates")
    print("- Progress is shown during long operations")
    print(f"- Device used: {device}")
    print("- Batch size: 32 for all measurements")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()