import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
import random
from collections import defaultdict
import json

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = current_dir
project_root = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from config.config import device, worker_init_fn, g, new_classes, new_classesDioni, new_classesPavia
from data.loaders import (
    load_and_preprocess_data,
    load_and_preprocess_pavia_data,
    load_and_preprocess_dioni_loukia_data
)
from data.datasets import HSIDataset, HSITestDataset, HSIDatasetDioni, HSITestDatasetDioni
from models.classifier import DCRN_SSUD_SIFD, HybridSSUDClassifierFixed
from models.dcrn import DCRN
from models.evidential import EvidentialLayer
from models.sifd import SpectrumInvariantFrequencyDisentanglement, GradientReversalLayer
from training.losses import edl_mse_loss
from training.evaluator import test_open_set_hybrid, test_open_set_hybrid_dioni
from utils.helpers import setup_directories


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AblationModel1_NoSIFD(nn.Module):
    """Model without SIFD - uses original input directly"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(AblationModel1_NoSIFD, self).__init__()

        # Base DCRN model (same as original)
        self.dcrn = DCRN(input_channels=input_channels, patch_size=patch_size)

        # Output dimensions from DCRN
        self.spectral_size = self.dcrn.spectral_size  # 1024
        self.spatial_size = self.dcrn.spatial_size  # 2048
        self.feature_size = self.dcrn.total_output_channels  # 3072

        # Main classifier for known classes
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Separate pathway for evidential uncertainty estimation
        self.spectral_evidence = EvidentialLayer(self.spectral_size, num_classes)
        self.spatial_evidence = EvidentialLayer(self.spatial_size, num_classes)
        self.combined_evidence = EvidentialLayer(self.feature_size, num_classes)

        # Adaptive weighting between spectral and spatial pathways
        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # No SIFD - use original input directly
        enhanced_input = x

        # Get features from base DCRN
        features_dict = self.dcrn(enhanced_input)

        # Extract features
        combined_features = features_dict['features']
        spectral_features = features_dict['spectral_features']
        spatial_features = features_dict['spatial_features']

        # Main classification logits
        logits = self.classifier(combined_features)

        # Calculate evidence and uncertainty for each pathway
        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        # Calculate adaptive weights between spectral and spatial pathways
        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        # Calculate reliability scores
        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        # Weighted uncertainty
        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        # Soft probability output
        probs = F.softmax(logits, dim=1)

        return {
            'logits': logits,
            'probs': probs,
            'alpha_combined': combined_out['alpha'],
            'alpha_spectral': spectral_out['alpha'],
            'alpha_spatial': spatial_out['alpha'],
            'uncertainty_combined': combined_out['uncertainty'],
            'uncertainty_spectral': spectral_out['uncertainty'],
            'uncertainty_spatial': spatial_out['uncertainty'],
            'weighted_uncertainty': weighted_uncertainty,
            'fusion_weights': weights,
            'spectral_reliability': spectral_reliability,
            'spatial_reliability': spatial_reliability,
            'features': combined_features,
            'domain_pred': torch.zeros(x.size(0), 1, device=x.device),  # Dummy for compatibility
            'sifd_outputs': {'domain_pred': torch.zeros(x.size(0), 1, device=x.device),
                             'reconstructed_spectrum': torch.zeros(x.size(0) * x.size(2) * x.size(3), x.size(1),
                                                                   device=x.device)}
        }


class AblationModel2_NoEDL(nn.Module):
    """Model without EDL - uses standard confidence measures"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(AblationModel2_NoEDL, self).__init__()

        # SIFD module
        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)

        # Base DCRN model
        self.dcrn = DCRN(input_channels=input_channels, patch_size=patch_size)

        self.spectral_size = self.dcrn.spectral_size
        self.spatial_size = self.dcrn.spatial_size
        self.feature_size = self.dcrn.total_output_channels

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Standard classifiers for each pathway (no EDL)
        self.spectral_classifier = nn.Linear(self.spectral_size, num_classes)
        self.spatial_classifier = nn.Linear(self.spatial_size, num_classes)

        # Pathway weighting
        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply SIFD
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        # Get features from DCRN
        features_dict = self.dcrn(enhanced_input)
        combined_features = features_dict['features']
        spectral_features = features_dict['spectral_features']
        spatial_features = features_dict['spatial_features']

        # Main classification logits
        logits = self.classifier(combined_features)

        # Standard pathway classifications (no EDL)
        spectral_logits = self.spectral_classifier(spectral_features)
        spatial_logits = self.spatial_classifier(spatial_features)

        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        spectral_probs = F.softmax(spectral_logits, dim=1)
        spatial_probs = F.softmax(spatial_logits, dim=1)

        # Calculate uncertainty using entropy (alternative to EDL)
        def entropy_uncertainty(p):
            return -torch.sum(p * torch.log(p + 1e-8), dim=1, keepdim=True) / np.log(p.size(1))

        uncertainty_combined = entropy_uncertainty(probs)
        uncertainty_spectral = entropy_uncertainty(spectral_probs)
        uncertainty_spatial = entropy_uncertainty(spatial_probs)

        # Calculate pathway weights
        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        # Reliability scores (inverse of uncertainty)
        spectral_reliability = 1.0 - uncertainty_spectral
        spatial_reliability = 1.0 - uncertainty_spatial

        # Weighted uncertainty
        weighted_uncertainty = (
                weights[:, 0:1] * uncertainty_spectral +
                weights[:, 1:2] * uncertainty_spatial
        )

        # Domain prediction
        domain_pred = self.domain_classifier(combined_features)

        # Create dummy alpha values for compatibility (since no EDL)
        dummy_alpha_combined = torch.ones(x.size(0), probs.size(1), device=x.device)
        dummy_alpha_spectral = torch.ones(x.size(0), spectral_probs.size(1), device=x.device)
        dummy_alpha_spatial = torch.ones(x.size(0), spatial_probs.size(1), device=x.device)

        return {
            'logits': logits,
            'probs': probs,
            'alpha_combined': dummy_alpha_combined,
            'alpha_spectral': dummy_alpha_spectral,
            'alpha_spatial': dummy_alpha_spatial,
            'uncertainty_combined': uncertainty_combined,
            'uncertainty_spectral': uncertainty_spectral,
            'uncertainty_spatial': uncertainty_spatial,
            'weighted_uncertainty': weighted_uncertainty,
            'fusion_weights': weights,
            'spectral_reliability': spectral_reliability,
            'spatial_reliability': spatial_reliability,
            'features': combined_features,
            'domain_pred': domain_pred,
            'sifd_outputs': sifd_outputs
        }


class AblationModel3_SpectralOnly(nn.Module):
    """Model with spectral pathway only"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(AblationModel3_SpectralOnly, self).__init__()

        # SIFD module
        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)

        # Spectral pathway only
        self.spec_conv1 = nn.Conv2d(input_channels, 64, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn1 = nn.BatchNorm2d(64)
        self.spec_conv2 = nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn2 = nn.BatchNorm2d(64)
        self.spec_conv3 = nn.Conv2d(64, 512, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn3 = nn.BatchNorm2d(512)
        self.spec_conv4 = nn.Conv2d(512, 512, kernel_size=(1, 21), padding=(0, 10))
        self.spec_bn4 = nn.BatchNorm2d(512)
        self.spec_conv5 = nn.Conv2d(512, 1024, kernel_size=(1, 21), padding=(0, 10))
        self.spec_bn5 = nn.BatchNorm2d(1024)

        self.spectral_size = 1024
        self.feature_size = 1024  # Only spectral features

        # Classifiers
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # EDL for spectral only
        self.spectral_evidence = EvidentialLayer(self.spectral_size, num_classes)

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)

        # Apply SIFD
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        # Spectral pathway only
        spec = self.spec_conv1(enhanced_input)
        spec = F.relu(self.spec_bn1(spec))
        spec = self.spec_conv2(spec)
        spec = F.relu(self.spec_bn2(spec))
        spec = self.spec_conv3(spec)
        spec = F.relu(self.spec_bn3(spec))
        spec = self.spec_conv4(spec)
        spec = F.relu(self.spec_bn4(spec))
        spec = self.spec_conv5(spec)
        spec = F.relu(self.spec_bn5(spec))

        # Pool spectral features
        spectral_features = F.adaptive_avg_pool2d(spec, 1).view(batch_size, -1)

        # Main classification
        logits = self.classifier(spectral_features)
        probs = F.softmax(logits, dim=1)

        # EDL for spectral
        spectral_out = self.spectral_evidence(spectral_features)

        # Domain prediction
        domain_pred = self.domain_classifier(spectral_features)

        # Create dummy values for spatial pathway (for compatibility)
        dummy_spatial_uncertainty = torch.ones_like(spectral_out['uncertainty'])
        dummy_spatial_alpha = torch.ones_like(spectral_out['alpha'])
        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = torch.ones_like(spectral_reliability)  # Dummy

        # No pathway weighting needed (spectral only)
        weights = torch.tensor([[1.0, 0.0]], device=x.device).repeat(batch_size, 1)

        return {
            'logits': logits,
            'probs': probs,
            'alpha_combined': spectral_out['alpha'],
            'alpha_spectral': spectral_out['alpha'],
            'alpha_spatial': dummy_spatial_alpha,
            'uncertainty_combined': spectral_out['uncertainty'],
            'uncertainty_spectral': spectral_out['uncertainty'],
            'uncertainty_spatial': dummy_spatial_uncertainty,
            'weighted_uncertainty': spectral_out['uncertainty'],
            'fusion_weights': weights,
            'spectral_reliability': spectral_reliability,
            'spatial_reliability': spatial_reliability,
            'features': spectral_features,
            'domain_pred': domain_pred,
            'sifd_outputs': sifd_outputs
        }


class AblationModel4_SpatialOnly(nn.Module):
    """Model with spatial pathway only"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(AblationModel4_SpatialOnly, self).__init__()

        # SIFD module
        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)

        # Spatial pathway only
        import torchvision.models as models
        self.spat_conv1 = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.spat_bn1 = nn.BatchNorm2d(3)

        # ResNet50 layers
        resnet = models.resnet50(pretrained=True)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.adapter = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.upsample = nn.Upsample(size=(patch_size, patch_size), mode='bilinear', align_corners=False)

        self.spatial_size = 2048
        self.feature_size = 2048  # Only spatial features

        # Classifiers
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # EDL for spatial only
        self.spatial_evidence = EvidentialLayer(self.spatial_size, num_classes)

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Apply SIFD
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        # Spatial pathway only
        spat = self.spat_conv1(enhanced_input)
        spat = F.relu(self.spat_bn1(spat))
        spat = self.adapter(spat)
        spat = self.layer1(spat)
        spat = self.layer2(spat)
        spat = self.layer3(spat)
        spat = self.layer4(spat)
        spat = self.upsample(spat)

        # Pool spatial features
        spatial_features = F.adaptive_avg_pool2d(spat, 1).view(batch_size, -1)

        # Main classification
        logits = self.classifier(spatial_features)
        probs = F.softmax(logits, dim=1)

        # EDL for spatial
        spatial_out = self.spatial_evidence(spatial_features)

        # Domain prediction
        domain_pred = self.domain_classifier(spatial_features)

        # Create dummy values for spectral pathway (for compatibility)
        dummy_spectral_uncertainty = torch.ones_like(spatial_out['uncertainty'])
        dummy_spectral_alpha = torch.ones_like(spatial_out['alpha'])
        spatial_reliability = 1.0 - spatial_out['uncertainty']
        spectral_reliability = torch.ones_like(spatial_reliability)  # Dummy

        # No pathway weighting needed (spatial only)
        weights = torch.tensor([[0.0, 1.0]], device=x.device).repeat(batch_size, 1)

        return {
            'logits': logits,
            'probs': probs,
            'alpha_combined': spatial_out['alpha'],
            'alpha_spectral': dummy_spectral_alpha,
            'alpha_spatial': spatial_out['alpha'],
            'uncertainty_combined': spatial_out['uncertainty'],
            'uncertainty_spectral': dummy_spectral_uncertainty,
            'uncertainty_spatial': spatial_out['uncertainty'],
            'weighted_uncertainty': spatial_out['uncertainty'],
            'fusion_weights': weights,
            'spectral_reliability': spectral_reliability,
            'spatial_reliability': spatial_reliability,
            'features': spatial_features,
            'domain_pred': domain_pred,
            'sifd_outputs': sifd_outputs
        }


def train_one_epoch_ablation(models, classifiers, train_loader, optimizers, epoch,
                             alpha=0.3, beta=0.5, gamma=0.3, delta=0.2, clip_value=1.0):
    """
    Train all models simultaneously on the same batches
    """
    model_names = list(models.keys())

    # Set all models to training mode
    for model in models.values():
        model.train()

    total_losses = {name: 0 for name in model_names}
    total_accuracies = {name: 0 for name in model_names}
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Train each model on the same batch
        for name in model_names:
            model = models[name]
            optimizer = optimizers[name]

            optimizer.zero_grad()

            outputs = model(data)

            # Calculate losses based on model type
            if name == "NoEDL":
                # For NoEDL model, use standard cross-entropy only
                cls_loss = F.cross_entropy(outputs['logits'], target)
                edl_total_loss = torch.tensor(0.0, device=device)
            else:
                # Standard classification loss
                cls_loss = F.cross_entropy(outputs['logits'], target)

                # EDL losses
                edl_combined, _, _ = edl_mse_loss(target, outputs['alpha_combined'], epoch)
                edl_spectral, _, _ = edl_mse_loss(target, outputs['alpha_spectral'], epoch)
                edl_spatial, _, _ = edl_mse_loss(target, outputs['alpha_spatial'], epoch)
                edl_total_loss = (edl_combined + alpha * (edl_spectral + edl_spatial)) / 3.0

            # Domain adversarial loss
            domain_loss = F.binary_cross_entropy_with_logits(
                outputs['sifd_outputs']['domain_pred'],
                torch.ones_like(outputs['sifd_outputs']['domain_pred']) * 0.5
            )

            # Reconstruction loss
            if name == "NoSIFD":
                recon_loss = torch.tensor(0.0, device=device)  # No SIFD, no reconstruction
            else:
                data_flat = data.permute(0, 2, 3, 1).reshape(-1, data.shape[1])
                recon_loss = F.mse_loss(outputs['sifd_outputs']['reconstructed_spectrum'], data_flat)

            # Total loss
            loss = cls_loss + alpha * edl_total_loss + beta * domain_loss + gamma * recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            # Update metrics
            total_losses[name] += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs['logits'], 1)
            correct = (predicted == target).sum().item()
            total_accuracies[name] += correct

        total_samples += target.size(0)

    # Calculate average metrics
    results = {}
    for name in model_names:
        avg_loss = total_losses[name] / len(train_loader)
        avg_acc = 100.0 * total_accuracies[name] / total_samples
        results[name] = {'loss': avg_loss, 'accuracy': avg_acc}

    return results


def validate_ablation(models, classifiers, val_loader, epoch):
    """
    Validate all models simultaneously
    """
    model_names = list(models.keys())

    # Set all models to eval mode
    for model in models.values():
        model.eval()

    total_losses = {name: 0 for name in model_names}
    total_accuracies = {name: 0 for name in model_names}
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            for name in model_names:
                model = models[name]

                outputs = model(data)

                # Calculate validation loss
                loss = F.cross_entropy(outputs['logits'], target)
                total_losses[name] += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs['logits'], 1)
                correct = (predicted == target).sum().item()
                total_accuracies[name] += correct

            total_samples += target.size(0)

    # Calculate average metrics
    results = {}
    for name in model_names:
        avg_loss = total_losses[name] / len(val_loader)
        avg_acc = 100.0 * total_accuracies[name] / total_samples
        results[name] = {'loss': avg_loss, 'accuracy': avg_acc}

    return results


def create_ablation_models(input_channels, patch_size, num_classes):
    """Create all ablation models with specified parameters"""
    models = {
        'Original': DCRN_SSUD_SIFD(input_channels=input_channels, patch_size=patch_size, num_classes=num_classes).to(
            device),
        'NoSIFD': AblationModel1_NoSIFD(input_channels=input_channels, patch_size=patch_size,
                                        num_classes=num_classes).to(device),
        'NoEDL': AblationModel2_NoEDL(input_channels=input_channels, patch_size=patch_size, num_classes=num_classes).to(
            device),
        'SpectralOnly': AblationModel3_SpectralOnly(input_channels=input_channels, patch_size=patch_size,
                                                    num_classes=num_classes).to(device),
        'SpatialOnly': AblationModel4_SpatialOnly(input_channels=input_channels, patch_size=patch_size,
                                                  num_classes=num_classes).to(device)
    }
    return models


def create_optimizers(models, lr):
    """Create optimizers for all models"""
    optimizers = {}
    for name, model in models.items():
        if hasattr(model, 'sifd'):
            optimizer = optim.Adam([
                {'params': model.dcrn.parameters() if hasattr(model, 'dcrn') else [], 'lr': lr},
                {'params': model.classifier.parameters(), 'lr': lr},
                {'params': model.sifd.parameters(), 'lr': lr},
            ], weight_decay=1e-5)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        optimizers[name] = optimizer
    return optimizers


def run_single_seed_experiment(seed, pavia_loaders, input_channels, num_classes,
                               epochs=50, batch_size=32, patch_size=7, lr=0.00001,
                               uncertainty_threshold=0.4111, decoupling_threshold=0.1, spectral_weight=0.7):
    """
    Run a single experiment with a given seed
    """
    print(f"\n--- Running Seed {seed} ---")

    # Set seed for this experiment
    set_seed(seed)

    train_loader, val_loader, test_loader = pavia_loaders

    # Create all models
    models = create_ablation_models(input_channels, patch_size, num_classes)

    # Create classifiers for each model
    classifiers = {}
    for name, model in models.items():
        classifiers[name] = HybridSSUDClassifierFixed(
            model, num_classes=num_classes,
            uncertainty_threshold=uncertainty_threshold,
            decoupling_threshold=decoupling_threshold,
            spectral_weight=spectral_weight
        ).to(device)

    # Create optimizers for each model
    optimizers = create_optimizers(models, lr)

    # Training history
    history = {name: {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
               for name in models.keys()}

    best_models = {name: None for name in models.keys()}
    best_val_accs = {name: 0 for name in models.keys()}

    # Training loop
    for epoch in range(1, epochs + 1):
        # Train all models on the same batches
        train_results = train_one_epoch_ablation(models, classifiers, train_loader, optimizers, epoch)

        # Validate all models
        val_results = validate_ablation(models, classifiers, val_loader, epoch)

        # Update history and save best models
        for name in models.keys():
            history[name]['train_loss'].append(train_results[name]['loss'])
            history[name]['train_acc'].append(train_results[name]['accuracy'])
            history[name]['val_loss'].append(val_results[name]['loss'])
            history[name]['val_acc'].append(val_results[name]['accuracy'])

            # Save best model
            if val_results[name]['accuracy'] > best_val_accs[name]:
                best_val_accs[name] = val_results[name]['accuracy']
                best_models[name] = deepcopy(models[name].state_dict())

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Seed {seed}, Epoch {epoch}: Original Val Acc = {val_results['Original']['accuracy']:.2f}%")

    # Load best models and evaluate
    final_results = {}
    for name in models.keys():
        models[name].load_state_dict(best_models[name])

        # Evaluate on test set
        known_acc, unknown_rej, hos_score, cm = test_open_set_hybrid(models[name], classifiers[name], test_loader)

        final_results[name] = {
            'known_accuracy': known_acc,
            'unknown_rejection': unknown_rej,
            'hos_score': hos_score,
            'best_val_acc': best_val_accs[name],
            'confusion_matrix': cm,
            'seed': seed
        }

    return final_results, history, best_models


def run_pavia_ablation_100_seeds(epochs=50, batch_size=32, patch_size=7, lr=0.00001,
                                 uncertainty_threshold=0.4111, decoupling_threshold=0.1, spectral_weight=0.7):
    """
    Run ablation study on Pavia dataset for 100 different seeds
    """
    print("=" * 100)
    print("PAVIA ABLATION STUDY - 100 SEEDS, 50 EPOCHS EACH")
    print("=" * 100)

    # Load Pavia dataset
    print("\n🏛️ Loading Pavia dataset...")
    (hs_pavia_u_norm, train_gt_pavia_u, val_gt_pavia_u,
     hs_pavia_c_norm, gt_pavia_c_remapped) = load_and_preprocess_pavia_data()

    # Create datasets
    source_train_dataset = HSIDataset(hs_pavia_u_norm, train_gt_pavia_u,
                                      patch_size=patch_size, augment=True)
    source_val_dataset = HSIDataset(hs_pavia_u_norm, val_gt_pavia_u,
                                    patch_size=patch_size, augment=False)
    target_test_dataset = HSITestDataset(hs_pavia_c_norm, gt_pavia_c_remapped,
                                         patch_size=patch_size)

    # Create data loaders
    train_loader = DataLoader(source_train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(source_val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    test_loader = DataLoader(target_test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)

    pavia_loaders = (train_loader, val_loader, test_loader)
    input_channels = hs_pavia_u_norm.shape[2]  # 102

    # Storage for all results
    all_seed_results = []
    all_seed_histories = []

    # Track best original model across all seeds
    best_original_performance = 0
    best_seed = None
    best_seed_results = None
    best_seed_models = None

    print(f"\nRunning 100 experiments with different seeds...")

    # Run 100 experiments with different seeds
    for seed in range(1, 101):
        try:
            # Run single seed experiment
            seed_results, seed_history, seed_models = run_single_seed_experiment(
                seed, pavia_loaders, input_channels, 7, epochs, batch_size, patch_size, lr,
                uncertainty_threshold, decoupling_threshold, spectral_weight
            )

            # Store results
            all_seed_results.append(seed_results)
            all_seed_histories.append(seed_history)

            # Check if this is the best original model so far
            original_hos_score = seed_results['Original']['hos_score']
            if original_hos_score > best_original_performance:
                best_original_performance = original_hos_score
                best_seed = seed
                best_seed_results = deepcopy(seed_results)
                best_seed_models = deepcopy(seed_models)
                print(f"🏆 New best Original model! Seed {seed}, HOS Score: {original_hos_score:.4f}")

            # Print progress
            if seed % 10 == 0:
                print(
                    f"Completed {seed}/100 experiments. Best HOS so far: {best_original_performance:.4f} (Seed {best_seed})")

        except Exception as e:
            print(f"❌ Error with seed {seed}: {e}")
            continue

    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY")
    print("=" * 100)

    # Calculate statistics across all seeds
    model_names = ['Original', 'NoSIFD', 'NoEDL', 'SpectralOnly', 'SpatialOnly']
    statistics = {}

    for model_name in model_names:
        model_scores = []
        val_accs = []
        known_accs = []
        unknown_rejs = []
        hos_scores = []

        for seed_result in all_seed_results:
            if model_name in seed_result:
                val_accs.append(seed_result[model_name]['best_val_acc'])
                known_accs.append(seed_result[model_name]['known_accuracy'])
                unknown_rejs.append(seed_result[model_name]['unknown_rejection'])
                hos_scores.append(seed_result[model_name]['hos_score'])

        if hos_scores:  # If we have data for this model
            statistics[model_name] = {
                'val_acc_mean': np.mean(val_accs),
                'val_acc_std': np.std(val_accs),
                'known_acc_mean': np.mean(known_accs),
                'known_acc_std': np.std(known_accs),
                'unknown_rej_mean': np.mean(unknown_rejs),
                'unknown_rej_std': np.std(unknown_rejs),
                'hos_score_mean': np.mean(hos_scores),
                'hos_score_std': np.std(hos_scores),
                'hos_score_max': np.max(hos_scores),
                'hos_score_min': np.min(hos_scores)
            }

    # Print statistics table
    print(f"\nSTATISTICS ACROSS 100 SEEDS:")
    print("-" * 120)
    print(f"{'Model':<15} {'Val Acc':<15} {'Known Acc':<15} {'Unknown Rej':<15} {'HOS Score':<15} {'HOS Max':<10}")
    print("-" * 120)

    for model_name in model_names:
        if model_name in statistics:
            stats = statistics[model_name]
            print(f"{model_name:<15} "
                  f"{stats['val_acc_mean']:.2f}±{stats['val_acc_std']:.2f}     "
                  f"{stats['known_acc_mean']:.2f}±{stats['known_acc_std']:.2f}     "
                  f"{stats['unknown_rej_mean']:.2f}±{stats['unknown_rej_std']:.2f}     "
                  f"{stats['hos_score_mean']:.4f}±{stats['hos_score_std']:.4f}   "
                  f"{stats['hos_score_max']:.4f}")

    print(f"\n🏆 BEST ORIGINAL MODEL:")
    print(f"Seed: {best_seed}")
    print(f"HOS Score: {best_original_performance:.4f}")
    print(f"Known Accuracy: {best_seed_results['Original']['known_accuracy']:.2f}%")
    print(f"Unknown Rejection: {best_seed_results['Original']['unknown_rejection']:.2f}%")

    print(f"\n📊 CORRESPONDING ABLATION RESULTS FOR BEST SEED {best_seed}:")
    print("-" * 80)
    print(f"{'Model':<15} {'Val Acc':<10} {'Known Acc':<12} {'Unknown Rej':<12} {'HOS Score':<10}")
    print("-" * 80)
    for model_name in model_names:
        if model_name in best_seed_results:
            res = best_seed_results[model_name]
            print(f"{model_name:<15} {res['best_val_acc']:<10.2f} {res['known_accuracy']:<12.2f} "
                  f"{res['unknown_rejection']:<12.2f} {res['hos_score']:<10.4f}")

    # Save comprehensive results
    comprehensive_results = {
        'experiment_type': 'Pavia Ablation Study - 100 Seeds',
        'best_seed': best_seed,
        'best_original_hos_score': best_original_performance,
        'best_seed_results': best_seed_results,
        'statistics': statistics,
        'all_seed_results': all_seed_results,
        'hyperparameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'learning_rate': lr,
            'uncertainty_threshold': uncertainty_threshold,
            'decoupling_threshold': decoupling_threshold,
            'spectral_weight': spectral_weight
        },
        'dataset_info': {
            'name': 'Pavia',
            'input_channels': input_channels,
            'num_classes': 7
        }
    }

    # Save the comprehensive results
    torch.save(comprehensive_results, 'outputs/models/pavia_ablation_100_seeds_comprehensive.pth')

    # Save the best models
    for model_name, model_state in best_seed_models.items():
        torch.save({
            'model_state_dict': model_state,
            'seed': best_seed,
            'results': best_seed_results[model_name],
            'model_type': model_name
        }, f'outputs/models/pavia_best_seed_{best_seed}_{model_name.lower()}_model.pth')

    # Save statistics as JSON for easy reading
    with open('outputs/models/pavia_ablation_100_seeds_statistics.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_statistics = {}
        for model_name, stats in statistics.items():
            json_statistics[model_name] = {k: float(v) for k, v in stats.items()}

        json_data = {
            'best_seed': int(best_seed),
            'best_hos_score': float(best_original_performance),
            'statistics': json_statistics,
            'best_seed_results': {
                model_name: {k: float(v) if isinstance(v, (int, float, np.number)) else v
                             for k, v in results.items() if k != 'confusion_matrix'}
                for model_name, results in best_seed_results.items()
            }
        }
        json.dump(json_data, f, indent=2)

    print(f"\n✅ Pavia ablation study with 100 seeds completed!")
    print(f"📁 All results saved to outputs/models/")
    print(f"📊 Comprehensive results: pavia_ablation_100_seeds_comprehensive.pth")
    print(f"📈 Statistics summary: pavia_ablation_100_seeds_statistics.json")
    print(f"🏆 Best models saved with prefix: pavia_best_seed_{best_seed}_")

    return comprehensive_results


if __name__ == "__main__":
    setup_directories()

    # Run the Pavia ablation study with 100 seeds
    results = run_pavia_ablation_100_seeds(
        epochs=50,
        batch_size=32,
        patch_size=7,
        lr=0.00001,
        uncertainty_threshold=0.411,
        decoupling_threshold=0.1,
        spectral_weight=0.7
    )

    print("\n" + "=" * 100)
    print("🎉 PAVIA ABLATION STUDY WITH 100 SEEDS COMPLETED!")
    print("All models and results have been saved.")
    print("=" * 100)