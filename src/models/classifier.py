import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcrn import DCRN
from .sifd import SpectrumInvariantFrequencyDisentanglement, GradientReversalLayer
from .evidential import EvidentialLayer


class DCRN_SSUD_SIFD(nn.Module):
    """
    Enhanced DCRN-SSUD with Spectrum-Invariant Frequency Disentanglement
    """

    def __init__(self, input_channels=48, patch_size=7, num_classes=7):
        super(DCRN_SSUD_SIFD, self).__init__()

        # SIFD module for domain generalization
        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)

        # Base DCRN model
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

        # Domain classifier on combined features
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        # Proper weight initialization for better training stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply SIFD to extract domain-invariant features
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']

        # Fuse with original input with adaptive weighting
        enhanced_input = x + 0.5 * invariant_features

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
        weights = self.pathway_weight(pathway_input)  # [w_spectral, w_spatial]

        # Calculate reliability scores (inverse of uncertainty)
        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        # Weighted uncertainty using adaptive weights
        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        # Soft probability output using softmax
        probs = F.softmax(logits, dim=1)

        # Domain adversarial classification
        domain_pred = self.domain_classifier(combined_features)

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
            'domain_pred': domain_pred,
            'sifd_outputs': sifd_outputs
        }


class HybridSSUDClassifierFixed(nn.Module):
    """
    Simplified hybrid classifier with fixed thresholds
    """

    def __init__(self, model, num_classes=7,
                 uncertainty_threshold=0.5,
                 decoupling_threshold=0.25,
                 spectral_weight=0.7,
                 probability_weight=1.0):
        super(HybridSSUDClassifierFixed, self).__init__()
        self.model = model
        self.num_classes = num_classes

        # Fixed thresholds (no calibration)
        self.uncertainty_threshold = uncertainty_threshold
        self.decoupling_threshold = decoupling_threshold
        self.spectral_weight = spectral_weight
        self.probability_weight = probability_weight

        print(f"Initialized classifier with FIXED parameters:")
        print(f"  - Uncertainty threshold: {uncertainty_threshold}")
        print(f"  - Decoupling threshold: {decoupling_threshold}")
        print(f"  - Spectral weight: {spectral_weight}")

    def predict(self, x, return_uncertainties=False):
        """
        Hybrid prediction with fixed thresholds
        """
        with torch.no_grad():
            outputs = self.model(x)

            # Get probabilities and uncertainty measures
            probs = outputs['probs']
            max_probs, predicted_class = torch.max(probs, 1)

            # Get uncertainty measures
            combined_uncertainty = outputs['uncertainty_combined'].view(-1)

            # Calculate reliability scores (inverse of uncertainty)
            spectral_reliability = outputs['spectral_reliability'].view(-1)
            spatial_reliability = outputs['spatial_reliability'].view(-1)

            # Apply spatial-spectral decoupling for samples with significant difference
            reliability_diff = torch.abs(spectral_reliability - spatial_reliability)
            decoupling_mask = reliability_diff > self.decoupling_threshold

            # Initialize final uncertainty with combined uncertainty
            final_uncertainty = combined_uncertainty.clone()

            # For samples with significant difference, use the more reliable pathway
            use_spectral = spectral_reliability > spatial_reliability

            # For samples where spectral is more reliable
            spectral_indices = torch.where(decoupling_mask & use_spectral)[0]
            if len(spectral_indices) > 0:
                spectral_uncertainty = 1.0 - spectral_reliability[spectral_indices]
                scaled_spatial_uncertainty = 0.7 * (1.0 - spatial_reliability[spectral_indices])
                final_uncertainty[spectral_indices] = torch.maximum(spectral_uncertainty, scaled_spatial_uncertainty)

            # For samples where spatial is more reliable
            spatial_indices = torch.where(decoupling_mask & ~use_spectral)[0]
            if len(spatial_indices) > 0:
                spatial_uncertainty = 1.0 - spatial_reliability[spatial_indices]
                scaled_spectral_uncertainty = 0.7 * (1.0 - spectral_reliability[spatial_indices])
                final_uncertainty[spatial_indices] = torch.maximum(spatial_uncertainty, scaled_spectral_uncertainty)

            # Calculate rejection score using fixed weights
            rejection_score = self.spectral_weight * final_uncertainty + self.probability_weight * (1.0 - max_probs)

            # Detect unknowns when rejection score exceeds FIXED threshold
            unknown_mask = rejection_score > self.uncertainty_threshold

            # Create final predictions
            final_predictions = predicted_class.clone()
            final_predictions[unknown_mask] = self.num_classes  # Set to unknown class

            if return_uncertainties:
                return {
                    'predictions': final_predictions,
                    'raw_predictions': predicted_class,
                    'uncertainty': final_uncertainty,
                    'max_probs': max_probs,
                    'rejection_score': rejection_score,
                    'spectral_reliability': spectral_reliability,
                    'spatial_reliability': spatial_reliability,
                    'unknown_mask': unknown_mask,
                    'decoupling_mask': decoupling_mask
                }

            return final_predictions