import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class SpectrumInvariantFrequencyDisentanglement(nn.Module):
    """
    Domain generalization module that disentangles domain-invariant from
    domain-specific spectral features in the frequency domain.
    """

    def __init__(self, input_channels, feature_dim=64):
        super(SpectrumInvariantFrequencyDisentanglement, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        # Calculate expected frequency domain size
        self.freq_size = input_channels // 2 + 1

        # Update freq_encoder to handle 2 channels (real + imaginary parts)
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),  # Changed from 1 to 2 input channels
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        # Domain-invariant extractor
        self.invariant_extractor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        # Attention mechanism (simplified)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # Domain adversarial component (previously missing)
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Decoder back to original spectral space
        self.spectrum_decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),  # Added intermediate layer for better reconstruction
            nn.ReLU(),
            nn.Linear(128, input_channels),
            nn.Sigmoid()  # Keep values in reasonable range
        )

        # Project to original input space (for modulation)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, input_channels),
            nn.Sigmoid()  # Keep values in [0,1] range for adding back
        )

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        num_pixels = h * w

        # Reshape to process each pixel's spectrum
        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_pixels, 1, channels)

        # Convert to frequency domain using both real and imaginary parts
        x_freq_complex = torch.fft.rfft(x_flat, dim=2)
        x_freq_real = x_freq_complex.real
        x_freq_imag = x_freq_complex.imag

        # Stack real and imaginary components as channels
        x_freq = torch.cat([x_freq_real, x_freq_imag], dim=1)

        # Store original magnitude for visualization
        x_freq_mag = torch.sqrt(x_freq_real ** 2 + x_freq_imag ** 2)

        # Encode frequency representations
        freq_features = self.freq_encoder(x_freq)

        # Extract invariant features
        invariant_features = self.invariant_extractor(freq_features)

        # Apply attention
        att_weights = self.attention(invariant_features)
        weighted_features = invariant_features * att_weights

        # Compute domain prediction (for adversarial loss)
        domain_features = weighted_features.mean(dim=2)
        domain_pred = self.domain_classifier(domain_features)

        # Pool across frequency dimension for reconstruction
        pooled_features = F.adaptive_avg_pool1d(weighted_features, 1).squeeze(2)

        # Create reconstruction from frequency features
        reconstructed_spectrum = self.spectrum_decoder(pooled_features)

        # Project features to modulation
        modulation = self.projector(pooled_features)

        # Reshape modulation to match original input shape
        modulation = modulation.view(batch_size, h, w, channels).permute(0, 3, 1, 2)

        # Prepare visualization data
        freq_info = {
            'magnitude': x_freq_mag.reshape(batch_size * num_pixels, -1)
        }

        return {
            'invariant_features': modulation,
            'domain_pred': domain_pred,
            'reconstructed_spectrum': reconstructed_spectrum,
            'original_shape': (batch_size, h, w),
            'freq_info': freq_info
        }