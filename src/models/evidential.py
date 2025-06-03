import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super(EvidentialLayer, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # Just predict log-evidence directly to avoid exponential instability
        self.evidence = nn.Linear(in_features, num_classes)
        nn.init.normal_(self.evidence.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.evidence.bias, -1.0)  # Start with low evidence

    def forward(self, x):
        # Use relu to ensure non-negative evidence
        evidence = F.relu(self.evidence(x)) + 1e-6  # Small epsilon for stability

        # Alpha parameters (evidence + 1)
        alpha = evidence + 1.0

        # Total strength
        strength = torch.sum(alpha, dim=1, keepdim=True)

        # Expected probability under Dirichlet
        prob = alpha / strength

        # Simple uncertainty measure based on Dirichlet strength
        uncertainty = self.num_classes / strength

        return {
            'evidence': evidence,
            'alpha': alpha,
            'strength': strength,
            'prob': prob,
            'uncertainty': uncertainty
        }


def edl_mse_loss(target, alpha, epoch=1, lambda_reg=0.1):
    # Convert target to one-hot encoding
    y = F.one_hot(target, num_classes=alpha.size(1)).float()

    # Calculate strength (sum of alpha)
    S = torch.sum(alpha, dim=1, keepdim=True)

    # Expected probabilities under Dirichlet
    expected_p = alpha / S

    # MSE loss
    mse_loss = torch.sum((y - expected_p) ** 2, dim=1).mean()

    # Regularization to penalize high certainty (small S is uncertain, large S is certain)
    # We want to be uncertain unless we have evidence
    # Start with weak regularization and increase it
    annealing_coef = min(1.0, epoch / 10)

    # Regularize more on classes that should be zero
    zeros = (1 - y)
    reg_loss = torch.sum(zeros * alpha, dim=1).mean()

    # Weighted total loss
    total_loss = mse_loss + annealing_coef * lambda_reg * reg_loss

    return total_loss, mse_loss, reg_loss