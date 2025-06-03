import torch
import torch.nn.functional as F


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