import torch
import torch.nn.functional as F
from .losses import edl_mse_loss


def train_one_epoch_hybrid(model, classifier, train_loader, optimizer, epoch,
                           alpha=0.3, beta=0.5, gamma=0.3, delta=0.2, clip_value=1.0, print_freq=100):
    """
    Train for one epoch with unified loss function
    """
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_edl_loss = 0
    total_domain_loss = 0
    total_recon_loss = 0

    # For monitoring accuracy
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        from config.config import device
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        outputs['input_data'] = data

        # Classification loss (cross entropy)
        cls_loss = F.cross_entropy(outputs['logits'], target)

        # Stable EDL losses using MSE
        edl_combined, mse_combined, reg_combined = edl_mse_loss(
            target, outputs['alpha_combined'], epoch)

        edl_spectral, mse_spectral, reg_spectral = edl_mse_loss(
            target, outputs['alpha_spectral'], epoch)

        edl_spatial, mse_spatial, reg_spatial = edl_mse_loss(
            target, outputs['alpha_spatial'], epoch)

        # Combined EDL loss with proper weighting
        edl_total_loss = (edl_combined + alpha * (edl_spectral + edl_spatial)) / 3.0

        # Domain adversarial loss (from SIFD)
        domain_loss = F.binary_cross_entropy_with_logits(
            outputs['sifd_outputs']['domain_pred'],
            torch.ones_like(outputs['sifd_outputs']['domain_pred']) * 0.5  # Target 0.5 for maximum confusion
        )

        # Spectrum reconstruction loss
        sifd_outputs = outputs['sifd_outputs']
        data_flat = data.permute(0, 2, 3, 1).reshape(-1, data.shape[1])
        recon_loss = F.mse_loss(sifd_outputs['reconstructed_spectrum'], data_flat)

        # Total loss with all components
        loss = cls_loss + alpha * edl_total_loss + beta * domain_loss + gamma * recon_loss

        loss.backward()

        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_edl_loss += edl_total_loss.item()
        total_domain_loss += domain_loss.item()
        total_recon_loss += recon_loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs['logits'], 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if (batch_idx + 1) % print_freq == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx + 1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Cls: {cls_loss.item():.4f}, '
                  f'EDL: {edl_total_loss.item():.4f}, '
                  f'Domain: {domain_loss.item():.4f}, '
                  f'Recon: {recon_loss.item():.4f}, '
                  f'Acc: {100.0 * correct / total:.2f}%')

    # Calculate average metrics
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_edl_loss = total_edl_loss / len(train_loader)
    avg_domain_loss = total_domain_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_acc = 100.0 * correct / total

    print(f'Epoch {epoch} completed. '
          f'Avg Loss: {avg_loss:.4f}, Cls: {avg_cls_loss:.4f}, '
          f'EDL: {avg_edl_loss:.4f}, '
          f'Domain: {avg_domain_loss:.4f}, Recon: {avg_recon_loss:.4f}, '
          f'Acc: {avg_acc:.2f}%')

    return avg_loss, avg_acc


def validate_hybrid(model, classifier, val_loader, epoch):
    """
    Validate the model on the validation set
    """
    from config.config import device
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            # Get model outputs
            outputs = model(data)

            # Calculate cross entropy loss
            loss = F.cross_entropy(outputs['logits'], target)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs['logits'], 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    # Calculate average metrics
    avg_loss = val_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    print(f'Validation - Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy