import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def test_open_set_hybrid(model, classifier, test_loader, known_classes=7):
    """
    Test the model for open set recognition.

    Returns:
        known_acc: Accuracy on known classes
        unknown_rejection: Percentage of unknown samples correctly rejected
        hos_score: Harmonic mean of known accuracy and unknown rejection
    """
    from src.config.config import device, new_classes
    model.eval()

    # Metrics
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0

    all_predictions = []
    all_targets = []

    # For analyzing distributions
    known_uncertainties = []
    unknown_uncertainties = []
    rejection_scores_known = []
    rejection_scores_unknown = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Get predictions with unknown detection
            results = classifier.predict(data, return_uncertainties=True)
            predictions = results['predictions']

            # Split metrics for known and unknown classes
            known_mask = target < known_classes
            unknown_mask = target >= known_classes

            # Store uncertainty metrics for analysis
            if known_mask.sum() > 0:
                known_uncertainties.extend(results['uncertainty'][known_mask].cpu().numpy())
                rejection_scores_known.extend(results['rejection_score'][known_mask].cpu().numpy())

            if unknown_mask.sum() > 0:
                unknown_uncertainties.extend(results['uncertainty'][unknown_mask].cpu().numpy())
                rejection_scores_unknown.extend(results['rejection_score'][unknown_mask].cpu().numpy())

            # Calculate metrics for known classes (correct if prediction matches target)
            if known_mask.sum() > 0:
                known_correct += (predictions[known_mask] == target[known_mask]).sum().item()
                known_total += known_mask.sum().item()

            # Calculate metrics for unknown classes (correct if predicted as unknown)
            if unknown_mask.sum() > 0:
                unknown_correct += (predictions[unknown_mask] == known_classes).sum().item()
                unknown_total += unknown_mask.sum().item()

            # Store for confusion matrix
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calculate final metrics
    known_acc = 100.0 * known_correct / known_total if known_total > 0 else 0
    unknown_rejection = 100.0 * unknown_correct / unknown_total if unknown_total > 0 else 0

    # Harmonic mean of known accuracy and unknown rejection (HOS)
    hos_score = 2 * (known_acc * unknown_rejection) / (known_acc + unknown_rejection) if (
                                                                                                     known_acc + unknown_rejection) > 0 else 0

    print(f'Test Results:')
    print(f'Known Class Accuracy: {known_acc:.2f}%')
    print(f'Unknown Class Rejection: {unknown_rejection:.2f}%')
    print(f'HOS Score: {hos_score:.2f}')

    # Print uncertainty statistics
    if known_uncertainties and unknown_uncertainties:
        print(f'\nUncertainty Analysis:')
        print(f'Known classes: mean={np.mean(known_uncertainties):.3f}, std={np.std(known_uncertainties):.3f}')
        print(f'Unknown classes: mean={np.mean(unknown_uncertainties):.3f}, std={np.std(unknown_uncertainties):.3f}')
        print(
            f'Rejection score known: mean={np.mean(rejection_scores_known):.3f}, std={np.std(rejection_scores_known):.3f}')
        print(
            f'Rejection score unknown: mean={np.mean(rejection_scores_unknown):.3f}, std={np.std(rejection_scores_unknown):.3f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)

    # Detailed classification report
    print("\nClassification Report:")
    target_names = [f"{new_classes[i + 1]}" for i in range(known_classes)] + ["Unknown"]
    print(classification_report(all_targets, all_predictions, target_names=target_names))

    return known_acc, unknown_rejection, hos_score, cm


def test_open_set_hybrid_dioni(model, classifier, test_loader, known_classes=9):
    """
    Test the model for open set recognition on Dioni/Loukia datasets.

    Returns:
        known_acc: Accuracy on known classes (1-9)
        unknown_rejection: Percentage of unknown samples (class 10) correctly rejected
        hos_score: Harmonic mean of known accuracy and unknown rejection
    """
    from src.config.config import device, new_classesDioni
    model.eval()

    # Metrics
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0

    all_predictions = []
    all_targets = []

    # For analyzing distributions
    known_uncertainties = []
    unknown_uncertainties = []
    rejection_scores_known = []
    rejection_scores_unknown = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Get predictions with unknown detection
            results = classifier.predict(data, return_uncertainties=True)
            predictions = results['predictions']

            ## Split metrics for known (1-9 -> 0-8) and unknown (10 -> 9) classes
            known_mask = target < known_classes  # Classes 0-8 (mapped from 1-9)
            unknown_mask = target >= known_classes  # Class 9 (mapped from 10)

            # Store uncertainty metrics for analysis
            if known_mask.sum() > 0:
                known_uncertainties.extend(results['uncertainty'][known_mask].cpu().numpy())
                rejection_scores_known.extend(results['rejection_score'][known_mask].cpu().numpy())

            if unknown_mask.sum() > 0:
                unknown_uncertainties.extend(results['uncertainty'][unknown_mask].cpu().numpy())
                rejection_scores_unknown.extend(results['rejection_score'][unknown_mask].cpu().numpy())

            # Calculate metrics for known classes (correct if prediction matches target)
            if known_mask.sum() > 0:
                known_correct += (predictions[known_mask] == target[known_mask]).sum().item()
                known_total += known_mask.sum().item()

            # Calculate metrics for unknown classes (correct if predicted as unknown)
            if unknown_mask.sum() > 0:
                unknown_correct += (predictions[unknown_mask] == known_classes).sum().item()
                unknown_total += unknown_mask.sum().item()

            # Store for confusion matrix
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calculate final metrics
    known_acc = 100.0 * known_correct / known_total if known_total > 0 else 0
    unknown_rejection = 100.0 * unknown_correct / unknown_total if unknown_total > 0 else 0

    # Harmonic mean of known accuracy and unknown rejection (HOS)
    hos_score = 2 * (known_acc * unknown_rejection) / (known_acc + unknown_rejection) if (known_acc + unknown_rejection) > 0 else 0

    print(f'Test Results:')
    print(f'Known Class Accuracy: {known_acc:.2f}%')
    print(f'Unknown Class Rejection: {unknown_rejection:.2f}%')
    print(f'HOS Score: {hos_score:.2f}')

    # Print uncertainty statistics
    if known_uncertainties and unknown_uncertainties:
        print(f'\nUncertainty Analysis:')
        print(f'Known classes: mean={np.mean(known_uncertainties):.3f}, std={np.std(known_uncertainties):.3f}')
        print(f'Unknown classes: mean={np.mean(unknown_uncertainties):.3f}, std={np.std(unknown_uncertainties):.3f}')
        print(f'Rejection score known: mean={np.mean(rejection_scores_known):.3f}, std={np.std(rejection_scores_known):.3f}')
        print(f'Rejection score unknown: mean={np.mean(rejection_scores_unknown):.3f}, std={np.std(rejection_scores_unknown):.3f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)

    # Detailed classification report
    print("\nClassification Report:")
    target_names = [f"{new_classesDioni[i+1]}" for i in range(known_classes)] + ["Unknown"]
    print(classification_report(all_targets, all_predictions, target_names=target_names))

    return known_acc, unknown_rejection, hos_score, cm

def collect_model_outputs(model, data_loader, device):
    """Collect model outputs for threshold calibration"""
    model.eval()
    uncertainties = []
    confidences = []
    entropies = []
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)

            # Get uncertainty measures
            uncertainty = outputs['uncertainty_combined'].view(-1).cpu().numpy()
            probs = outputs['probs'].cpu().numpy()
            max_probs = np.max(probs, axis=1)

            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
            normalized_entropy = entropy / np.log(probs.shape[1])

            # Store predictions and values
            pred = np.argmax(probs, axis=1)

            uncertainties.extend(uncertainty)
            confidences.extend(max_probs)
            entropies.extend(normalized_entropy)
            predictions.extend(pred)
            targets.extend(target.cpu().numpy())

    return {
        'uncertainty': np.array(uncertainties),
        'confidence': np.array(confidences),
        'entropy': np.array(entropies),
        'predictions': np.array(predictions),
        'targets': np.array(targets)
    }

def evaluate_subset(classifier, data_loader, device):
    """
    Evaluate classifier on a subset (for known class accuracy)
    """
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            predictions = classifier.predict(data)
            correct += (predictions == target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total if total > 0 else 0
    return accuracy, correct, total

def evaluate_unknown_rejection(classifier, data_loader, device, num_classes):
    """
    Evaluate classifier for unknown class rejection
    """
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            predictions = classifier.predict(data)
            correct += (predictions == num_classes).sum().item()  # Predicted as unknown
            total += target.size(0)

    rejection_rate = 100.0 * correct / total if total > 0 else 0
    return rejection_rate, correct, total