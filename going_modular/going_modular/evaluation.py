import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, dataloader, class_names, device):
    """
    Evaluates a trained PyTorch model on a test DataLoader.
    
    Returns:
        results = {
            "accuracy": float,
            "classification_report": str,
            "specificity": dict,
            "confusion_matrix": np.ndarray,
            "labels": list,
            "preds": list
        }
    """
    
    model.eval()
    all_preds = []
    all_labels = []

    # Collect predictions and labels
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # ----- Accuracy -----
    accuracy = accuracy_score(all_labels, all_preds)

    # ----- Classification Report -----
    cls_report = classification_report(all_labels, all_preds, target_names=class_names)

    # ----- Confusion Matrix -----
    cm = confusion_matrix(all_labels, all_preds)

    # ----- Specificity per class -----
    specificities = {}
    for i, class_name in enumerate(class_names):
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        FP = cm[:, i].sum() - cm[i, i]
        specificity = TN / (TN + FP)
        specificities[class_name] = float(specificity)

    # Pack results
    results = {
        "accuracy": float(accuracy),
        "classification_report": cls_report,
        "specificity": specificities,
        "confusion_matrix": cm,
        "labels": all_labels,
        "preds": all_preds
    }

    return results

def plot_results(results):
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(14,5))

    # ---- LOSS ----
    plt.subplot(1,2,1)
    plt.plot(epochs, results["train_loss"], label="Train Loss")
    plt.plot(epochs, results["test_loss"], label="Test Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    # ---- ACCURACY ----
    plt.subplot(1,2,2)
    plt.plot(epochs, results["train_acc"], label="Train Accuracy")
    plt.plot(epochs, results["test_acc"], label="Test Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
