import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
CURVE_DIR = "logs/curves"
os.makedirs(CURVE_DIR, exist_ok=True)

# -------------------------
# TRAIN vs VAL LOSS
# -------------------------
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training vs Validation Loss")
    plt.savefig(f"{CURVE_DIR}/train_val_loss.png")
    plt.close()

# -------------------------
# ACCURACY CURVE
# -------------------------
def plot_accuracy(train_acc, val_acc):
    plt.figure(figsize=(8,6))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title("Training vs Validation Accuracy")
    plt.savefig(f"{CURVE_DIR}/train_val_accuracy.png")
    plt.close()

# -------------------------
# ROC CURVE
# -------------------------
def plot_roc(labels, probs, name="validation"):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{CURVE_DIR}/roc_{name}.png")
    plt.close()

# -------------------------
# PRECISION RECALL
# -------------------------
def plot_pr(labels, probs, name="validation"):
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.grid(True)
    plt.savefig(f"{CURVE_DIR}/pr_{name}.png")
    plt.close()

# -------------------------
# CONFUSION MATRIX
# -------------------------
def plot_confusion(labels, preds, name):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"{CURVE_DIR}/confusion_{name}.png")
    plt.close()

# -------------------------
# SCORE DISTRIBUTION
# -------------------------
def plot_score_distribution(labels, probs, name):
    plt.figure(figsize=(8,6))
    sns.histplot(np.array(probs)[np.array(labels)==0], color="blue", label="Real", kde=True)
    sns.histplot(np.array(probs)[np.array(labels)==1], color="red", label="Fake", kde=True)
    plt.legend()
    plt.title("Score Distribution")
    plt.savefig(f"{CURVE_DIR}/score_distribution_{name}.png")
    plt.close()

# -------------------------
# t-SNE FEATURE PLOT
# -------------------------
def plot_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap="coolwarm")
    plt.colorbar(scatter)
    plt.title("t-SNE Feature Distribution")
    plt.savefig(f"{CURVE_DIR}/tsne.png")
    plt.close()