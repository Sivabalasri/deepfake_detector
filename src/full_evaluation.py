import os
import json
import torch
import open_clip
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    det_curve
)

from config import *
from model import HybridModel
from frequency_encoder import FrequencyEncoder
from evaluate import evaluate_dataset


# ============================================================
# Create Unique Experiment Folder
# ============================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = f"logs/experiment_{timestamp}"
os.makedirs(EXP_DIR, exist_ok=True)

print(f"\nSaving results to: {EXP_DIR}\n")


# ============================================================
# Load CLIP
# ============================================================

model_clip, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL,
    pretrained=CLIP_PRETRAIN
)

model_clip = model_clip.to(DEVICE).eval()


# ============================================================
# Load Saved Model
# ============================================================

model = HybridModel().to(DEVICE)
freq_encoder = FrequencyEncoder().to(DEVICE)

ckpt = torch.load(f"{MODEL_DIR}/best_model.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model"])
freq_encoder.load_state_dict(ckpt["freq"])

model.eval()
freq_encoder.eval()


# ============================================================
# Datasets
# ============================================================

datasets = {
    "FFPP": "data/ffpp",
    "DFDC": "data/dfdc",
    "CelebDF": "data/celeb_df",
    "WildFake": "data/wildfake"
}

final_results = {}


# ============================================================
# Evaluation Loop
# ============================================================

for name, path in datasets.items():

    print(f"\nEvaluating {name}...\n")

    result = evaluate_dataset(
        model_clip,       # 1
        preprocess,       # 2
        model,            # 3
        freq_encoder,     # 4
        path,             # 5
        name              # 6
    )

    labels = np.array(result["labels"])
    probs = np.array(result["probs"])
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)

    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    try:
        fpr95 = fpr[np.argmax(tpr >= 0.95)]
    except:
        fpr95 = 0.0

    cm = confusion_matrix(labels, preds)

    final_results[name] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc": float(auc),
        "average_precision": float(ap),
        "eer": float(eer),
        "fpr@95tpr": float(fpr95),
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# Save Metrics
# ============================================================

with open(f"{EXP_DIR}/metrics.json", "w") as f:
    json.dump(final_results, f, indent=4)

print("\nEvaluation complete.")
print(f"Results saved in {EXP_DIR}")