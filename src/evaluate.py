# =========================
# COMPLETE RESEARCH-GRADE EVALUATION
# =========================

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import *
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from scipy.stats import bootstrap
from statsmodels.stats.contingency_tables import mcnemar

from config import *
from dataset import FrameDataset
from model import HybridModel
from frequency_encoder import FrequencyEncoder
import open_clip


# =========================================================
# Create Timestamped Results Folder
# =========================================================

BASE_RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(BASE_RESULTS, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join(BASE_RESULTS, f"experiment_{timestamp}")
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# Utility Functions
# =========================================================

def compute_eer(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])

def compute_fpr95(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    idx = np.argmin(np.abs(tpr - 0.95))
    return float(fpr[idx])

def bootstrap_ci(metric_func, labels, preds_or_probs):
    res = bootstrap(
        (labels, preds_or_probs),
        lambda l, p: metric_func(l, p),
        confidence_level=0.95,
        n_resamples=1000,
        method="percentile"
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


# =========================================================
# Evaluate Dataset
# =========================================================

def evaluate_dataset(model_clip, preprocess, model, freq_encoder, dataset_path, name):

    samples = []
    for label, cls in enumerate(["real", "fake"]):
        folder = os.path.join(dataset_path, cls)
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        samples += [(f, label) for f in files]

    dataset = FrameDataset(samples, preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    all_probs, all_labels, all_preds, all_features = [], [], [], []

    model.eval()
    freq_encoder.eval()

    with torch.no_grad():
        for clip_tensor, fft_tensor, labels in tqdm(loader, desc=f"Evaluating {name}"):

            clip_tensor = clip_tensor.to(DEVICE)
            fft_tensor = fft_tensor.to(DEVICE)
            labels = labels.to(DEVICE)

            clip_feat = model_clip.encode_image(clip_tensor)
            freq_feat = freq_encoder(fft_tensor)
            #logits, features = model(clip_feat, freq_feat)
            logits, _, features = model(clip_feat, freq_feat)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_features.extend(features.cpu().numpy())

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    features = np.array(all_features)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1_score": float(f1_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "sensitivity": float(tp / (tp + fn + 1e-8)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
        "auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "eer": compute_eer(labels, probs),
        "fpr@95tpr": compute_fpr95(labels, probs),
        "confusion_matrix": cm.tolist()
    }

    # ========================= BOOTSTRAP CI =========================
    results["accuracy_ci"] = bootstrap_ci(accuracy_score, labels, preds)
    results["f1_ci"] = bootstrap_ci(f1_score, labels, preds)
    results["auc_ci"] = bootstrap_ci(roc_auc_score, labels, probs)

    # ========================= SAVE CURVES =========================
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC - {name}")
    plt.savefig(os.path.join(RESULT_DIR, f"roc_{name}.png"))
    plt.close()

    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(recall_curve, precision_curve)
    plt.title(f"PR - {name}")
    plt.savefig(os.path.join(RESULT_DIR, f"pr_{name}.png"))
    plt.close()

    # Confusion
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion - {name}")
    plt.savefig(os.path.join(RESULT_DIR, f"confusion_{name}.png"))
    plt.close()

    with open(os.path.join(RESULT_DIR, f"{name}_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results, features, labels


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("Using device:", DEVICE)

    model_clip, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAIN
    )
    model_clip = model_clip.to(DEVICE).eval()

    model = HybridModel().to(DEVICE)
    freq_encoder = FrequencyEncoder().to(DEVICE)

    ckpt = torch.load(f"{MODEL_DIR}/lodo_dann_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    freq_encoder.load_state_dict(ckpt["freq"])

    datasets = {
        "FFPP": "data/ffpp",
        "DFDC": "data/dfdc",
        "CelebDF": "data/celeb_df",
        "WildFake": "data/wildfake"
    }

    all_results = {}
    all_features = []
    all_labels = []

    for name, path in datasets.items():
        res, feats, labels = evaluate_dataset(
            model_clip, preprocess, model, freq_encoder, path, name
        )
        all_results[name] = res
        all_features.append(feats)
        all_labels.append(labels)

    # ========================= t-SNE =========================
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(combined_features)

    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:,0], reduced[:,1], c=combined_labels, cmap="coolwarm")
    plt.title("t-SNE")
    plt.savefig(os.path.join(RESULT_DIR, "tsne.png"))
    plt.close()

    # ========================= SAVE SUMMARY =========================
    df = pd.DataFrame(all_results).T
    df.to_csv(os.path.join(RESULT_DIR, "summary.csv"))

    with open(os.path.join(RESULT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    # ========================= LaTeX Table =========================
    latex_table = df.round(4).to_latex()
    with open(os.path.join(RESULT_DIR, "results_table.tex"), "w") as f:
        f.write(latex_table)

    print("\nEvaluation complete.")
    print("Results saved in:", RESULT_DIR)