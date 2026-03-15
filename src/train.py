import os
import torch
import open_clip
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score

from config import *
from dataset import load_multi_source_dataset
from model import HybridModel
from frequency_encoder import FrequencyEncoder
from losses import supervised_contrastive_loss
from visualization import plot_loss, plot_accuracy

print("Using device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)


# =========================================================
# FOCAL LOSS
# =========================================================

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal

        return focal.mean()


# =========================================================
# TRAIN FUNCTION
# =========================================================

def train():

    # -------------------------
    # Load CLIP Backbone (Frozen)
    # -------------------------
    model_clip, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=CLIP_PRETRAIN
    )
    model_clip = model_clip.to(DEVICE).eval()

    for p in model_clip.parameters():
        p.requires_grad = False

    # -------------------------
    # Multi-Source Dataset
    # -------------------------
    train_roots = [
        "data/ffpp",
        "data/celeb_df",
        "data/wildfake"
    ]

    train_ds = load_multi_source_dataset(train_roots, preprocess)

    # -------------------------
    # Domain Balanced Sampling
    # -------------------------
    domain_labels = [sample[2] for sample in train_ds.samples]
    domain_counts = Counter(domain_labels)

    sample_weights = [1.0 / domain_counts[d] for d in domain_labels]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,   # IMPORTANT for Windows
        pin_memory=True
    )

    # -------------------------
    # Model + Optimizer
    # -------------------------
    model = HybridModel().to(DEVICE)
    freq_encoder = FrequencyEncoder().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(freq_encoder.parameters()),
        lr=LR
    )

    scaler = GradScaler()

    # -------------------------
    # Class Weights (Focal Loss)
    # -------------------------
    class_labels = [sample[1] for sample in train_ds.samples]
    class_counts = Counter(class_labels)

    total = class_counts[0] + class_counts[1]
    weight_real = total / (2 * class_counts[0])
    weight_fake = total / (2 * class_counts[1])

    class_weights = torch.tensor(
        [weight_real, weight_fake],
        dtype=torch.float32
    ).to(DEVICE)

    print("Class Weights:", class_weights)

    criterion = FocalLoss(alpha=class_weights, gamma=2)

    # -------------------------
    # Training Loop
    # -------------------------
    train_losses = []
    train_accs = []

    for epoch in range(EPOCHS):

        model.train()
        freq_encoder.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        for clip_tensor, fft_tensor, label, domain_label in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{EPOCHS}"
        ):

            clip_tensor = clip_tensor.to(DEVICE)
            fft_tensor = fft_tensor.to(DEVICE)
            label = label.to(DEVICE)
            domain_label = domain_label.to(DEVICE)

            with torch.no_grad():
                clip_feat = model_clip.encode_image(clip_tensor)
                clip_feat = torch.nn.functional.normalize(clip_feat, dim=1)

            optimizer.zero_grad()

            with autocast():

                freq_feat = freq_encoder(fft_tensor)

                logits, domain_logits, features = model(
                    clip_feat,
                    freq_feat
                )

                # Fake/Real classification loss
                cls_loss = criterion(logits, label)

                # Domain adversarial loss
                domain_loss = torch.nn.functional.cross_entropy(
                    domain_logits,
                    domain_label
                )

                # Contrastive loss
                contrast_loss = supervised_contrastive_loss(
                    features,
                    label
                )

                # Variance regularization
                feature_var = torch.var(features, dim=0).mean()
                variance_loss = torch.relu(1.0 - feature_var)

                # Final loss
                loss = (
                    cls_loss
                    + 0.2 * contrast_loss
                    + 0.1 * variance_loss
                    + 0.1 * domain_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {train_loss:.4f} | "
            f"Accuracy: {train_acc:.4f}"
        )

    # -------------------------
    # Save Model
    # -------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "freq": freq_encoder.state_dict()
        },
        f"{MODEL_DIR}/lodo_dann_model.pth"
    )

    print("\nModel saved to:", f"{MODEL_DIR}/lodo_dann_model.pth")

    plot_loss(train_losses, train_losses)
    plot_accuracy(train_accs, train_accs)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train()