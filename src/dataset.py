import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import random
from io import BytesIO


# =========================================================
# DFDC-STYLE AUGMENTATION FUNCTION
# =========================================================

def strong_compression_augmentation(img):
    """
    Simulates DFDC-like distortions:
    - Random downscale & upscale
    - Strong JPEG compression
    """

    if random.random() < 0.7:

        # Random resize (simulate resolution change)
        scale = random.uniform(0.5, 1.0)
        w, h = img.size
        new_w = max(32, int(w * scale))
        new_h = max(32, int(h * scale))

        img = img.resize((new_w, new_h))
        img = img.resize((w, h))

        # Strong JPEG compression
        quality = random.randint(20, 60)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")

    return img


# =========================================================
# SINGLE SOURCE DATASET
# =========================================================

class FrameDataset(Dataset):
    def __init__(self, samples, preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        # Apply compression-style augmentation
        img = strong_compression_augmentation(img)

        clip_tensor = self.preprocess(img)

        # Frequency domain
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        fft = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log(np.abs(fft) + 1e-8)
        mag = cv2.resize(mag, (224, 224))
        fft_tensor = torch.tensor(mag).unsqueeze(0).float()

        return clip_tensor, fft_tensor, torch.tensor(label)


# =========================================================
# MULTI-SOURCE DATASET (FOR DANN TRAINING)
# =========================================================

class MultiSourceDataset(Dataset):
    def __init__(self, samples, preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, domain = self.samples[idx]

        img = Image.open(path).convert("RGB")

        # Apply DFDC-style augmentation
        img = strong_compression_augmentation(img)

        clip_tensor = self.preprocess(img)

        # Frequency domain
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        fft = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log(np.abs(fft) + 1e-8)
        mag = cv2.resize(mag, (224, 224))
        fft_tensor = torch.tensor(mag).unsqueeze(0).float()

        return (
            clip_tensor,
            fft_tensor,
            torch.tensor(label),
            torch.tensor(domain)
        )


# =========================================================
# LOAD SINGLE DATASET
# =========================================================

def load_and_split_dataset(root_dir, preprocess, split_ratio=0.8):

    samples = []

    for label, cls in enumerate(["real", "fake"]):
        folder = os.path.join(root_dir, cls)

        if not os.path.exists(folder):
            continue

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        samples += [(f, label) for f in files]

    np.random.shuffle(samples)

    split_index = int(len(samples) * split_ratio)

    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    return (
        FrameDataset(train_samples, preprocess),
        FrameDataset(val_samples, preprocess)
    )


# =========================================================
# LOAD MULTI-SOURCE DATASET
# =========================================================

def load_multi_source_dataset(dataset_roots, preprocess):

    samples = []

    for domain_id, root_dir in enumerate(dataset_roots):

        for label, cls in enumerate(["real", "fake"]):

            folder = os.path.join(root_dir, cls)

            if not os.path.exists(folder):
                continue

            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]

            samples += [
                (f, label, domain_id)
                for f in files
            ]

    np.random.shuffle(samples)

    return MultiSourceDataset(samples, preprocess)