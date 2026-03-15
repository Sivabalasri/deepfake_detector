import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_MODEL = "ViT-B-16"
CLIP_PRETRAIN = "laion2b_s34b_b88k"

#BATCH_SIZE = 32
BATCH_SIZE = 64
#EPOCHS = 20
EPOCHS = 15
LR = 1e-4
SPLIT_RATIO = 0.8

LOG_DIR = "logs"
CURVE_DIR = os.path.join(LOG_DIR, "curves")
MODEL_DIR = "models"

os.makedirs(CURVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

