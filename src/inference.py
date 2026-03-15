import torch
import open_clip
from PIL import Image
import numpy as np
import cv2
from src.config import *
from src.model import HybridModel
from src.frequency_encoder import FrequencyEncoder

# -------- LOAD ONCE --------
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_PRETRAIN
)
clip_model = clip_model.to(DEVICE).eval()

model = HybridModel().to(DEVICE)
freq = FrequencyEncoder().to(DEVICE)

ckpt = torch.load("D:/z_research/models/improved_LODO_version+adversial/lodo_dann_model.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model"])
freq.load_state_dict(ckpt["freq"])

model.eval()
freq.eval()


def predict_image(file):

    img = Image.open(file.file).convert("RGB")
    clip_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log(np.abs(fft)+1e-8)
    mag = cv2.resize(mag,(224,224))
    fft_tensor = torch.tensor(mag).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        clip_feat = clip_model.encode_image(clip_tensor)
        freq_feat = freq(fft_tensor)
        outputs = model(clip_feat, freq_feat)

        # If model returns 3 values (DANN version)
        if len(outputs) == 3:
            logits, _, _ = outputs
        else:
            logits, _ = outputs
        prob = torch.softmax(logits,1)[0,1].item()

    return {
        "prediction":"FAKE" if prob>0.5 else "REAL",
        "confidence":prob
    }

