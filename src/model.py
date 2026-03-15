import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# =========================================================
# Gradient Reversal Layer
# =========================================================

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# =========================================================
# Hybrid Model (Stable Version)
# =========================================================

class HybridModel(nn.Module):
    def __init__(self, clip_dim=512, freq_dim=64, num_domains=3):
        super().__init__()

        # Slightly smaller projection (less overfitting)
        self.projection = nn.Sequential(
            nn.Linear(clip_dim + freq_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.classifier = nn.Linear(128, 2)

        self.grl = GradientReversalLayer(lambda_=1.0)

        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, clip_feat, freq_feat):

        x = torch.cat([clip_feat, freq_feat], dim=1)

        features = self.projection(x)

        # Normalize features (important for stability)
        features = F.normalize(features, dim=1)

        logits = self.classifier(features)

        reversed_features = self.grl(features)
        domain_logits = self.domain_classifier(reversed_features)

        return logits, domain_logits, features