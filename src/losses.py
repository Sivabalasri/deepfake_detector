import torch
import torch.nn.functional as F

def supervised_contrastive_loss(features, labels, temperature=0.07):
    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, features.T) / temperature

    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()

    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True))

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = -mean_log_prob_pos.mean()

    return loss