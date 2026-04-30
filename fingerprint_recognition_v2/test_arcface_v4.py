import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import transforms, models
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import math


from train_arcface_v4 import  collect_image_paths, FingerprintDataset

class ArcFaceModel(nn.Module):
    def __init__(self, n_classes, emb_dim=512):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1")
        self.feature = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Linear(base.fc.in_features, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.arcface = ArcMarginProduct(emb_dim, n_classes, s=30.0, m=0.3)

    def forward(self, x, label):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        emb = self.bn(self.embedding(x))
        logits = self.arcface(emb, label)
        return logits, emb

class ArcFaceModelViT(nn.Module):
    def __init__(self, n_classes, emb_dim=512):
        super().__init__()
        base = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        base.heads = nn.Identity()  

        self.feature = base  
        self.embedding = nn.Linear(768, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.arcface = ArcMarginProduct(emb_dim, n_classes, s=30.0, m=0.3)

    def forward(self, x, label):
        x = self.feature(x)  # [B, 768]
        emb = self.bn(self.embedding(x))  # [B, emb_dim]
        logits = self.arcface(emb, label)
        return logits, emb


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        return output

def evaluate_tar_far(model, loader, device, n_trials=5000, fars=[0.1, 0.01, 0.001, 0.0001]):
    model.eval()
    embs, labs = [], []

    print("loading...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            x = model.feature(imgs)
            if x.ndim == 4:
                x = x.view(x.size(0), -1)
            emb = model.bn(model.embedding(x))
            embs.append(emb.cpu())
            labs.append(labels.cpu())

    embs = torch.cat(embs)
    labs = torch.cat(labs).numpy()
    pos, neg = [], []
    print("done loading...")
    rng = np.random.RandomState(42)
    print("start testing")
    for _ in range(n_trials):
        l = rng.choice(labs)
        idx = np.where(labs == l)[0]
        if len(idx) > 1:
            i1, i2 = rng.choice(idx, 2, replace=False)
            pos.append(F.cosine_similarity(embs[i1:i1+1], embs[i2:i2+1]).item())
        l1, l2 = rng.choice(labs, 2, replace=False)
        if l1 != l2:
            i1 = rng.choice(np.where(labs == l1)[0])
            i2 = rng.choice(np.where(labs == l2)[0])
            neg.append(F.cosine_similarity(embs[i1:i1+1], embs[i2:i2+1]).item())
    print("done testing")
    scores = np.array(pos + neg)
    labels = np.array([1] * len(pos) + [0] * len(neg))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    tar_results = {far: tpr[np.searchsorted(fpr, far) - 1] if np.searchsorted(fpr, far) > 0 else 0.0
                   for far in fars}


    return tar_results

def test_model(test_dir, model_path, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dict = collect_image_paths(test_dir)
    keys = list(image_dict.keys())
    print(f"Test classes: {len(keys)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = FingerprintDataset(image_dict, keys, transform)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=24, pin_memory=True)
    model = ArcFaceModel(n_classes=1440).to(device)

    state_dict = torch.load(model_path, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("arcface.weight")}
    model.load_state_dict(filtered_state_dict, strict=False)

    model.eval()
    print(f"Loaded model from {model_path}")

    tar = evaluate_tar_far(model, test_loader, device)
    print("TAR@FAR:")
    for k, v in tar.items():
        print(f"  FAR={k}: TAR={v:.4f}")

if __name__ == "__main__":
    test_model(
        test_dir="./datasets",
        model_path="./model_weights/resnet50_arcface_n9395_1023_v1_clean/best_model.pth",
        batch_size=64
    )
