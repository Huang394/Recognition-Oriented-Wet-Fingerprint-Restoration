import os
import math
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F

# ========== 1. 資料整理 ==========
def collect_image_paths(root_dir):
    image_dict = {}
    for subdir in os.listdir(root_dir):
        full_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(full_path):
            continue
        imgs = [os.path.join(full_path, f)
                for f in os.listdir(full_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if imgs:
            image_dict[subdir] = imgs
    return image_dict

def split_subjects(image_dict, train_ratio=0.85):
    all_keys = list(image_dict.keys())
    train_keys, val_keys = train_test_split(all_keys, train_size=train_ratio, random_state=42)
    return {"train": train_keys, "val": val_keys}

# ========== 2. Dataset ==========
class FingerprintDataset(Dataset):
    def __init__(self, image_dict, keys, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {k: i for i, k in enumerate(keys)}
        for k in keys:
            for p in image_dict[k]:
                self.samples.append((p, self.class_to_idx[k]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ========== 3. ArcFace Layer ==========
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

# ========== 4. 模型 ==========

class ArcFaceModel(nn.Module):
    def __init__(self, n_classes, emb_dim=512):
        super().__init__()
        # 使用 ImageNet 預訓練的 ResNet101
        base = models.resnet50(weights="IMAGENET1K_V1")
        
        # 移除最後一層 classification head
        self.feature = nn.Sequential(*list(base.children())[:-1])  # 去掉 fc
        
        # 全連接嵌入層 → embedding 維度
        self.embedding = nn.Linear(base.fc.in_features, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)

        # ArcFace head
        self.arcface = ArcMarginProduct(emb_dim, n_classes, s=64.0, m=0.6)

    def forward(self, x, label):
        x = self.feature(x)  # output shape: [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # flatten to [B, 2048]
        emb = self.bn(self.embedding(x))  # get normalized embedding
        logits = self.arcface(emb, label)  # apply margin-based classification
        return logits

class ArcFaceModelViT(nn.Module):
    def __init__(self, n_classes, emb_dim=512):
        super().__init__()
        base = models.vit_b_16(weights="IMAGENET1K_V1")
        vit_out_dim = base.heads.head.in_features
        base.heads = nn.Identity()
        self.feature = base
        self.embedding = nn.Linear(vit_out_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.arcface = ArcMarginProduct(emb_dim, n_classes, s=30.0, m=0.5)

    def forward(self, x, label):
        x = self.feature(x)
        emb = self.bn(self.embedding(x))
        logits = self.arcface(emb, label)
        return logits

# ========== 5. TAR@FAR 評估 ==========
def evaluate_tar_far(model, loader, device, n_trials=5000, fars=[0.1, 0.01, 0.001, 0.0001]):
    model.eval()
    embs, labs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            x = model.feature(imgs)
            x = x.view(x.size(0), -1)  # ✅ 展平
            emb = model.bn(model.embedding(x))
            embs.append(emb)
            labs.append(labels)
    embs = torch.cat(embs)
    labs = torch.cat(labs).cpu().numpy()
    pos, neg = [], []
    rng = np.random.RandomState(42)
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
    from sklearn.metrics import roc_curve
    sims = np.array(pos + neg)
    y = np.array([1]*len(pos) + [0]*len(neg))
    fpr, tpr, _ = roc_curve(y, sims)
    return {far: tpr[np.searchsorted(fpr, far) - 1] if np.searchsorted(fpr, far) > 0 else 0.0 for far in fars}

# ========== 6. 訓練主流程 ==========
def train_arcface(root_dir, epochs=100, batch_size=64, train_ratio=0.85, resume_from=None):
    output_dir = "./model_weights/resnet50_n2n_only_arcface_s0.64_m0.6"
    os.makedirs(output_dir, exist_ok=True)

    image_dict = collect_image_paths(root_dir)
    splits = split_subjects(image_dict, train_ratio)
    print(f"Train/Val classes: {len(splits['train'])}/{len(splits['val'])}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    datasets = {
        k: FingerprintDataset(image_dict, splits[k], transform)
        for k in ("train", "val")
    }
    loaders = {
        k: DataLoader(datasets[k], batch_size=batch_size, shuffle=(k=="train"),
                      num_workers=8, pin_memory=True)
        for k in ("train", "val")
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArcFaceModel(n_classes=len(splits['train'])).to(device)

    if resume_from and os.path.isfile(resume_from):
        model.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"Loaded checkpoint from {resume_from}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_tar = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in tqdm(loaders['train'], desc=f"Epoch {epoch} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs, labels)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tar = evaluate_tar_far(model, loaders['val'], device)
        print(f"Epoch {epoch} - TAR@FAR=0.1: {tar[0.1]:.4f}, 0.01: {tar[0.01]:.4f}, "
              f"0.001: {tar[0.001]:.4f}, 0.0001: {tar[0.0001]:.4f}")

        if tar[0.001] > best_tar:
            best_tar = tar[0.001]
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch}.pth"))
        scheduler.step()

if __name__ == "__main__":
    train_arcface("/home/formosa/formosa8T/fingerprint_recognition_v2/datasets/n2n_train/", epochs=100, batch_size=64, train_ratio=0.9)
