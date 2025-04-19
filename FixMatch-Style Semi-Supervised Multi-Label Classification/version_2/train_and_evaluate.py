import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, precision_score, recall_score, average_precision_score, label_ranking_loss, coverage_error
from utils.arff_parser import parse_arff_file
import pandas as pd
import os
import matplotlib.pyplot as plt

class MultiLabelDataset(Dataset):
    def __init__(self, X, Y, mask=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32) if mask is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mask is not None:
            return self.X[idx], self.Y[idx], self.mask[idx]
        else:
            return self.X[idx], self.Y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.norm(self.fc1(x)))
        return self.fc2(x)

def mask_labels(Y, ratio=0.1):
    num_samples = Y.shape[0]
    num_labeled = int(num_samples * ratio)
    labeled_idx = np.random.choice(num_samples, num_labeled, replace=False)
    mask = np.zeros_like(Y)
    mask[labeled_idx] = 1
    return Y * mask, mask

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_scores, all_targets = [], [], []
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.numpy()
            scores = torch.sigmoid(model(X)).cpu().numpy()
            preds = (scores > 0.5).astype(int)
            all_scores.append(scores)
            all_preds.append(preds)
            all_targets.append(Y)

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)
    y_score = np.vstack(all_scores)

    results = {
        'Micro-F1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'Macro-F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Precision': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'Subset Accuracy': accuracy_score(y_true, y_pred),
        'Average Precision': average_precision_score(y_true, y_score, average='macro'),
        'Hamming Loss': hamming_loss(y_true, y_pred),
        'Coverage': coverage_error(y_true, y_score) - 1,
        'Ranking Loss': label_ranking_loss(y_true, y_score),
        'One-error': (np.argmax(y_score, axis=1) != np.argmax(y_true, axis=1)).mean()
    }
    return results

def train_and_evaluate(dataset_name, dataset_path_or_cfg, label_count, split_type='random', epochs=10, batch_size=64, lr=1e-3):
    if split_type == 'random':
        attr, data = parse_arff_file(dataset_path_or_cfg)
        df = pd.DataFrame(data, columns=attr)
        X = df.iloc[:, :-label_count].values
        Y = df.iloc[:, -label_count:].values
        X_train, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    else:
        attr_tr, data_tr = parse_arff_file(dataset_path_or_cfg['train_file'])
        df_tr = pd.DataFrame(data_tr, columns=attr_tr)
        X_train = df_tr.iloc[:, :-label_count].values
        Y_train_full = df_tr.iloc[:, -label_count:].values

        attr_te, data_te = parse_arff_file(dataset_path_or_cfg['test_file'])
        df_te = pd.DataFrame(data_te, columns=attr_te)
        X_test = df_te.iloc[:, :-label_count].values
        Y_test = df_te.iloc[:, -label_count:].values

    Y_train_masked, mask_train = mask_labels(Y_train_full, ratio=0.1)
    train_dataset = MultiLabelDataset(X_train, Y_train_masked, mask_train)
    test_dataset = MultiLabelDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2)

    model = MLP(input_dim=X_train.shape[1], output_dim=label_count)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metric_per_epoch = []
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            X, Y, M = [b.to(device) for b in batch]
            X_u = X[~M.any(dim=1)]
            X_l, Y_l, M_l = X[M.any(dim=1)], Y[M.any(dim=1)], M[M.any(dim=1)]
            if len(X_u) == 0 or len(X_l) == 0:
                continue

            logits_l = model(X_l)
            loss_sup = F.binary_cross_entropy_with_logits(logits_l, Y_l, reduction='none')
            loss_sup = (loss_sup * M_l).mean()

            logits_u = model(X_u)
            pseudo_labels = (torch.sigmoid(logits_u) > 0.5).float()
            loss_unsup = F.binary_cross_entropy_with_logits(logits_u, pseudo_labels).mean()

            loss = loss_sup + 1.0 * loss_unsup
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, test_loader, device)
        metric_per_epoch.append(metrics)
        print(f"[{dataset_name}] Epoch {epoch+1}/{epochs}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # 可视化结果
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(12, 8))
    for i, key in enumerate(['Micro-F1', 'Macro-F1', 'Hamming Loss', 'Average Precision']):
        plt.subplot(2, 2, i+1)
        plt.plot([m[key] for m in metric_per_epoch], marker='o')
        plt.title(key)
        plt.xlabel("Epoch")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_metrics.png")

    return metric_per_epoch[-1]  # 返回最后一轮的指标
