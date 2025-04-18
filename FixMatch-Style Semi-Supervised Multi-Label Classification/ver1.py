import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, hamming_loss
from skmultilearn.dataset import load_from_arff

# ===== 1. 数据预处理 =====
def load_bibtex(train_path, test_path, num_labels=159):
    X_train, Y_train = load_from_arff(train_path, label_count=num_labels)
    X_test, Y_test = load_from_arff(test_path, label_count=num_labels)
    return X_train.toarray(), Y_train.toarray(), X_test.toarray(), Y_test.toarray()

def mask_labels(Y, ratio=0.1):
    num_samples = Y.shape[0]
    num_labeled = int(num_samples * ratio)
    labeled_idx = np.random.choice(num_samples, num_labeled, replace=False)
    mask = np.zeros_like(Y)
    mask[labeled_idx] = 1
    return Y * mask, mask

class BibtexDataset(Dataset):
    def __init__(self, X, Y, mask=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32) if mask is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.mask is not None:
            return self.X[idx], self.Y[idx], self.mask[idx]
        else:
            return self.X[idx], self.Y[idx]

# ===== 2. 模型结构 =====
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # ✅ 换成 LayerNorm
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))  # ✅ 调整为 ln1
        return self.fc2(x)

# ===== 3. 训练函数（半监督一致性） =====
def train_step(model, x_l, y_l, mask_l, x_u, lambda_u=1.0, threshold=0.5):
    model.train()
    logits_l = model(x_l)
    loss_sup = F.binary_cross_entropy_with_logits(logits_l, y_l, reduction='none')
    loss_sup = (loss_sup * mask_l).mean()

    logits_u = model(x_u)
    pseudo_labels = (torch.sigmoid(logits_u) > threshold).float()
    loss_unsup = F.binary_cross_entropy_with_logits(logits_u, pseudo_labels).mean()

    return loss_sup + lambda_u * loss_unsup

# ===== 4. 评估函数 =====
from sklearn.metrics import (
    f1_score, hamming_loss, accuracy_score, precision_score, recall_score,
    average_precision_score, label_ranking_loss, coverage_error
)
import numpy as np

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
        'Micro-F1': f1_score(y_true, y_pred, average='micro'),
        'Macro-F1': f1_score(y_true, y_pred, average='macro'),
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

# ===== 5. 主程序 =====
if __name__ == "__main__":
    # 路径请根据你本地修改
    path_train = "D:/6405/dataset/Bibtex-RandomTrainTest-Mulan/Bibtex-train.arff"
    path_test = "D:/6405/dataset/Bibtex-RandomTrainTest-Mulan/Bibtex-test.arff"

    X_train, Y_train_full, X_test, Y_test = load_bibtex(path_train, path_test)
    Y_train_masked, mask_train = mask_labels(Y_train_full, ratio=0.1)

    train_dataset = BibtexDataset(X_train, Y_train_masked, mask_train)
    test_dataset = BibtexDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model = MLP(input_dim=X_train.shape[1], output_dim=Y_train_full.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics_per_epoch = []

    # Train
    for epoch in range(10):
        for X, Y, M in train_loader:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            X_u = X[~M.any(dim=1)]  # 无标签样本
            X_l, Y_l, M_l = X[M.any(dim=1)], Y[M.any(dim=1)], M[M.any(dim=1)]
            if len(X_u) == 0 or len(X_l) == 0: continue

            loss = train_step(model, X_l, Y_l, M_l, X_u)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scores = evaluate(model, test_loader, device)
        metrics_per_epoch.append(scores)

        print(f"\nEpoch {epoch + 1} Evaluation Results:")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")



import matplotlib.pyplot as plt

# 准备数据
epochs = list(range(1, len(metrics_per_epoch) + 1))
micro_f1 = [m['Micro-F1'] for m in metrics_per_epoch]
macro_f1 = [m['Macro-F1'] for m in metrics_per_epoch]
hamming = [m['Hamming Loss'] for m in metrics_per_epoch]
average_precision = [m['Average Precision'] for m in metrics_per_epoch]
subset_acc = [m['Subset Accuracy'] for m in metrics_per_epoch]

# 创建子图布局
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Micro-F1
axs[0, 0].plot(epochs, micro_f1, marker='o')
axs[0, 0].set_title("Micro-F1")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Score")
axs[0, 0].grid(True)

# Macro-F1
axs[0, 1].plot(epochs, macro_f1, marker='o', color='orange')
axs[0, 1].set_title("Macro-F1")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Score")
axs[0, 1].grid(True)

# Hamming Loss
axs[1, 0].plot(epochs, hamming, marker='o', color='green')
axs[1, 0].set_title("Hamming Loss")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Loss")
axs[1, 0].grid(True)

# Average Precision
axs[1, 1].plot(epochs, average_precision, marker='o', color='purple')
axs[1, 1].set_title("Average Precision")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Score")
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig("multimetric_curves.png")
plt.show()
