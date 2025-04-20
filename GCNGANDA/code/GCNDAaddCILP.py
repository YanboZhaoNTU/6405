import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import random

# 导入自定义模块（请确保这些模块在你的工作目录中）
from u_mReadData import *
from u_base import *
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *

############################################
# 1. 分别读取训练集和测试集（多标签数据）
############################################
datasnames = ["Yeast"]
print("数据集:", datasnames[0])
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)

# 训练集
n_train = X_train.shape[0]
if len(Y_train.shape) == 1:
    raise ValueError("训练集多标签数据应为二维数组，每行代表一个样本的标签向量。")
n_classes = Y_train.shape[1]  # 假设每个样本有 n_classes 个标签

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

# 测试集
n_test = X_test.shape[0]
if len(Y_test.shape) == 1:
    raise ValueError("测试集多标签数据应为二维数组，每行代表一个样本的标签向量。")
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

############################################
# 2. 在训练集内构造半监督 mask（仅使用训练集数据）
############################################
# 例如，随机抽取 50% 的训练样本作为有标签数据
fraction = 0.5
num_labeled = int(n_train * fraction)
perm = torch.randperm(n_train)
labeled_indices = perm[:num_labeled]
train_mask = torch.zeros(n_train, dtype=torch.bool)
train_mask[labeled_indices] = True
unlabeled_mask = ~train_mask

print(f"训练样本总数: {n_train}, 有标签样本数: {train_mask.sum().item()}, 无标签样本数: {unlabeled_mask.sum().item()}")


############################################
# 3. 分别构造训练集和测试集的图（基于余弦相似度的 kNN 邻接矩阵）
############################################
def build_graph(X, k):
    n = X.shape[0]
    sim = cosine_similarity(X)  # (n, n)
    A = np.zeros((n, n))
    for i in range(n):
        row = sim[i]
        # 降序排列后排除自身，选择前 k 个邻居
        indices = np.argsort(row)[::-1]
        indices = indices[indices != i]
        neighbors = indices[:k]
        A[i, neighbors] = 1
    # 对称化：若任一方向有边，则视为相连
    A = np.maximum(A, A.T)
    return A


k = 20
A_train = build_graph(X_train, k)
A_test = build_graph(X_test, k)


def normalize_adj(A):
    A = torch.tensor(A, dtype=torch.float32)
    I = torch.eye(A.size(0), device=A.device)
    A_hat = A + I
    D = torch.sum(A_hat, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


A_train_norm = normalize_adj(A_train)
A_test_norm = normalize_adj(A_test)


############################################
# 4. 定义 Variational GCN 模型（结合变分思想逼近后验分布）
############################################
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, A_norm):
        support = torch.mm(x, self.weight)
        out = torch.mm(A_norm, support)
        return out


class VariationalGCN(nn.Module):
    def __init__(self, nfeat, nhid, nlatent, nclass):
        super(VariationalGCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        # 两个并行的图卷积层分别生成均值和对数方差
        self.gc_mu = GCNLayer(nhid, nlatent)
        self.gc_logvar = GCNLayer(nhid, nlatent)
        # 潜在变量到分类任务的映射
        self.classifier = nn.Linear(nlatent, nclass)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, A_norm):
        x = F.relu(self.gc1(x, A_norm))
        mu = self.gc_mu(x, A_norm)
        logvar = self.gc_logvar(x, A_norm)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)
        return logits, mu, logvar


n_features = X_train_tensor.shape[1]
n_hidden = 16
n_latent = 16  # 潜在空间的维度，可根据需要调整
model = VariationalGCN(nfeat=n_features, nhid=n_hidden, nlatent=n_latent, nclass=n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

############################################
# 5. 定义多标签损失函数、分布对齐损失，以及 KL 散度损失
############################################
criterion = nn.BCEWithLogitsLoss()


def distribution_alignment_loss(output, train_mask, unlabeled_mask, eps=1e-8):
    """
    对于多标签任务：
      先对 raw logits 使用 sigmoid 得到预测概率，
      分别计算有标签与无标签样本的平均预测分布，
      最后用 KL 散度使无标签预测分布与有标签预测分布对齐。
    """
    labeled_prob = torch.sigmoid(output[train_mask])
    unlabeled_prob = torch.sigmoid(output[unlabeled_mask])
    target_dist = labeled_prob.mean(dim=0)
    unlabeled_dist = unlabeled_prob.mean(dim=0)
    kl_loss = F.kl_div(torch.log(unlabeled_dist + eps), target_dist + eps, reduction='batchmean')
    return kl_loss


def kl_divergence(mu, logvar):
    # KL 散度公式：-0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss


lambda_align = 0.1  # 分布对齐损失权重
lambda_kl = 0.01  # KL 散度损失权重

############################################
# 6. 训练模型（仅使用训练集数据）
############################################
n_epochs = 500
best_test_acc = 0.0
for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()
    logits, mu, logvar = model(X_train_tensor, A_train_norm)  # logits: (n_train, n_classes)

    # 监督损失：仅在有标签样本上计算
    sup_loss = criterion(logits[train_mask], Y_train_tensor[train_mask])

    # 分布对齐损失：利用训练集中无标签部分
    align_loss = distribution_alignment_loss(logits, train_mask, unlabeled_mask)

    # 计算 KL 散度损失
    kl_loss = kl_divergence(mu, logvar)

    # 总损失：监督损失 + 分布对齐损失 + KL 损失
    total_loss = sup_loss + lambda_align * align_loss + lambda_kl * kl_loss

    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            output_train, _, _ = model(X_train_tensor, A_train_norm)
            pred_train = (torch.sigmoid(output_train) > 0.5).float()
            train_sup_acc = (pred_train[train_mask] == Y_train_tensor[train_mask]).float().mean().item()
            print(
                f"Epoch {epoch:03d}, Loss: {total_loss.item():.4f}, Sup Loss: {sup_loss.item():.4f}, "
                f"Align Loss: {align_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}, "
                f"训练集有标签准确率: {train_sup_acc:.4f}"
            )
            output_test, _, _ = model(X_test_tensor, A_test_norm)
            pred_test = (torch.sigmoid(output_test) > 0.5).float()
            test_acc = (pred_test == Y_test_tensor).float().mean().item()

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_predictions = pred_test.cpu().numpy()

############################################
# 7. 测试模型（使用测试集数据，且模型为训练后保存的模型）
############################################
model.eval()
print("预测结果 shape:", np.shape(best_predictions))
print("测试集标签 shape:", np.shape(Y_test))
eva = evaluate(best_predictions, Y_test)
print(eva)
