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
datasnames = ["20NG"]
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
# 例如，随机抽取 10% 的训练样本作为有标签数据
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


k = 10
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
# 4. 定义两层 GCN 模型（适应多标签任务）
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


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)

    def forward(self, x, A_norm):
        x = F.relu(self.gc1(x, A_norm))
        x = F.dropout(x, training=self.training)
        x = self.gc2(x, A_norm)
        # 多标签任务中直接输出 raw logits，不经过 softmax
        return x


n_features = X_train_tensor.shape[1]
model = GCN(nfeat=n_features, nhid=16, nclass=n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

############################################
# 5. 定义多标签损失函数及分布对齐损失
############################################
criterion = nn.BCEWithLogitsLoss()


def distribution_alignment_loss(output, train_mask, unlabeled_mask, eps=1e-8):
    """
    对于多标签任务：
      先对 raw logits 使用 sigmoid 得到预测概率，
      分别计算有标签与无标签样本的平均预测分布，
      最后用 JS 散度使无标签预测分布与有标签预测分布对齐。

      Jensen-Shannon 散度定义为：
      JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
      其中 M = 0.5*(P + Q)
    """
    import torch.nn.functional as F

    labeled_prob = torch.sigmoid(output[train_mask])
    unlabeled_prob = torch.sigmoid(output[unlabeled_mask])
    target_dist = labeled_prob.mean(dim=0)
    unlabeled_dist = unlabeled_prob.mean(dim=0)

    # 计算平均分布 M
    m = 0.5 * (target_dist + unlabeled_dist)

    # 分别计算 target 和 unlabeled 与 m 之间的 KL 散度
    js_loss = 0.5 * F.kl_div(torch.log(target_dist + eps), m + eps, reduction='batchmean') \
              + 0.5 * F.kl_div(torch.log(unlabeled_dist + eps), m + eps, reduction='batchmean')
    return js_loss


lambda_align = 0.1  # 分布对齐损失的权重

############################################
# 6. 训练模型（仅使用训练集数据）
############################################
n_epochs = 500
best_test_acc = 0.0
for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()
    output_train = model(X_train_tensor, A_train_norm)  # (n_train, n_classes)

    # 监督损失：仅在有标签训练样本上计算
    sup_loss = criterion(output_train[train_mask], Y_train_tensor[train_mask])
    # 分布对齐损失：利用训练集中无标签部分
    align_loss = distribution_alignment_loss(output_train, train_mask, unlabeled_mask)
    total_loss = sup_loss + lambda_align * align_loss

    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            # 对训练集进行简单评估
            pred_train = (torch.sigmoid(output_train) > 0.5).float()
            train_sup_acc = (pred_train[train_mask] == Y_train_tensor[train_mask]).float().mean().item()
            print(
                f"Epoch {epoch:03d}, Loss: {total_loss.item():.4f}, Sup Loss: {sup_loss.item():.4f}, Align Loss: {align_loss.item():.4f}, 训练集有标签准确率: {train_sup_acc:.4f}")
            output_test = model(X_test_tensor, A_test_norm)
            pred_test = (torch.sigmoid(output_test) > 0.5).float()
            test_acc = (pred_test == Y_test_tensor).float().mean().item()

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # 将最佳预测结果转换为 NumPy 数组（整个数据集的预测）
                best_predictions = pred_test.cpu().numpy()

############################################
# 7. 测试模型（使用测试集数据，且模型为训练后保存的模型）
############################################
model.eval()

print(np.shape(best_predictions))
print(np.shape(Y_test))
eva = evaluate(best_predictions,Y_test)
print(eva)
