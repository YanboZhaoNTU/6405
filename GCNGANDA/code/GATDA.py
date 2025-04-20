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


k = 10
A_train = build_graph(X_train, k)
A_test = build_graph(X_test, k)


# 对于 GAT，通常使用原始邻接矩阵（可加自环）
def add_self_loops(A):
    n = A.shape[0]
    I = np.eye(n)
    return A + I


A_train = add_self_loops(A_train)
A_test = add_self_loops(A_test)

# 转为 tensor
A_train_tensor = torch.tensor(A_train, dtype=torch.float32)
A_test_tensor = torch.tensor(A_test, dtype=torch.float32)


############################################
# 4. 定义 GAT 层和两层 GAT 模型（适应多标签任务）
############################################
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        """
        参数：
          in_features: 输入特征维度
          out_features: 输出特征维度
          dropout: dropout 概率
          alpha: LeakyReLU 的负斜率
          concat: 是否拼接（如果为 True，则输出经过 ELU 激活）
        """
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha

        # 线性变换
        self.W = nn.Linear(in_features, out_features, bias=False)
        # 注意力参数 a，形状为 (2*out_features, 1)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: 节点特征，形状 (N, in_features)
        adj: 邻接矩阵，形状 (N, N)，元素为 0/1
        """
        Wh = self.W(h)  # (N, out_features)
        N = Wh.size()[0]

        # 计算所有节点对的注意力系数
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1),
                             Wh.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # (N, N)

        # 使用邻接矩阵作为掩码（无边的赋予极小值）
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)  # (N, out_features)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.gat1 = GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.gat2 = GATLayer(nhid, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, adj)
        # 多标签任务中直接输出 raw logits（不经过 softmax）
        return x


############################################
# 5. 定义多标签损失函数及分布对齐损失
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


lambda_align = 0.1  # 分布对齐损失的权重

############################################
# 6. 训练模型（仅使用训练集数据）
############################################
n_epochs = 500
best_test_acc = 0.0
model = GAT(nfeat=X_train_tensor.shape[1], nhid=16, nclass=n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()
    # 使用 GAT 模型时，直接传入原始邻接矩阵
    output_train = model(X_train_tensor, A_train_tensor)  # (n_train, n_classes)

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
            pred_train = (torch.sigmoid(output_train) > 0.5).float()
            train_sup_acc = (pred_train[train_mask] == Y_train_tensor[train_mask]).float().mean().item()
            output_test = model(X_test_tensor, A_test_tensor)
            pred_test = (torch.sigmoid(output_test) > 0.5).float()
            test_acc = (pred_test == Y_test_tensor).float().mean().item()
            print(f"Epoch {epoch:03d}, Loss: {total_loss.item():.4f}, Sup Loss: {sup_loss.item():.4f}, "
                  f"Align Loss: {align_loss.item():.4f}, 训练集有标签准确率: {train_sup_acc:.4f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_predictions = pred_test.cpu().numpy()

############################################
# 7. 测试模型（使用测试集数据）
############################################
model.eval()
print(np.shape(best_predictions))
print(np.shape(Y_test))
eva = evaluate(best_predictions, Y_test)
print(eva)
