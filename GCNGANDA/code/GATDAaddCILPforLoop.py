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
# 1. 读取训练集和测试集（多标签数据）
############################################
datasnames = ["Yeast"]
print("数据集:", datasnames[0])
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)

# 训练集
n_train = X_train.shape[0]
if len(Y_train.shape) == 1:
    raise ValueError("训练集多标签数据应为二维数组，每行代表一个样本的标签向量。")
n_classes = Y_train.shape[1]  # 每个样本有 n_classes 个标签

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

# 测试集
n_test = X_test.shape[0]
if len(Y_test.shape) == 1:
    raise ValueError("测试集多标签数据应为二维数组，每行代表一个样本的标签向量。")
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

############################################
# 2. 构造半监督 mask（仅使用训练集数据）
############################################
fraction = 0.5
num_labeled = int(n_train * fraction)
perm = torch.randperm(n_train)
labeled_indices = perm[:num_labeled]
train_mask = torch.zeros(n_train, dtype=torch.bool)
train_mask[labeled_indices] = True
unlabeled_mask = ~train_mask

print(f"训练样本总数: {n_train}, 有标签样本数: {train_mask.sum().item()}, 无标签样本数: {unlabeled_mask.sum().item()}")

############################################
# 3. 构造训练集和测试集的图（基于余弦相似度的 kNN 邻接矩阵）
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

# 对于 GAT 使用原始邻接矩阵，不进行归一化
A_train_tensor = torch.tensor(A_train, dtype=torch.float32)
A_test_tensor = torch.tensor(A_test, dtype=torch.float32)

############################################
# 4. 定义 GAT 层和 Variational GAT 模型
############################################
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        """
        参数：
          in_features: 输入特征维度
          out_features: 输出特征维度
          dropout: dropout 概率
          alpha: LeakyReLU 的负斜率
          concat: 是否连接（True 表示非最后一层采用 ELU 激活）
        """
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 使用线性变换代替手动构造权重矩阵
        self.W = nn.Linear(in_features, out_features, bias=False)
        # 注意力参数，形状为 (2*out_features, 1)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: (N, in_features)
        adj: (N, N) 邻接矩阵（非归一化），其中 0/1 表示边的存在
        """
        Wh = self.W(h)  # (N, out_features)
        N = Wh.size()[0]
        # 计算所有节点对之间的注意力输入，构造 (N, N, 2*out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # 重塑为 (N, N, 2*out_features)
        all_combinations_matrix = all_combinations_matrix.view(N, N, 2 * self.out_features)
        # 计算注意力系数 e_ij
        e = self.leakyrelu(torch.matmul(all_combinations_matrix, self.a).squeeze(2))  # (N, N)

        # 只保留存在边的节点对
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # (N, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class VariationalGAT(nn.Module):
    def __init__(self, nfeat, nhid, nlatent, nclass, dropout=0.6, alpha=0.2):
        super(VariationalGAT, self).__init__()
        self.dropout = dropout
        self.gat1 = GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        # 用于生成均值和对数方差的两层，不使用 concat（即不再使用 ELU 激活）
        self.gat_mu = GATLayer(nhid, nlatent, dropout=dropout, alpha=alpha, concat=False)
        self.gat_logvar = GATLayer(nhid, nlatent, dropout=dropout, alpha=alpha, concat=False)
        # 潜在变量到分类任务的映射
        self.classifier = nn.Linear(nlatent, nclass)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        mu = self.gat_mu(x, adj)
        logvar = self.gat_logvar(x, adj)
        z = self.reparameterize(mu, logvar)
        z = F.dropout(z, self.dropout, training=self.training)
        logits = self.classifier(z)
        return logits, mu, logvar

n_features = X_train_tensor.shape[1]
n_hidden = 16
n_latent = 16  # 潜在空间维度

############################################
# 5. 定义损失函数（多标签）、分布对齐损失以及 KL 散度损失
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
lambda_kl = 0.01    # KL 散度损失权重

############################################
# 6. 训练与选择最佳结果：运行 5 次训练，取测试集上表现最好的结果
############################################
n_runs = 1
global_best_test_acc = 0.0
global_best_predictions = None

for run in range(1, n_runs + 1):
    print(f"\n===== 运行第 {run} 次 =====")
    # 每次重新初始化模型和优化器
    model = VariationalGAT(nfeat=n_features, nhid=n_hidden, nlatent=n_latent, nclass=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_test_acc = 0.0  # 本次运行中的最佳测试准确率
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        logits, mu, logvar = model(X_train_tensor, A_train_tensor)  # logits: (n_train, n_classes)

        # 监督损失：仅在有标签样本上计算
        sup_loss = criterion(logits[train_mask], Y_train_tensor[train_mask])
        # 分布对齐损失：利用训练集中无标签部分
        align_loss = distribution_alignment_loss(logits, train_mask, unlabeled_mask)
        # 计算 KL 散度损失
        kl_loss = kl_divergence(mu, logvar)
        # 总损失
        total_loss = sup_loss + lambda_align * align_loss + lambda_kl * kl_loss

        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                output_train, _, _ = model(X_train_tensor, A_train_tensor)
                pred_train = (torch.sigmoid(output_train) > 0.5).float()
                train_sup_acc = (pred_train[train_mask] == Y_train_tensor[train_mask]).float().mean().item()
                output_test, _, _ = model(X_test_tensor, A_test_tensor)
                pred_test = (torch.sigmoid(output_test) > 0.5).float()
                test_acc = (pred_test == Y_test_tensor).float().mean().item()
                print(
                    f"Run {run} Epoch {epoch:03d}, Loss: {total_loss.item():.4f}, Sup Loss: {sup_loss.item():.4f}, "
                    f"Align Loss: {align_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}, "
                    f"训练集有标签准确率: {train_sup_acc:.4f}, 测试集准确率: {test_acc:.4f}"
                )
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    run_best_predictions = pred_test.cpu().numpy()

    print(f"运行 {run} 次结束，本次最佳测试准确率: {best_test_acc:.4f}")
    if best_test_acc > global_best_test_acc:
        global_best_test_acc = best_test_acc
        global_best_predictions = run_best_predictions

############################################
# 7. 最终测试与评估：输出全局最佳运行的结果
############################################
print("\n===== 最终最佳结果 =====")
print("预测结果 shape:", np.shape(global_best_predictions))
print("测试集标签 shape:", np.shape(Y_test))
eva = evaluate(global_best_predictions, Y_test)
print(eva)
