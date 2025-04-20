import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import random

# 导入自定义的数据读取等模块（请确保这些模块在你的工作目录中）
from u_mReadData import *
from u_base import *
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *

############################################
# 1. 读取数据（多标签）：使用自定义代码读取 20NG 数据集
############################################
datasnames = ["20NG"]
print("数据集:", datasnames[0])
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)

# 合并训练和测试数据构成完整数据集
X_total = np.concatenate([X_train, X_test], axis=0)
Y_total = np.concatenate([Y_train, Y_test], axis=0)
n_total = X_total.shape[0]

# 假设多标签数据的格式为 (n_total, n_classes) 的二值矩阵
if len(Y_total.shape) == 1:
    raise ValueError("期望多标签数据为二维数组，每行代表一个样本的多标签信息。")
n_classes = Y_total.shape[1]

# 转换为 torch 张量
X_tensor = torch.tensor(X_total, dtype=torch.float32)
# 对于 BCEWithLogitsLoss，标签要求是 float 类型的二值矩阵
Y_tensor = torch.tensor(Y_total, dtype=torch.float32)

############################################
# 2. 构造半监督训练 mask（随机抽取 10% 样本作为有标签数据）
############################################
fraction = 0.1
num_train = int(n_total * fraction)
perm = torch.randperm(n_total)
train_indices = perm[:num_train]
train_mask = torch.zeros(n_total, dtype=torch.bool)
train_mask[train_indices] = True
# 将剩余样本作为无标签数据（用于测试和分布对齐）
unlabeled_mask = ~train_mask

print(f"样本总数: {n_total}, 有标签样本数: {train_mask.sum().item()}, 无标签样本数: {unlabeled_mask.sum().item()}")

############################################
# 3. 构造图：基于文档之间余弦相似度构造 kNN 邻接矩阵
############################################
k = 10  # 每个节点选择最近邻个数
similarity = cosine_similarity(X_total)  # 计算所有文档间余弦相似度
A = np.zeros((n_total, n_total))
for i in range(n_total):
    row = similarity[i]
    # 降序排列索引，排除自身
    indices = np.argsort(row)[::-1]
    indices = indices[indices != i]
    neighbors = indices[:k]
    A[i, neighbors] = 1

# 对称化：如果 i 与 j 任一方向存在边，则认为二者相连
A = np.maximum(A, A.T)
A = torch.tensor(A, dtype=torch.float32)


def normalize_adj(A):
    """
    归一化邻接矩阵： A_norm = D^{-1/2}(A+I)D^{-1/2}
    """
    I = torch.eye(A.size(0), device=A.device)
    A_hat = A + I
    D = torch.sum(A_hat, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


A_norm = normalize_adj(A)


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
        # 多标签任务中不使用 softmax，直接输出 raw logits
        return x


n_features = X_tensor.shape[1]
model = GCN(nfeat=n_features, nhid=16, nclass=n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

############################################
# 5. 定义多标签损失函数（BCEWithLogitsLoss）及分布对齐损失
############################################
criterion = nn.BCEWithLogitsLoss()


def distribution_alignment_loss(output, train_mask, unlabeled_mask, eps=1e-8):
    """
    利用 KL 散度将无标签样本的平均预测分布与有标签样本的预测分布对齐
    对于多标签任务，先对 raw logits 使用 sigmoid 得到概率
    """
    labeled_prob = torch.sigmoid(output[train_mask])
    unlabeled_prob = torch.sigmoid(output[unlabeled_mask])

    # 计算各自的平均概率分布（每个类别的平均激活）
    target_dist = labeled_prob.mean(dim=0)
    unlabeled_dist = unlabeled_prob.mean(dim=0)

    # 计算 KL 散度（让无标签分布接近目标分布）
    kl_loss = F.kl_div(torch.log(unlabeled_dist + eps), target_dist + eps, reduction='batchmean')
    return kl_loss


lambda_align = 0.1  # 分布对齐损失的权重

############################################
# 6. 训练与评估 GCN 模型（半监督+分布对齐），记录最佳测试结果
############################################
n_epochs = 20
best_test_acc = 0.0
best_predictions = None

for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor, A_norm)  # 输出 shape: (n_total, n_classes)

    # 计算有标签数据的监督损失
    sup_loss = criterion(output[train_mask], Y_tensor[train_mask])
    # 计算分布对齐损失（利用无标签数据）
    align_loss = distribution_alignment_loss(output, train_mask, unlabeled_mask)
    total_loss = sup_loss + lambda_align * align_loss

    total_loss.backward()
    optimizer.step()

    # 在评估时使用 sigmoid 得到预测结果
    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(output) > 0.5).float()
        # 这里使用无标签样本的平均准确率作为测试指标
        test_acc = (pred[unlabeled_mask] == Y_tensor[unlabeled_mask]).float().mean().item()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # 将最佳预测结果转换为 NumPy 数组（整个数据集的预测）
            best_predictions = pred.cpu().numpy()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}, Total Loss: {total_loss.item():.4f}, Sup Loss: {sup_loss.item():.4f}, Align Loss: {align_loss.item():.4f}")
            print(f"当前无标签样本 Avg Acc: {test_acc:.4f}, Best Acc: {best_test_acc:.4f}")

############################################
# 7. 输出最佳测试结果的全部预测结果（转换为 np.array 矩阵）
############################################
if best_predictions is not None:
    print("\n最佳测试结果的全部预测结果（np.array 矩阵）：")
    print(best_predictions)
else:
    print("未获得最佳测试结果预测。")
print(np.shape(best_predictions))
print(np.shape(Y_test))
eva = evaluate(best_predictions,Y_test)
print(eva)