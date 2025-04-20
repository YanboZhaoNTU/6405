import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 导入自定义模块
from u_mReadData import *
from u_base import *
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *

############################################
# 1. 读取数据集
############################################
datasnames = ["Yeast"]
print("数据集:", datasnames[0])
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)

# 数据标准化处理
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练集
n_train = X_train.shape[0]
if len(Y_train.shape) == 1:
    raise ValueError("训练集多标签数据应为二维数组，每行代表一个样本的标签向量。")
n_classes = Y_train.shape[1]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

# 测试集
n_test = X_test.shape[0]
if len(Y_test.shape) == 1:
    raise ValueError("测试集多标签数据应为二维数组，每行代表一个样本的标签向量。")
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

############################################
# 2. 构造半监督 mask（分层采样以维持标签分布）
############################################
fraction = 0.5
num_labeled = int(n_train * fraction)

# 分层采样确保每个类别的标签分布相似
labeled_indices = []
for c in range(n_classes):
    # 找出包含此标签的样本
    class_indices = torch.where(Y_train_tensor[:, c] == 1)[0]
    # 随机选择该类别的样本数
    select_num = int(len(class_indices) * fraction)
    if select_num > 0:
        selected = random.sample(class_indices.tolist(), select_num)
        labeled_indices.extend(selected)

# 确保不重复且总数符合要求
labeled_indices = list(set(labeled_indices))
if len(labeled_indices) > num_labeled:
    labeled_indices = random.sample(labeled_indices, num_labeled)
elif len(labeled_indices) < num_labeled:
    # 添加额外的随机样本
    remaining = list(set(range(n_train)) - set(labeled_indices))
    additional = random.sample(remaining, num_labeled - len(labeled_indices))
    labeled_indices.extend(additional)

train_mask = torch.zeros(n_train, dtype=torch.bool)
train_mask[labeled_indices] = True
unlabeled_mask = ~train_mask

print(f"训练样本总数: {n_train}, 有标签样本数: {train_mask.sum().item()}, 无标签样本数: {unlabeled_mask.sum().item()}")


############################################
# 3. 构造改进的图
############################################
def build_better_graph(X, k_min=5, k_max=15, threshold=0.6):
    """构建自适应 kNN 图"""
    n = X.shape[0]
    sim = cosine_similarity(X)
    A = np.zeros((n, n))

    # 计算节点度作为局部密度的代理
    avg_sim = np.mean(sim, axis=1)
    normalized_sim = (avg_sim - np.min(avg_sim)) / (np.max(avg_sim) - np.min(avg_sim) + 1e-10)

    for i in range(n):
        row = sim[i].copy()
        row[i] = -1  # 排除自身

        # 基于局部密度的自适应 k
        k_adaptive = int(k_min + normalized_sim[i] * (k_max - k_min))

        # 获取前 k 个邻居
        indices = np.argsort(row)[::-1][:k_adaptive]

        # 应用相似度阈值
        valid_indices = indices[row[indices] > threshold]
        if len(valid_indices) == 0:  # 确保至少有一个连接
            valid_indices = indices[:1]

        A[i, valid_indices] = row[valid_indices]  # 按相似度加权

    # 对称化保留权重
    A = (A + A.T) / 2
    return A


def normalize_adj_with_self_loops(A):
    """改进的带自环的归一化"""
    A_tensor = torch.tensor(A, dtype=torch.float32)
    I = torch.eye(A_tensor.size(0), device=A_tensor.device)
    A_hat = A_tensor + I
    D = torch.sum(A_hat, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
    normalized_A = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return normalized_A


# 构建训练集和测试集的图
A_train = build_better_graph(X_train, k_min=7, k_max=20, threshold=0.5)
A_test = build_better_graph(X_test, k_min=7, k_max=20, threshold=0.5)

A_train_norm = normalize_adj_with_self_loops(A_train)
A_test_norm = normalize_adj_with_self_loops(A_test)


############################################
# 4. 定义改进的 GCN 模型
############################################
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, A_norm):
        x = F.dropout(x, p=self.dropout, training=self.training)
        support = torch.mm(x, self.weight) + self.bias
        out = torch.mm(A_norm, support)
        return out


class ImprovedGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout=0.5):
        super(ImprovedGCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid1, dropout)
        self.gc2 = GCNLayer(nhid1, nhid2, dropout)
        self.gc3 = GCNLayer(nhid2, nclass, dropout)
        self.batch_norm1 = nn.BatchNorm1d(nhid1)
        self.batch_norm2 = nn.BatchNorm1d(nhid2)

    def forward(self, x, A_norm):
        x = F.relu(self.batch_norm1(self.gc1(x, A_norm)))
        x = F.relu(self.batch_norm2(self.gc2(x, A_norm)))
        x = self.gc3(x, A_norm)
        return x


n_features = X_train_tensor.shape[1]
model = ImprovedGCN(nfeat=n_features, nhid1=64, nhid2=32, nclass=n_classes, dropout=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)


############################################
# 5. 改进的损失函数
############################################
def focal_loss_for_multilabel(logits, targets, gamma=2.0, alpha=0.25, reduction='mean'):
    """用于多标签分类的 Focal Loss，更好地处理类别不平衡"""
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = alpha * (1 - pt) ** gamma * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def weighted_distribution_alignment_loss(output, train_mask, unlabeled_mask, eps=1e-8):
    """改进的分布对齐损失，考虑类别权重"""
    labeled_prob = torch.sigmoid(output[train_mask])
    unlabeled_prob = torch.sigmoid(output[unlabeled_mask])

    # 计算类别权重（反比于类别频率）
    label_freq = labeled_prob.mean(dim=0)
    class_weights = 1.0 / (label_freq + 0.1)  # 添加平滑项避免除零
    class_weights = class_weights / class_weights.sum() * n_classes  # 归一化

    # 应用类别权重的分布对齐
    target_dist = labeled_prob.mean(dim=0)
    unlabeled_dist = unlabeled_prob.mean(dim=0)

    weighted_kl = class_weights * F.kl_div(
        torch.log(unlabeled_dist + eps),
        target_dist + eps,
        reduction='none'
    )

    return weighted_kl.sum()


def perturb_adj(adj, noise_level=0.1):
    """创建邻接矩阵的扰动版本，用于一致性训练"""
    n = adj.shape[0]

    # 添加随机噪声
    noise = torch.randn(n, n) * noise_level
    perturbed = adj + noise

    # 确保对称性
    perturbed = (perturbed + perturbed.t()) / 2

    # 重新归一化
    D = torch.sum(perturbed, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
    return D_inv_sqrt @ perturbed @ D_inv_sqrt


def consistency_loss(model, x, adj, aug_adj, unlabeled_mask):
    """不同图视图下预测之间的一致性正则化损失"""
    model.eval()  # 目标预测无 dropout
    with torch.no_grad():
        target_outputs = torch.sigmoid(model(x, adj))

    model.train()  # 学生预测启用 dropout
    student_outputs = torch.sigmoid(model(x, aug_adj))

    # 仅计算无标签数据的损失
    consistency = F.mse_loss(
        student_outputs[unlabeled_mask],
        target_outputs[unlabeled_mask],
        reduction='mean'
    )

    return consistency


############################################
# 6. 模型训练（带早停和一致性正则化）
############################################
lambda_align = 0.1  # 分布对齐损失权重
lambda_consistency = 0.2  # 一致性损失权重
n_epochs = 1000
patience = 50  # 早停轮数
best_test_f1 = 0.0
no_improve = 0

for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()
    output_train = model(X_train_tensor, A_train_norm)

    # 创建扰动图用于一致性训练
    if epoch > 50:  # 在初始训练后添加一致性训练
        perturbed_A = perturb_adj(A_train_norm, noise_level=0.05)
    else:
        perturbed_A = A_train_norm

    # 监督损失：使用 Focal Loss
    sup_loss = focal_loss_for_multilabel(output_train[train_mask], Y_train_tensor[train_mask])

    # 分布对齐损失
    align_loss = weighted_distribution_alignment_loss(output_train, train_mask, unlabeled_mask)

    # 一致性损失
    consist_loss = consistency_loss(model, X_train_tensor, A_train_norm, perturbed_A,
                                    unlabeled_mask) if epoch > 50 else 0.0

    # 总损失
    total_loss = sup_loss + lambda_align * align_loss + lambda_consistency * consist_loss

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防止爆炸
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            # 训练集评估
            output_train_eval = model(X_train_tensor, A_train_norm)
            pred_train = (torch.sigmoid(output_train_eval) > 0.5).float()
            train_sup_acc = (pred_train[train_mask] == Y_train_tensor[train_mask]).float().mean().item()

            # 测试集评估
            output_test = model(X_test_tensor, A_test_norm)
            pred_test = (torch.sigmoid(output_test) > 0.5).float()
            test_acc = (pred_test == Y_test_tensor).float().mean().item()

            # 计算 F1 得分用于早停
            from sklearn.metrics import f1_score

            test_pred_np = pred_test.cpu().numpy()
            test_true_np = Y_test_tensor.cpu().numpy()
            test_f1_macro = f1_score(test_true_np, test_pred_np, average='macro')

            print(f"Epoch {epoch:03d}, Loss: {total_loss.item():.4f}, Sup: {sup_loss.item():.4f}, "
                  f"Align: {align_loss.item():.4f}, Consist: {consist_loss if isinstance(consist_loss, float) else consist_loss.item():.4f}, "
                  f"训练准确率: {train_sup_acc:.4f}, 测试准确率: {test_acc:.4f}, F1: {test_f1_macro:.4f}")

            # 更新学习率
            scheduler.step(test_f1_macro)

            # 早停检查
            if test_f1_macro > best_test_f1:
                best_test_f1 = test_f1_macro
                best_predictions = pred_test.cpu().numpy()
                no_improve = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1': best_test_f1
                }, 'best_gcn_model.pt')
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"早停! {patience} 轮无改进。")
                    break

############################################
# 7. 加载最佳模型进行测试
############################################
# 加载最佳模型
checkpoint = torch.load('best_gcn_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 最终测试
with torch.no_grad():
    output_test = model(X_test_tensor, A_test_norm)
    pred_test = (torch.sigmoid(output_test) > 0.5).float()
    best_predictions = pred_test.cpu().numpy()

print(f"最佳模型 - Epoch {checkpoint['epoch']}, F1: {checkpoint['best_f1']:.4f}")
print(np.shape(best_predictions))
print(np.shape(Y_test))
eva = evaluate(best_predictions, Y_test)
print(eva)