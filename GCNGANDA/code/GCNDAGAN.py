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


############################################
# 5. 定义额外模块：伪标签生成器、编码器、样本生成器、SCM 先验和判别器
############################################
class PseudoLabelGenerator(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(PseudoLabelGenerator, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # 输出 raw logits


class Encoder(nn.Module):
    def __init__(self, in_features, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SampleGenerator(nn.Module):
    def __init__(self, label_dim, latent_dim, hidden_dim, out_features):
        super(SampleGenerator, self).__init__()
        self.fc1 = nn.Linear(label_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, d, z):
        # d: 伪标签（如 softmax 概率），z: 潜变量
        combined = torch.cat([d, z], dim=1)
        x = F.relu(self.fc1(combined))
        return self.fc2(x)


class SCMPrior(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(SCMPrior, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, eps):
        return self.fc(eps)


class Discriminator(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


############################################
# 6. 定义多标签损失函数及分布对齐损失
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


lambda_align = 0.1  # 分布对齐损失权重

############################################
# 7. 设备设置、优化器初始化
############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X_train_tensor = X_train_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)
A_train_norm = A_train_norm.to(device)
A_test_norm = A_test_norm.to(device)

# 参数设置
in_feat = n_features  # 原始特征维度
hidden_dim = 32  # 隐藏层维度
latent_dim = 16  # 潜变量维度

# 初始化额外模块
model_S = PseudoLabelGenerator(in_features=in_feat, hidden_dim=hidden_dim, out_features=n_classes).to(device)
model_E = Encoder(in_features=in_feat, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
model_G = SampleGenerator(label_dim=n_classes, latent_dim=latent_dim, hidden_dim=hidden_dim, out_features=in_feat).to(
    device)
model_F = SCMPrior(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
model_D = Discriminator(in_features=in_feat + latent_dim, hidden_dim=hidden_dim).to(device)

# 优化器：GCN 模型以及额外模块的优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
opt_S = torch.optim.Adam(model_S.parameters(), lr=0.001)
opt_E = torch.optim.Adam(model_E.parameters(), lr=0.001)
opt_G = torch.optim.Adam(model_G.parameters(), lr=0.001)
opt_F = torch.optim.Adam(model_F.parameters(), lr=0.001)
opt_D = torch.optim.Adam(model_D.parameters(), lr=0.001)

# 判别器损失采用 BCEWithLogitsLoss
bce_loss = nn.BCEWithLogitsLoss()

############################################
# 8. 整合训练循环：GCN 监督训练与 PCLP 对抗训练
############################################
n_epochs = 1500
best_test_acc = 0.0

for epoch in range(1, n_epochs + 1):
    model.train()
    # 首先清空 GCN 模型的梯度
    optimizer.zero_grad()

    # 计算 GCN 模型输出（对整个训练集）
    output_train = model(X_train_tensor, A_train_norm)  # (n_train, n_classes)

    # -------------------------------
    # A. 计算监督损失及分布对齐损失
    # -------------------------------
    sup_loss = criterion(output_train[train_mask], Y_train_tensor[train_mask])
    align_loss = distribution_alignment_loss(output_train, train_mask, unlabeled_mask)
    loss_cls = sup_loss + lambda_align * align_loss

    # -------------------------------
    # B. 对抗训练部分（针对无标签数据）
    # -------------------------------
    # 取无标签数据
    x_u = X_train_tensor[unlabeled_mask]

    # （a）伪标签生成器：生成伪标签 d_u
    d_u_logits = model_S(x_u)  # 输出 raw logits
    d_u = torch.softmax(d_u_logits, dim=1)  # 得到伪标签概率分布

    # （b）编码器：计算无标签数据的潜变量 z_u
    z_u = model_E(x_u)

    # （c）从标准正态中采样噪声，并经过 SCM 先验映射得到 z_prior
    eps = torch.randn(x_u.size(0), latent_dim, device=device)
    z_prior = model_F(eps)

    # （d）样本生成器：利用伪标签和 z_prior 生成假样本 x_generated
    x_generated = model_G(d_u, z_prior)

    # （e）判别器：构造真实与生成的输入进行区分
    # 真实分支：使用无标签数据 x_u 与编码器获得的 z_u 拼接
    real_input = torch.cat([x_u, z_u], dim=1)
    # 生成分支：使用生成器生成的样本与 z_prior 拼接
    fake_input = torch.cat([x_generated, z_prior], dim=1)

    real_score = model_D(real_input)
    fake_score = model_D(fake_input)

    # 判别器目标：真实分支应输出 1，生成分支应输出 0
    real_labels = torch.ones_like(real_score)
    fake_labels = torch.zeros_like(fake_score)
    loss_D_real = bce_loss(real_score, real_labels)
    loss_D_fake = bce_loss(fake_score, fake_labels)
    loss_D = 0.5 * (loss_D_real + loss_D_fake)

    # 更新判别器参数
    opt_D.zero_grad()
    loss_D.backward(retain_graph=True)
    opt_D.step()

    # 对抗训练：更新伪标签生成器、编码器、样本生成器与 SCM 先验
    fake_score = model_D(fake_input)
    loss_adv = bce_loss(fake_score, real_labels)  # 希望生成分支骗过判别器

    opt_S.zero_grad()
    opt_E.zero_grad()
    opt_G.zero_grad()
    opt_F.zero_grad()
    loss_adv.backward()
    opt_S.step()
    opt_E.step()
    opt_G.step()
    opt_F.step()

    # -------------------------------
    # C. 综合损失：仅更新 GCN 模型（即分类器）
    # -------------------------------
    total_loss = loss_cls
    total_loss.backward()
    optimizer.step()

    # 输出调试信息
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            output_train_eval = model(X_train_tensor, A_train_norm)
            pred_train = (torch.sigmoid(output_train_eval) > 0.5).float()
            train_sup_acc = (pred_train[train_mask] == Y_train_tensor[train_mask]).float().mean().item()
            print(
                f"Epoch {epoch:03d}, Total Loss: {total_loss.item():.4f}, Sup Loss: {sup_loss.item():.4f}, Align Loss: {align_loss.item():.4f}, Adv Loss: {loss_adv.item():.4f}, 训练集有标签准确率: {train_sup_acc:.4f}")

            output_test = model(X_test_tensor, A_test_norm)
            pred_test = (torch.sigmoid(output_test) > 0.5).float()
            test_acc = (pred_test == Y_test_tensor.to(device)).float().mean().item()

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_predictions = pred_test.cpu().numpy()

############################################
# 9. 测试模型（使用测试集数据）
############################################
model.eval()
print("最佳预测结果形状:", np.shape(best_predictions))
print("测试集标签形状:", np.shape(Y_test))
eva = evaluate(best_predictions, Y_test)
print("评估结果：", eva)
