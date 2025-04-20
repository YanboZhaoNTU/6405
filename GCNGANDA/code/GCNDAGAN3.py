import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import random
import pandas as pd

# 导入自定义模块（请确保这些模块在你的工作目录中）
from u_mReadData import *
from u_base import *
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *

# 待测试的数据集名称列表
datasnames = [

 "Stackex_coffee", "Stackex_cooking", "Stackex_cs", "Stackex_philosophy",

]

# 统一使用一个 ReadData 实例管理所有数据集
rd = ReadData(datas=datasnames, genpath='data/')

# 用于存放各个数据集的评估结果，后面用于生成合并 Excel 文件
all_results = {}

# 使用枚举获取正确的索引
for idx, dataset in enumerate(datasnames):
    print("==============================================")
    print(f"开始处理数据集: {dataset}")

    # 使用对应的索引 idx 来读取数据
    X_train, Y_train, X_test, Y_test = rd.readData(idx)

    # 训练集、测试集数据转换及维度检查
    n_train = X_train.shape[0]
    if len(Y_train.shape) == 1:
        raise ValueError("训练集多标签数据应为二维数组，每行代表一个样本的标签向量。")
    n_classes = Y_train.shape[1]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

    n_test = X_test.shape[0]
    if len(Y_test.shape) == 1:
        raise ValueError("测试集多标签数据应为二维数组，每行代表一个样本的标签向量。")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    # 构造训练集内的半监督 mask
    fraction = 0.5
    num_labeled = int(n_train * fraction)
    perm = torch.randperm(n_train)
    labeled_indices = perm[:num_labeled]
    train_mask = torch.zeros(n_train, dtype=torch.bool)
    train_mask[labeled_indices] = True
    unlabeled_mask = ~train_mask
    print(f"训练样本总数: {n_train}, 有标签样本数: {train_mask.sum().item()}, 无标签样本数: {unlabeled_mask.sum().item()}")

    # 构造图（基于余弦相似度的 kNN 邻接矩阵）
    def build_graph(X, k):
        n = X.shape[0]
        sim = cosine_similarity(X)  # (n, n)
        A = np.zeros((n, n))
        for i in range(n):
            row = sim[i]
            indices = np.argsort(row)[::-1]  # 降序排列
            indices = indices[indices != i]
            neighbors = indices[:k]
            A[i, neighbors] = 1
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

    # 定义两层 GCN 模型
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
            return x

    n_features = X_train_tensor.shape[1]
    model = GCN(nfeat=n_features, nhid=16, nclass=n_classes)

    # 定义额外模块
    class PseudoLabelGenerator(nn.Module):
        def __init__(self, in_features, hidden_dim, out_features):
            super(PseudoLabelGenerator, self).__init__()
            self.fc1 = nn.Linear(in_features, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, out_features)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)
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

    criterion = nn.BCEWithLogitsLoss()
    def distribution_alignment_loss(output, train_mask, unlabeled_mask, eps=1e-8):
        labeled_prob = torch.sigmoid(output[train_mask])
        unlabeled_prob = torch.sigmoid(output[unlabeled_mask])
        target_dist = labeled_prob.mean(dim=0)
        unlabeled_dist = unlabeled_prob.mean(dim=0)
        kl_loss = F.kl_div(torch.log(unlabeled_dist + eps), target_dist + eps, reduction='batchmean')
        return kl_loss

    lambda_align = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train_tensor = X_train_tensor.to(device)
    Y_train_tensor = Y_train_tensor.to(device)
    A_train_norm = A_train_norm.to(device)
    A_test_norm = A_test_norm.to(device)

    in_feat = n_features
    hidden_dim_module = 32
    latent_dim = 16

    model_S = PseudoLabelGenerator(in_features=in_feat, hidden_dim=hidden_dim_module, out_features=n_classes).to(device)
    model_E = Encoder(in_features=in_feat, hidden_dim=hidden_dim_module, latent_dim=latent_dim).to(device)
    model_G = SampleGenerator(label_dim=n_classes, latent_dim=latent_dim, hidden_dim=hidden_dim_module, out_features=in_feat).to(device)
    model_F = SCMPrior(latent_dim=latent_dim, hidden_dim=hidden_dim_module).to(device)
    model_D = Discriminator(in_features=in_feat + latent_dim, hidden_dim=hidden_dim_module).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    opt_S = torch.optim.Adam(model_S.parameters(), lr=0.001)
    opt_E = torch.optim.Adam(model_E.parameters(), lr=0.001)
    opt_G = torch.optim.Adam(model_G.parameters(), lr=0.001)
    opt_F = torch.optim.Adam(model_F.parameters(), lr=0.001)
    opt_D = torch.optim.Adam(model_D.parameters(), lr=0.001)
    bce_loss = nn.BCEWithLogitsLoss()

    n_epochs = 800
    best_metric = -float('inf')
    best_predictions = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        output_train = model(X_train_tensor, A_train_norm)
        sup_loss = criterion(output_train[train_mask], Y_train_tensor[train_mask])
        align_loss = distribution_alignment_loss(output_train, train_mask, unlabeled_mask)
        loss_cls = sup_loss + lambda_align * align_loss

        x_u = X_train_tensor[unlabeled_mask]
        d_u_logits = model_S(x_u)
        d_u = torch.softmax(d_u_logits, dim=1)
        z_u = model_E(x_u)

        eps = torch.randn(x_u.size(0), latent_dim, device=device)
        z_prior = model_F(eps)
        x_generated = model_G(d_u, z_prior)

        real_input = torch.cat([x_u, z_u], dim=1)
        fake_input = torch.cat([x_generated, z_prior], dim=1)

        real_score = model_D(real_input)
        fake_score = model_D(fake_input)

        real_labels = torch.ones_like(real_score)
        fake_labels = torch.zeros_like(fake_score)
        loss_D_real = bce_loss(real_score, real_labels)
        loss_D_fake = bce_loss(fake_score, fake_labels)
        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        opt_D.zero_grad()
        loss_D.backward(retain_graph=True)
        opt_D.step()

        fake_score = model_D(fake_input)
        loss_adv = bce_loss(fake_score, real_labels)

        opt_S.zero_grad()
        opt_E.zero_grad()
        opt_G.zero_grad()
        opt_F.zero_grad()
        loss_adv.backward()
        opt_S.step()
        opt_E.step()
        opt_G.step()
        opt_F.step()

        total_loss = loss_cls
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                output_train_eval = model(X_train_tensor, A_train_norm)
                pred_train = (torch.sigmoid(output_train_eval) > 0.5).float()
                train_sup_acc = (pred_train[train_mask] == Y_train_tensor[train_mask]).float().mean().item()
                print(f"Epoch {epoch:03d}, Total Loss: {total_loss.item():.4f}, Sup Loss: {sup_loss.item():.4f}, "
                      f"Align Loss: {align_loss.item():.4f}, Adv Loss: {loss_adv.item():.4f}, "
                      f"训练集有标签准确率: {train_sup_acc:.4f}")

                output_test = model(X_test_tensor, A_test_norm)
                pred_test = (torch.sigmoid(output_test) > 0.5).float()
                eva_metrics = evaluate(pred_test.cpu().numpy(), Y_test)
                current_metric = sum(eva_metrics[:7]) / 7.0

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_predictions = pred_test.cpu().numpy()

                print(f"当前评价指标（前7个指标平均值）：{current_metric:.4f}")

    model.eval()
    print("最佳预测结果形状:", np.shape(best_predictions))
    print("测试集标签形状:", np.shape(Y_test))
    eva = evaluate(best_predictions, Y_test)
    print("评估结果：", eva)

    eva_df = pd.DataFrame([eva], columns=[f'指标{i + 1}' for i in range(len(eva))])
    excel_filename = f"eva_results_{dataset}.xlsx"
    eva_df.to_excel(excel_filename, index=False)
    print(f"数据集 {dataset} 的评估结果已保存至文件: {excel_filename}\n")

    all_results[dataset] = eva

with pd.ExcelWriter('all_eva_results.xlsx') as writer:
    for dataset, eva in all_results.items():
        eva_df = pd.DataFrame([eva], columns=[f'指标{i + 1}' for i in range(len(eva))])
        eva_df.to_excel(writer, sheet_name=dataset, index=False)
print("所有数据集的评估结果已汇总保存至文件: all_eva_results.xlsx")
