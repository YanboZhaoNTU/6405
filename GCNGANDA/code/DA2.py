import torch
import torch.nn as nn
import torch.optim as optim



import torch
import torch.nn as nn

from u_base import *
import numpy as np
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *


class DistributionAlignmentLoss(nn.Module):
    def __init__(self, bce_weight=1.0, align_weight=1.0):
        """
        :param bce_weight: BCE 损失的权重
        :param align_weight: 分布对齐（KL 散度）损失的权重
        """
        super(DistributionAlignmentLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        # 使用 KLDivLoss 计算 KL 散度（注意输入需要是 log-probability）
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.bce_weight = bce_weight
        self.align_weight = align_weight

    def forward(self, logits, targets):
        """
        :param logits: 模型输出的原始分数，形状为 [batch_size, num_classes]
        :param targets: 标签，通常为 0/1 值，形状为 [batch_size, num_classes]
        """
        # 1. 计算标准的二元交叉熵损失
        bce = self.bce_loss(logits, targets.float())

        # 2. 计算分布对齐损失
        # 首先，将 logits 通过 sigmoid 转换为概率
        probs = torch.sigmoid(logits)  # [batch_size, num_classes]

        # 在一个 mini-batch 内计算每个类别的平均预测概率和真实标签分布
        pred_distribution = torch.mean(probs, dim=0)       # [num_classes]
        true_distribution = torch.mean(targets.float(), dim=0)  # [num_classes]

        # 为了计算 KL 散度，这两个分布需要归一化为概率分布（即和为 1）
        eps = 1e-8  # 防止除 0
        pred_distribution = pred_distribution / (pred_distribution.sum() + eps)
        true_distribution = true_distribution / (true_distribution.sum() + eps)

        # KLDivLoss 要求第一个输入为 log 概率
        kl = self.kl_loss(torch.log(pred_distribution + eps), true_distribution)

        # 3. 最终损失为两部分的加权和
        loss = self.bce_weight * bce + self.align_weight * kl
        return loss


# 定义一个简单的模型示例（这里假设是一个单层全连接）
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# 假设你的输入特征维度为 10，类别数为 5（多标签任务，每个标签为 0 或 1）
input_dim = 10
num_classes = 5

# 创建模型实例
model = MyModel(input_dim, num_classes)

# 初始化自定义损失函数
loss_fn = DistributionAlignmentLoss(bce_weight=1.0, align_weight=1.0)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模拟一些数据
batch_size = 32


datasnames = ["20NG"]
rd = ReadData(datas=datasnames, genpath='data/')

X_train, Y_train, X_test, Y_test = rd.readData(0)
X = torch.randn(batch_size, input_dim)            # 输入特征
Y = torch.randint(0, 2, (batch_size, num_classes))  # 多标签，取值 0 或 1
X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)
# 训练循环
model.train()
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()             # 梯度清零
    logits = model(X_train)                 # 前向传播，输出 logits，形状为 [batch_size, num_classes]
    loss = loss_fn(logits,  Y_train)         # 计算损失
    loss.backward()                   # 反向传播
    optimizer.step()                  # 参数更新

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
