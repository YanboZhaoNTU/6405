import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from u_mReadData import *
from u_base import *
import numpy as np
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *

# ====================
# 1. 数据准备 (示例)
# ====================

class LabeledDataset(Dataset):
    """
    多标签有监督数据集示例：
    X：形如 (N, D) 的特征
    Y：形如 (N, C) 的多标签标签 (0/1)
    transform：数据增广操作（可选）
    """

    def __init__(self, X, Y, transform=None):
        super().__init__()
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class UnlabeledDataset(Dataset):
    """
    无标签数据集示例：
    只包含X，没有Y
    transform：数据增广操作（可选）
    """

    def __init__(self, X, transform=None):
        super().__init__()
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x


# ====================
# 2. 定义模型
# ====================

class SimpleMLP(nn.Module):
    """
    简易多层感知器示例
    input_dim: 输入维度
    num_classes: 多标签的类别数
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)  # 输出 logits, 后面配合 BCEWithLogitsLoss


# ====================
# 3. 定义增广操作
# ====================
# 在真实图像任务中可用 torchvision.transforms.RandomCrop / RandomFlip 等
# 这里用最简单的“恒等”操作来示例 weak_transform / strong_transform

weak_transform = transforms.Compose([
    # TODO: 替换为真实弱增强操作，如随机裁剪、翻转等
    transforms.Lambda(lambda x: x)  # 仅做占位，恒等
])

strong_transform = transforms.Compose([
    # TODO: 替换为真实强增强操作，如随机颜色扰动、随机剪裁、随机旋转等
    transforms.Lambda(lambda x: x)  # 仅做占位，恒等
])


# ====================
# 4. FixMatch 训练流程
# ====================
def train_fixmatch(
        X_train, Y_train,
        X_unlabeled=None,
        input_dim=10, num_classes=5,
        threshold=0.95, alpha=1.0,
        batch_size=32, epochs=10, lr=1e-3, device='cpu'
):
    """
    X_train: 有标签数据特征, 形状 (N, input_dim)
    Y_train: 有标签数据多标签标签, 形状 (N, num_classes)
    X_unlabeled: 无标签数据特征, 形状 (M, input_dim), 如果没有可传 None
    threshold: 伪标签置信度阈值
    alpha: 伪标签损失系数
    """

    # 转为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    if X_unlabeled is not None:
        X_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32)

    # 构建数据集
    labeled_dataset = LabeledDataset(X_train, Y_train, transform=weak_transform)
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)

    if X_unlabeled is not None:
        unlabeled_dataset_weak = UnlabeledDataset(X_unlabeled, transform=weak_transform)
        unlabeled_dataset_strong = UnlabeledDataset(X_unlabeled, transform=strong_transform)

        unlabeled_loader_weak = DataLoader(unlabeled_dataset_weak, batch_size=batch_size, shuffle=True)
        unlabeled_loader_strong = DataLoader(unlabeled_dataset_strong, batch_size=batch_size, shuffle=True)
    else:
        # 没有无标签数据时，可以只训练有标签数据
        unlabeled_loader_weak = None
        unlabeled_loader_strong = None

    # 定义模型
    model = SimpleMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')  # 多标签使用BCE

    model.train()

    for epoch in range(epochs):
        # 迭代器打包：有标签数据和无标签数据（若有）并行取batch
        if unlabeled_loader_weak and unlabeled_loader_strong:
            # zip 迭代
            for (x_lab, y_lab), x_unlab_weak, x_unlab_strong in zip(labeled_loader, unlabeled_loader_weak,
                                                                    unlabeled_loader_strong):
                x_lab, y_lab = x_lab.to(device), y_lab.to(device)
                x_unlab_weak = x_unlab_weak.to(device)
                x_unlab_strong = x_unlab_strong.to(device)

                # ----------- 监督损失 -----------
                logits_lab = model(x_lab)  # (B, num_classes)
                loss_supervised = criterion(logits_lab, y_lab)

                # ----------- 伪标签损失 -----------
                with torch.no_grad():
                    # 对弱增强的无标签数据做前向
                    logits_weak = model(x_unlab_weak)
                    probs_weak = torch.sigmoid(logits_weak)  # (B, num_classes)

                    # 生成伪标签
                    pseudo_labels = (probs_weak > threshold).float()

                # 对强增强的数据做前向
                logits_strong = model(x_unlab_strong)

                # 只对高置信度样本计算损失
                # 这里采用 mask 方式：只有 pseudo_labels > 0 才计算 1标签的损失
                # 当然也可对 pseudo_labels=0 的类别计算 log(1 - sigmoid(...))，示例仅做正标签
                mask = (probs_weak > threshold).float()

                # BCEWithLogitsLoss: y=1 => log(sigmoid(logits)), y=0 => log(1 - sigmoid(logits))
                # 这里简单处理：只计算 y=1 的部分，mask掉其余
                loss_pseudo = criterion(logits_strong * mask, pseudo_labels * mask)

                # 合并损失
                loss = loss_supervised + alpha * loss_pseudo

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        else:
            # 没有无标签数据时，仅计算监督损失
            for x_lab, y_lab in labeled_loader:
                x_lab, y_lab = x_lab.to(device), y_lab.to(device)
                logits_lab = model(x_lab)
                loss_supervised = criterion(logits_lab, y_lab)
                optimizer.zero_grad()
                loss_supervised.backward()
                optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}] finished.")

    return model


# ====================
# 5. 测试运行 (示例)
# ====================
if __name__ == "__main__":


    datasnames = ["20NG"]
    rd = ReadData(datas=datasnames, genpath='data/')

    X_train, Y_train, X_test, Y_test = rd.readData(0)

    # 假设我们有10维特征，5个类别的多标签问题
    input_dim = 10
    num_classes = 5

    # 生成随机有标签数据
    N = 100  # 有标签样本数
    X_train = torch.randn(N, input_dim).numpy()
    # Y_train 取0或1, 多标签
    Y_train = (torch.rand(N, num_classes) > 0.5).float().numpy()

    # 生成随机无标签数据
    M = 200  # 无标签样本数
    X_unlabeled = torch.randn(M, input_dim).numpy()

    # 训练
    trained_model = train_fixmatch(
        X_train, Y_train,
        X_unlabeled=X_unlabeled,
        input_dim=input_dim,
        num_classes=num_classes,
        threshold=0.95,
        alpha=1.0,
        batch_size=16,
        epochs=5,
        lr=1e-3,
        device='cpu'
    )

    # 推断(Inference)
    trained_model.eval()
    test_input = torch.randn(1, input_dim)
    with torch.no_grad():
        logits = trained_model(test_input)
        probs = torch.sigmoid(logits)
    print("Test sample prediction probabilities:", probs)
