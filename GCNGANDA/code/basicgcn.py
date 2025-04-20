import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A):
    """
    计算归一化邻接矩阵：A_norm = D^{-1/2}(A + I)D^{-1/2}
    其中 A 为原始邻接矩阵，I 为单位矩阵
    """
    I = torch.eye(A.size(0)).to(A.device)
    A_hat = A + I  # 加上自连接
    D = torch.sum(A_hat, dim=1)  # 度向量
    # 计算 D^{-1/2}，注意 D 为对角矩阵，可直接用向量运算
    D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm


class GCNLayer(nn.Module):
    """
    自定义图卷积层，实现公式：H' = A_norm * H * W
    """

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
    """
    两层 GCN 模型
    """

    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)

    def forward(self, x, A_norm):
        x = F.relu(self.gc1(x, A_norm))
        # 在训练时可以使用 dropout 来防止过拟合
        x = F.dropout(x, training=self.training)
        x = self.gc2(x, A_norm)
        return F.log_softmax(x, dim=1)


# 构造一个简单图示例
# 这里构造一个 4 个节点的图，边的连接如下（无向图）：
# 0-1, 1-2, 2-3, 3-0 构成一个环状图
A = torch.tensor([[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 1, 0]], dtype=torch.float32)

# 归一化邻接矩阵
A_norm = normalize_adj(A)

# 随机生成 4 个节点，每个节点 5 维的特征
X = torch.rand(4, 5)

# 定义每个节点的标签，这里假设有 2 个类别
labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

# 定义训练 mask（假设只使用前 2 个节点进行训练）
train_mask = torch.tensor([True, True, False, False])

# 初始化模型，设置输入特征维度为 5，隐藏层 4 个单元，输出类别数 2
model = GCN(nfeat=5, nhid=4, nclass=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def custom_loss(output, labels, model, l2_lambda=0.01):
    """
    自定义损失函数：
    1. 使用负对数似然损失（NLL Loss），输入要求为对数概率
    2. 手动添加 L2 正则化项，你可以根据需要修改或扩展
    """
    loss = F.nll_loss(output, labels)
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)
    loss = loss + l2_lambda * l2_reg
    return loss


# 训练模型
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    # 前向传播，注意模型输入包括节点特征 X 和归一化后的邻接矩阵 A_norm
    output = model(X, A_norm)
    # 仅计算训练节点的损失
    loss = custom_loss(output[train_mask], labels[train_mask], model)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 测试（这里简单打印所有节点的预测结果）
model.eval()
with torch.no_grad():
    pred = output.argmax(dim=1)
    print("预测类别：", pred.tolist())
