import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.nn import Parameter


# 定义 GraphConvolution 层，参考论文 https://arxiv.org/abs/1609.02907
class GraphConvolution(nn.Module):
    """
    简单的 GCN 层，类似于 https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input: 节点特征 (num_labels, in_features)
        # adj: 邻接矩阵 (num_labels, num_labels)
        support = torch.matmul(input, self.weight)  # (num_labels, out_features)
        output = torch.matmul(adj, support)  # 信息传播
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# 修改后的 ML-GCN 模型，适用于表格数据多标签分类任务
class MLGCN_Tabular(nn.Module):
    def __init__(self, input_dim, feature_dim, num_labels, label_embed_dim, gc_hidden_dim, adj=None):
        """
        input_dim: 表格数据每个样本的特征数
        feature_dim: 特征提取后得到的向量维度，同时也是生成分类器的维度
        num_labels: 标签数量
        label_embed_dim: 标签初始嵌入（例如预训练词向量）的维度
        gc_hidden_dim: GCN 中间层维度
        adj: 标签相关矩阵，如果为 None 则使用单位矩阵
        """
        super(MLGCN_Tabular, self).__init__()
        # 表格数据特征提取模块：简单的 MLP
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        # 利用两层 GraphConvolution 构建 GCN 模块（结构参照给定代码）
        self.gc1 = GraphConvolution(label_embed_dim, gc_hidden_dim, bias=False)
        self.gc2 = GraphConvolution(gc_hidden_dim, feature_dim, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # 如果没有给定标签相关矩阵，则使用单位矩阵
        if adj is None:
            adj = torch.eye(num_labels)
        # 将邻接矩阵作为可训练参数（如果需要微调标签关系）
        self.A = Parameter(adj.float())

    def forward(self, x, label_embeddings):
        """
        x: 表格数据，形状 (B, input_dim)
        label_embeddings: 标签嵌入，形状 (num_labels, label_embed_dim)
        """
        # 提取表格数据特征，输出 shape: (B, feature_dim)
        features = self.feature_extractor(x)
        # GCN 部分：利用两层 GraphConvolution生成分类器权重
        # 这里直接使用 self.A 作为邻接矩阵，如果需要可以添加额外处理
        adj = self.A
        out = self.gc1(label_embeddings, adj)  # (num_labels, gc_hidden_dim)
        out = self.leaky_relu(out)
        out = self.gc2(out, adj)  # (num_labels, feature_dim)
        # 转置后形状为 (feature_dim, num_labels)
        out = out.transpose(0, 1)
        # 内积得到每个样本每个标签的预测得分
        logits = torch.matmul(features, out)  # (B, num_labels)
        return logits


# 一个简单的训练示例（使用随机数据演示）
def train_model():
    # 超参数设置
    batch_size = 32
    input_dim = 10  # 表格数据特征数
    feature_dim = 64  # 特征提取后向量维度
    num_labels = 5  # 标签数量
    label_embed_dim = 50  # 标签嵌入维度
    gc_hidden_dim = 32  # GCN 中间层维度
    num_epochs = 10

    model = MLGCN_Tabular(input_dim, feature_dim, num_labels, label_embed_dim, gc_hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()  # 多标签二分类损失

    for epoch in range(num_epochs):
        # 模拟随机生成表格数据和对应多标签目标
        x = torch.randn(batch_size, input_dim)
        y = (torch.rand(batch_size, num_labels) > 0.5).float()
        # 模拟标签嵌入，实际应用中可以使用预训练词向量等
        label_embeddings = torch.randn(num_labels, label_embed_dim)

        logits = model(x, label_embeddings)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train_model()
