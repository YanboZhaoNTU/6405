import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import random

# -------------------------
# 多损失函数代码
# -------------------------
eps = 1e-7


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)


class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE and cls_num_list is not None:
            idx = 1  # 可根据需要调整
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)
        return self

    def forward(self, output_logits, target):
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                idx = 1  # 可根据需要调整
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)
        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch
            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits
        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s
        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)
        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=1,
                 reweight=True, reweight_epoch=-1, base_loss_factor=1.0,
                 additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = nn.MultiLabelSoftMarginLoss()
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # 初始化默认属性，避免后续访问未定义
        self.per_cls_weights_base = None
        self.per_cls_weights_diversity = None

        if cls_num_list is None:
            self.m_list = None
            self.per_cls_weights_enabled = None
        else:
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            self.m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.s = s
            assert s > 0

            if reweight_epoch != -1:
                idx = 1  # 根据需要调整
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)
            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            self.per_cls_weights_enabled_diversity = torch.tensor(
                per_cls_weights, dtype=torch.float, requires_grad=False)
        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)
        if hasattr(self, 'per_cls_weights_enabled_diversity') and self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)
        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch
            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits
        index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), True)
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s
        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)
        loss = 0
        for logits_item in extra_info['logits']:
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)

            base_diversity_temperature = self.base_diversity_temperature
            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature

            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist,
                                                                                                      mean_output_dist,
                                                                                                      reduction='batchmean')
        return loss


# （此处省略 RIDELossWithDistill，如果需要可参照上面代码添加）

# -------------------------
# 定义适合表格数据的 GCN+MLP 模型
# -------------------------
class GCNMLP(nn.Module):
    def __init__(self, input_dim, num_classes, in_channel=None, num_experts=4):
        """
        input_dim : 表格数据的特征维度
        num_classes : 分类类别数
        in_channel : 用于图卷积的输入维度（默认设为 input_dim）
        num_experts : 多专家分支数量
        """
        super(GCNMLP, self).__init__()
        if in_channel is None:
            in_channel = input_dim
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.num_experts = num_experts

        # 特征提取部分：简单 MLP
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )
        # 多专家分支：每个专家先将 1024 维映射到 512 维
        self.layer4s = nn.ModuleList([
            nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
            for _ in range(num_experts)
        ])
        # 每个专家对应一个简单的 GCN 模块
        self.experts = nn.ModuleList([
            GCN_block(in_channel, 1024, 512)
            for _ in range(num_experts)
        ])
        # 固定的邻接矩阵（这里使用单位矩阵，可根据实际需求更改）
        self.A = Parameter(torch.eye(num_classes, dtype=torch.float32))
        # Transformer 编码器，用于融合
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # 可训练的类嵌入 [num_classes, in_channel]
        self.class_embeddings = Parameter(torch.randn(num_classes, in_channel))

    def forward(self, x):
        # x: [batch, input_dim]
        features = self.feature_extractor(x)  # [batch, 1024]
        adj = self.A  # [num_classes, num_classes]
        expert_outputs = []  # 存放每个专家的 logits
        for i in range(self.num_experts):
            # 分支1：将特征映射到 512 维
            feat_i = self.layer4s[i](features)  # [batch, 512]
            # 分支2：对类嵌入进行 GCN 处理，得到 [num_classes, 512]
            expert_out = self.experts[i](self.class_embeddings, adj)  # [num_classes, 512]
            # TransformerEncoder 要求输入 (seq_len, batch, feat_dim)
            feat_trans = self.transformer_encoder(feat_i.unsqueeze(0)).squeeze(0)  # [batch, 512]
            expert_trans = self.transformer_encoder(expert_out.unsqueeze(0)).squeeze(0)  # [num_classes, 512]
            # 计算 logits：每个样本与每个类别的内积
            logits = torch.matmul(feat_trans, expert_trans.t())  # [batch, num_classes]
            expert_outputs.append(logits)
        # 多专家输出取均值作为最终预测
        final_logits = torch.stack(expert_outputs, dim=0).mean(dim=0)  # [batch, num_classes]
        return {"output": final_logits, "logits": expert_outputs}


# -------------------------
# 辅助：定义简单的 GCN 模块（两层 GCN）
# -------------------------
class GraphConvolution(nn.Module):
    """
    简单的图卷积层，参考 https://arxiv.org/abs/1609.02907
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
        # input: [num_nodes, in_features], adj: [num_nodes, num_nodes]
        support = torch.matmul(input, self.weight)  # [num_nodes, out_features]
        output = torch.matmul(adj, support)  # [num_nodes, out_features]
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_block(nn.Module):
    """
    两层 GCN 模块
    """

    def __init__(self, in_features, mid_features, out_features):
        super(GCN_block, self).__init__()
        self.gc1 = GraphConvolution(in_features, mid_features)
        self.gc2 = GraphConvolution(mid_features, out_features)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, inp, adj):
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        return x


# -------------------------
# 辅助函数：多数投票
# -------------------------
def majority_vote(preds_list):
    # preds_list: list of [batch] tensor，每个元素为专家的预测结果
    preds_stack = torch.stack(preds_list, dim=0)  # [num_experts, batch]
    preds_stack = preds_stack.t()  # [batch, num_experts]
    mode_vals, _ = torch.mode(preds_stack, dim=1)
    return mode_vals


# -------------------------
# 训练代码示例
# -------------------------
def train_model(X_train, Y_train, num_epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转换为 torch.Tensor（假设 X_train, Y_train 为 numpy 数组或类似对象）
    if not torch.is_tensor(X_train):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not torch.is_tensor(Y_train):
        Y_train = torch.tensor(Y_train, dtype=torch.long)

    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(torch.unique(Y_train))
    input_dim = X_train.shape[1]

    # 统计每个类别数量，用于损失函数 reweight
    cls_counts = np.array([(Y_train == i).sum().item() for i in range(num_classes)])

    model = GCNMLP(input_dim=input_dim, num_classes=num_classes, in_channel=input_dim, num_experts=4)
    model.to(device)

    # 初始化多损失函数（可根据需要调整参数和权重）
    ce_loss_fn = CrossEntropyLoss(cls_num_list=cls_counts, reweight_CE=True).to(device)
    focal_loss_fn = FocalLoss(gamma=2.0).to(device)
    ldam_loss_fn = LDAMLoss(cls_num_list=cls_counts, max_m=0.5, s=30, reweight_epoch=-1).to(device)
    ride_loss_fn = RIDELoss(base_diversity_temperature=1.0, max_m=0.5, s=1,
                            reweight=True, reweight_epoch=-1, base_loss_factor=1.0,
                            additional_diversity_factor=-0.2, reweight_factor=0.05).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        correct_final = 0
        correct_vote = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            logits_final = outputs["output"]  # [batch, num_classes]
            expert_logits = outputs["logits"]  # list of [batch, num_classes]

            # 计算各损失函数
            loss_ce = ce_loss_fn(logits_final, batch_y)
            loss_focal = focal_loss_fn(logits_final, batch_y)
            loss_ldam = ldam_loss_fn(logits_final, batch_y)
            loss_ride = ride_loss_fn(logits_final, batch_y, extra_info={"logits": expert_logits})

            loss = (loss_ce + loss_focal + loss_ldam + loss_ride) / 4.0
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

            # 预测：聚合输出与各专家投票
            preds_final = torch.argmax(logits_final, dim=1)
            preds_experts = [torch.argmax(l, dim=1) for l in expert_logits]
            preds_vote = majority_vote(preds_experts)

            correct_final += (preds_final == batch_y).sum().item()
            correct_vote += (preds_vote == batch_y).sum().item()

        epoch_loss = running_loss / total_samples
        acc_final = correct_final / total_samples
        acc_vote = correct_vote / total_samples
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Final_Acc: {acc_final:.4f}, Vote_Acc: {acc_vote:.4f}")
    return model


# -------------------------
# 示例：使用随机数据进行训练
# -------------------------
if __name__ == "__main__":
    # 假设表格数据有 300 个特征，分类任务有 5 个类别
    num_samples = 1000
    num_features = 300
    num_classes = 5
    X_train_dummy = np.random.randn(num_samples, num_features).astype(np.float32)
    Y_train_dummy = np.random.randint(0, num_classes, size=(num_samples,))

    trained_model = train_model(X_train_dummy, Y_train_dummy, num_epochs=10, batch_size=32, learning_rate=0.001)
