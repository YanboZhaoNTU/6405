import os
import shutil
import time as timer  # 用 timer 代替 time 模块，防止冲突
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import random

# 导入自定义模块（请确保这些模块在你的工作目录中）
from u_mReadData import *
from u_base import *
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *


###############################################################################
# 损失函数定义：FocalLoss, CrossEntropyLoss, LDAMLoss, CombinedLoss
###############################################################################
def focal_loss(input_values, gamma):
    """计算 Focal Loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)


class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE:
            idx = 1  # 根据需要调整
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
                idx = 1
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        return self

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


class CombinedLoss(nn.Module):
    def __init__(self, cls_num_list=None, focal_gamma=2.0, reweight_CE=False,
                 max_m=0.5, s=30, reweight_epoch=-1, weight=None):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(cls_num_list=cls_num_list, weight=weight, gamma=focal_gamma)
        self.ce_loss = CrossEntropyLoss(cls_num_list=cls_num_list, reweight_CE=reweight_CE)
        self.ldam_loss = LDAMLoss(cls_num_list=cls_num_list, max_m=max_m, s=s, reweight_epoch=reweight_epoch)

    def forward(self, output_logits, target):
        loss1 = self.focal_loss(output_logits, target)
        loss2 = self.ce_loss(output_logits, target)
        loss3 = self.ldam_loss(output_logits, target)
        combined_loss = (loss1 + loss2 + loss3) / 3.0
        return combined_loss


###############################################################################
# 数据读取部分
###############################################################################
datasnames = ["20NG"]
print("数据集:", datasnames[0])
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)


###############################################################################
# 自定义数据集（若标签为 one-hot，则转换为类别索引）
###############################################################################
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y  # 如果 Y 为 one-hot 编码，则转换为类别索引

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx]).float()
        y = torch.tensor(self.Y[idx])
        if y.ndim > 0 and y.numel() > 1:
            y = torch.argmax(y)
        y = y.long()
        return x, y


train_dataset = CustomDataset(X_train, Y_train)
val_dataset = CustomDataset(X_test, Y_test)


###############################################################################
# 模型定义
###############################################################################
class TabularModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TabularModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


input_dim = len(X_train[0])
# 判断标签是否为 one-hot 编码，如果是则取第二个维度，否则取 np.unique
if np.array(Y_train).ndim > 1:
    output_dim = np.array(Y_train).shape[1]
else:
    output_dim = len(np.unique(Y_train))
model = TabularModel(input_dim=input_dim, output_dim=output_dim)


###############################################################################
# 辅助计量器实现
###############################################################################
class AveragePrecisionMeter:
    def __init__(self, difficult_examples=False):
        self.difficult_examples = difficult_examples
        self.reset()

    def add(self, output, target):
        pass

    def value(self):
        return torch.tensor([0.5] * output_dim)

    def reset(self):
        pass


class AverageValueMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def add(self, value):
        self.sum += value
        self.count += 1

    def value(self):
        return [self.sum / self.count if self.count > 0 else 0]


###############################################################################
# 训练引擎定义
###############################################################################
class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()
        if self._state('batch_size') is None:
            self.state['batch_size'] = 64
        if self._state('workers') is None:
            self.state['workers'] = 0
        if self._state('device_ids') is None:
            self.state['device_ids'] = [0]
        if self._state('evaluate') is None:
            self.state['evaluate'] = False
        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0
        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 200
        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []
        self.state['meter_loss'] = AverageValueMeter()
        self.state['batch_time'] = AverageValueMeter()
        self.state['data_time'] = AverageValueMeter()
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 10

    def _state(self, name):
        return self.state.get(name, None)

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{}]\tLoss {:.4f}'.format(self.state['epoch'], loss))
            else:
                print('Test:\tLoss {:.4f}'.format(loss))
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True, acc_count=False):
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])
        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{}][{}/{}]\tTime {:.3f} ({:.3f})\tData {:.3f} ({:.3f})\tLoss {:.4f} ({:.4f})'
                      .format(self.state['epoch'], self.state['iteration'], len(data_loader),
                              self.state['batch_time_current'], batch_time,
                              self.state['data_time_batch'], data_time,
                              self.state['loss_batch'], loss))
            else:
                print('Test: [{}/{}]\tTime {:.3f} ({:.3f})\tData {:.3f} ({:.3f})\tLoss {:.4f} ({:.4f})'
                      .format(self.state['iteration'], len(data_loader),
                              self.state['batch_time_current'], batch_time,
                              self.state['data_time_batch'], data_time,
                              self.state['loss_batch'], loss))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        input_var = self.state['input']
        target_var = self.state['target'].long()  # 转换为 Long 型，CrossEntropyLoss要求
        if not training:
            with torch.no_grad():
                self.state['output'] = model(input_var)
                self.state['loss'] = criterion(self.state['output'], target_var)
        else:
            self.state['output'] = model(input_var)
            self.state['loss'] = criterion(self.state['output'], target_var)
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def init_learning(self, model, criterion):
        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None, cb_sampling=False):
        self.init_learning(model, criterion)
        train_loader = DataLoader(train_dataset, batch_size=self.state['batch_size'],
                                  shuffle=True, num_workers=self.state['workers'])
        val_loader = DataLoader(val_dataset, batch_size=self.state['batch_size'],
                                shuffle=False, num_workers=self.state['workers'])
        if self.state['use_gpu']:
            model = model.to('cuda:{}'.format(self.state['device_ids'][0]))
            model = nn.DataParallel(model, device_ids=self.state['device_ids'])
            criterion = criterion.to('cuda:{}'.format(self.state['device_ids'][0]))
        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return
        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            self.train(train_loader, model, criterion, optimizer, epoch)
            prec1 = self.validate(val_loader, model, criterion)
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({'epoch': epoch + 1,
                                  'state_dict': model.state_dict(),
                                  'best_score': self.state['best_score']}, is_best)
            print('*** best={:.3f}'.format(self.state['best_score']))
        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):
        model.train()
        self.on_start_epoch(True, model, criterion, data_loader, optimizer)
        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')
        end = timer.time()
        for i, (input, target) in enumerate(data_loader):
            self.state['iteration'] = i
            self.state['data_time_batch'] = timer.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])
            self.state['input'] = input if not self.state['use_gpu'] else input.to(
                'cuda:{}'.format(self.state['device_ids'][0]))
            self.state['target'] = target if not self.state['use_gpu'] else target.to(
                'cuda:{}'.format(self.state['device_ids'][0]))
            self.on_start_batch(True, model, criterion, data_loader, optimizer)
            self.on_forward(True, model, criterion, data_loader, optimizer)
            self.state['batch_time_current'] = timer.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = timer.time()
            self.on_end_batch(True, model, criterion, data_loader, optimizer)
        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):
        model.eval()
        self.on_start_epoch(False, model, criterion, data_loader)
        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')
        end = timer.time()
        for i, (input, target) in enumerate(data_loader):
            self.state['iteration'] = i
            self.state['data_time_batch'] = timer.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])
            self.state['input'] = input if not self.state['use_gpu'] else input.to(
                'cuda:{}'.format(self.state['device_ids'][0]))
            self.state['target'] = target if not self.state['use_gpu'] else target.to(
                'cuda:{}'.format(self.state['device_ids'][0]))
            self.on_start_batch(False, model, criterion, data_loader)
            self.on_forward(False, model, criterion, data_loader)
            self.state['batch_time_current'] = timer.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = timer.time()
            self.on_end_batch(False, model, criterion, data_loader, acc_count=True)
        score = self.on_end_epoch(False, model, criterion, data_loader)
        return score

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename = os.path.join(self.state['save_model_path'], filename)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('Save model: {}'.format(filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)

    def adjust_learning_rate(self, optimizer):
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)


###############################################################################
# 多标签训练引擎（继承自 Engine）
###############################################################################
class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])
        self.sigmoid = nn.Sigmoid()

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.cnt = 0
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map_val = self.state['ap_meter'].value().mean().item()
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{}]\tLoss {:.4f}\tmAP {:.3f}'.format(self.state['epoch'], loss, map_val))
            else:
                print('Test:\tLoss {:.4f}\tmAP {:.3f}\tacc {:.3f}'.format(loss, map_val, self.cnt))
        return map_val

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True, acc_count=False):
        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])
        if acc_count:
            for i in range(len(self.state['target_gt'])):
                output = torch.round(self.sigmoid(self.state['output'].data)[i]).type(torch.int64)
                gt = self.state['target_gt'][i].to('cuda:{}'.format(self.state['device_ids'][0])).type(torch.int64) if \
                self.state['use_gpu'] else self.state['target_gt'][i].type(torch.int64)
                if torch.equal(output, gt):
                    self.cnt += 1
        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{}][{}/{}]\tTime {:.3f} ({:.3f})\tData {:.3f} ({:.3f})\tLoss {:.4f} ({:.4f})'
                      .format(self.state['epoch'], self.state['iteration'], len(data_loader),
                              self.state['batch_time_current'], batch_time,
                              self.state['data_time_batch'], data_time,
                              self.state['loss_batch'], loss))
            else:
                print('Test: [{}/{}]\tTime {:.3f} ({:.3f})\tData {:.3f} ({:.3f})\tLoss {:.4f} ({:.4f})'
                      .format(self.state['iteration'], len(data_loader),
                              self.state['batch_time_current'], batch_time,
                              self.state['data_time_batch'], data_time,
                              self.state['loss_batch'], loss))


###############################################################################
# 获取测试结果矩阵（转换为 np.array）
###############################################################################
def get_test_predictions(model, dataset, engine_state):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    test_loader = DataLoader(dataset, batch_size=engine_state['batch_size'],
                             shuffle=False, num_workers=engine_state['workers'])
    model.eval()
    predictions = []
    with torch.no_grad():
        for input, _ in test_loader:
            if engine_state['use_gpu']:
                input = input.to('cuda:{}'.format(engine_state['device_ids'][0]))
            output = model(input)
            preds = torch.round(torch.sigmoid(output))
            predictions.append(preds.cpu())
    predictions = torch.cat(predictions, dim=0)
    return predictions.numpy()


###############################################################################
# 训练设置与选择损失函数（选择使用 CombinedLoss）
###############################################################################
# 这里选择使用三个损失函数平均组合的方式
loss_type = "combined"

if loss_type == "combined":
    # 注意：此处output_dim为模型输出类别数，需与 cls_num_list 长度一致
    cls_num_list = [100] * output_dim  # 例如，每个类别样本数均设为 100（示例）
    criterion = CombinedLoss(cls_num_list=cls_num_list, focal_gamma=2.0, reweight_CE=False,
                             max_m=0.5, s=30, reweight_epoch=-1)
elif loss_type == "focal":
    criterion = FocalLoss(gamma=2.0)
elif loss_type == "ce":
    criterion = CrossEntropyLoss()
elif loss_type == "ldam":
    cls_num_list = [100] * output_dim
    criterion = LDAMLoss(cls_num_list=cls_num_list)
else:
    raise ValueError("Unknown loss_type: " + loss_type)

optimizer = optim.Adam(model.parameters(), lr=0.001)

state = {
    'batch_size': 64,
    'max_epochs': 200,
    'print_freq': 10,
    'device_ids': [0],
    'evaluate': False,
    'save_model_path': './checkpoints',
    'epoch_step': [5],
    'workers': 0,
}

###############################################################################
# 使用 MultiLabelMAPEngine 进行多标签分类训练
###############################################################################
engine = MultiLabelMAPEngine(state)
best_score = engine.learning(model, criterion, train_dataset, val_dataset, optimizer=optimizer)
print("Training completed. Best score:", best_score)

###############################################################################
# 加载最佳模型（取最好一次的训练结果）
###############################################################################
best_model_path = os.path.join(state['save_model_path'], 'model_best.pth.tar')
if os.path.exists(best_model_path):
    print("Loading best model from:", best_model_path)
    checkpoint = torch.load(best_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
else:
    print("Best model not found. Using current model.")

###############################################################################
# 获取测试结果矩阵（0/1矩阵，转换为 np.array），并保存到文件
###############################################################################
pred_matrix = get_test_predictions(model, val_dataset, state)
print("Test Predictions (0-1 matrix as np.array):")
print(pred_matrix)

np.save('best_test_predictions.npy', pred_matrix)
print("Best test predictions saved as np.array in 'best_test_predictions.npy'")

# 调用 evaluate 函数进行评估（假设 evaluate 已在 u_evaluation 中定义）
eva = evaluate(pred_matrix, Y_test)
print(eva)
