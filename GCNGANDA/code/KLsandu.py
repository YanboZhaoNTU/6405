from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats
from u_mReadData import *
from u_base import *
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *

datasnames = ["20NG"]
rd = ReadData(datas=datasnames, genpath='data/')

X_train, Y_train, X_test, Y_test = rd.readData(0)
print(np.shape(X_train), np.shape(Y_train), np.shape(X_test), np.shape(Y_test))

# 从训练数据中抽取一小部分作为标记数据
X_train_label, X_temp, Y_train_label, y_temp = train_test_split(X_train, Y_train, test_size=0.99, random_state=42)

# 用标记数据训练一个初步模型，并对未标记数据生成伪标签
BRt = BRclass()
BRt.BRC_train(X_train_label, Y_train_label)
Plabel = BRt.BRC_test(X_temp)

Y_train_label = np.array(Y_train_label)
Plabel = np.array(Plabel)

print("标记数据标签形状:", np.shape(Y_train_label))
print("伪标签形状:", np.shape(Plabel))

# 计算标记数据与伪标签的类别分布（假设标签为 one-hot 或概率分布）
p_dist = np.mean(Y_train_label, axis=0)
q_dist = np.mean(Plabel, axis=0)

# 计算 KL 散度
kl_div = scipy.stats.entropy(p_dist, q_dist)
print("KL divergence:", kl_div)

# 根据 KL 散度确定修正系数 alpha（这里采用简单的截断方法作为示例）
alpha = min(kl_div, 1.0)
# 修正伪标签，使其更接近标记数据的分布
Plabel_adjusted = (1 - alpha) * Plabel + alpha * p_dist

# 将修正后的伪标签转换为二值标签（离散标签），例如以 0.5 为阈值
Plabel_adjusted_binary = (Plabel_adjusted > 0.7).astype(int)

# 将原始标记数据和转换后的伪标签合并
Y_p_train = np.vstack([Y_train_label, Plabel_adjusted_binary])

# 使用全量数据（包括离散化后的伪标签）训练最终模型
TRt = BRclass()
TRt.BRC_train(X_train, Y_p_train)
test_result = TRt.BRC_test(X_test)
print("预测结果形状:", np.shape(test_result))
print("真实标签形状:", np.shape(Y_test))

eva = evaluate(test_result, Y_test)
print(eva)
