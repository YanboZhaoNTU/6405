from sklearn.model_selection import train_test_split
from u_mReadData import *
from u_base import *
import numpy as np
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *


def distribution_alignment(X_source, X_target):
    """
    利用目标数据的统计信息（均值和标准差）对源数据进行分布对齐。

    参数:
      X_source: 待对齐的数据矩阵，形状为 [样本数, 特征数]
      X_target: 目标数据矩阵，形状为 [样本数, 特征数]

    返回:
      X_source_aligned: 对齐后的源数据，其统计分布接近目标数据
    """
    target_mean = np.mean(X_target, axis=0)
    target_std = np.std(X_target, axis=0) + 1e-6

    source_mean = np.mean(X_source, axis=0)
    source_std = np.std(X_source, axis=0) + 1e-6

    X_source_aligned = (X_source - source_mean) * (target_std / source_std) + target_mean
    return X_source_aligned


# ------------------- 数据读取 -------------------
datasnames = ["20NG"]
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)
print("训练数据形状:", np.shape(X_train), np.shape(Y_train))
print("测试数据形状:", np.shape(X_test), np.shape(Y_test))

# 划分有标签与无标签数据（这里有标签数据约占20%）
X_train_l, X_train_ul, Y_train_l, Y_train_ul = train_test_split(
    X_train, Y_train, test_size=0.8, random_state=42)

# 设定迭代次数
num_iter = 5

# 初始训练：只用有标签数据训练初始分类器
classifier = BRclass()
classifier.BRC_train(X_train_l, Y_train_l)

for iter in range(num_iter):
    print(f"\n=== 迭代 {iter + 1}/{num_iter} ===")

    # 用当前模型对无标签数据进行伪标签预测
    pseudo_labels = classifier.BRC_test(X_train_ul)
    print("原始伪标签预测形状:", np.shape(pseudo_labels))

    # 保证预测输出为二维数组，与有标签数据维度一致
    if pseudo_labels.ndim == 1:
        pseudo_labels = np.expand_dims(pseudo_labels, axis=-1)

    # 检查伪标签列数是否与有标签数据一致，否则尝试复制扩展
    if pseudo_labels.shape[1] != Y_train_l.shape[1]:
        print("警告: 伪标签维度与有标签数据不匹配，尝试复制扩展伪标签。")
        pseudo_labels = np.repeat(pseudo_labels, Y_train_l.shape[1], axis=1)

    # 如果维度匹配，则进行分布对齐
    if pseudo_labels.shape[1] == Y_train_l.shape[1]:
        pseudo_labels_aligned = distribution_alignment(pseudo_labels, Y_train_l)
    else:
        pseudo_labels_aligned = pseudo_labels  # 若仍不匹配，则直接使用原始伪标签

    print("对齐后伪标签形状:", np.shape(pseudo_labels_aligned))

    # 阈值化处理：将连续值转换为离散标签（此处阈值设置为0.7，可根据实际情况调整）
    pseudo_labels_aligned = (pseudo_labels_aligned >= 0.7).astype(int)

    # 合并有标签数据与经过处理的伪标签构建新的训练标签集
    Y_p_train = np.vstack([Y_train_l, pseudo_labels_aligned])
    print("合并后的训练标签形状:", np.shape(Y_p_train))

    # 重新训练分类器：用整个训练数据（有标签+无标签部分）和更新后的标签重新训练
    classifier = BRclass()  # 这里选择重新初始化模型，也可改为增量训练
    classifier.BRC_train(X_train, Y_p_train)

    # 对测试数据进行预测并评估
    test_result = classifier.BRC_test(X_test)
    eva = evaluate(test_result, Y_test)
    print("测试评价指标:", eva)
