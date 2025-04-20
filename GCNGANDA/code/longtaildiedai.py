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
    具体步骤：
      1. 计算目标数据（如测试集）的均值和标准差
      2. 计算源数据（如训练集）的均值和标准差
      3. 将源数据先标准化，再根据目标数据的均值和标准差进行缩放和平移
    """
    # 计算目标数据的均值和标准差（防止除零，加上微小值）
    target_mean = np.mean(X_target, axis=0)
    target_std = np.std(X_target, axis=0) + 1e-5

    # 计算源数据的均值和标准差
    source_mean = np.mean(X_source, axis=0)
    source_std = np.std(X_source, axis=0) + 1e-5

    # 标准化源数据后调整到目标分布
    X_source_aligned = (X_source - source_mean) * (target_std / source_std) + target_mean

    return X_source_aligned


# ------------------- 数据读取 -------------------
datasnames = ["20NG"]
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)
print("原始数据形状:", np.shape(X_train), np.shape(Y_train), np.shape(X_test), np.shape(Y_test))

# ------------------- 分布对齐 -------------------
# 利用测试数据的分布对齐训练数据
X_train_aligned = distribution_alignment(X_train, X_test)

# ------------------- 划分带标签数据与未标记数据 -------------------
# 这里设定仅取极少一部分真实标签数据，其余作为待生成伪标签数据（比例可根据实际需求调整）
X_train_label, X_unlabeled, Y_train_label, _ = train_test_split(
    X_train_aligned, Y_train, test_size=0.99, random_state=42
)
print("初始带标签数据形状:", np.shape(X_train_label))
print("初始未标记数据形状:", np.shape(X_unlabeled))

# ------------------- 迭代自训练过程 -------------------
n_iterations = 5  # 最大迭代次数
frac_to_add = 0.1  # 每轮迭代中从未标记数据中加入带伪标签数据的比例

for i in range(n_iterations):
    print(f"\n====== 第 {i + 1} 轮迭代 ======")

    # 1. 利用当前带标签数据训练模型
    model = BRclass()
    model.BRC_train(X_train_label, Y_train_label)

    # 2. 利用模型对剩余未标记数据生成伪标签
    pseudo_labels = model.BRC_test(X_unlabeled)
    pseudo_labels = np.array(pseudo_labels)

    # 3. 从未标记数据中按比例选取一部分样本加入带标签数据集
    num_to_add = max(1, int(len(X_unlabeled) * frac_to_add))
    if num_to_add > len(X_unlabeled):
        num_to_add = len(X_unlabeled)

    X_batch = X_unlabeled[:num_to_add]
    Y_batch = pseudo_labels[:num_to_add]

    # 将选取的伪标签数据加入带标签数据集中
    X_train_label = np.concatenate((X_train_label, X_batch), axis=0)
    Y_train_label = np.concatenate((Y_train_label, Y_batch), axis=0)

    # 从未标记集中移除已加入的样本
    X_unlabeled = X_unlabeled[num_to_add:]

    print("当前带标签数据集大小:", np.shape(X_train_label))
    print("剩余未标记数据集大小:", np.shape(X_unlabeled))

    # 4. 利用当前模型对测试数据进行预测并评估
    test_result = model.BRC_test(X_test)
    eva = evaluate(test_result, Y_test)
    print(f"第 {i + 1} 轮迭代测试评估:", eva)

    # 若所有未标记数据均已用尽，则退出迭代
    if len(X_unlabeled) == 0:
        print("未标记数据已全部使用。")
        break

# ------------------- 对剩余未标记数据进行最终伪标签生成（若有） -------------------
if len(X_unlabeled) > 0:
    print("\n对剩余未标记数据进行最终伪标签生成。")
    model = BRclass()
    model.BRC_train(X_train_label, Y_train_label)
    remaining_pseudo = model.BRC_test(X_unlabeled)
    X_train_label = np.concatenate((X_train_label, X_unlabeled), axis=0)
    Y_train_label = np.concatenate((Y_train_label, np.array(remaining_pseudo)), axis=0)

# ------------------- 最终模型训练与测试 -------------------
final_model = BRclass()
final_model.BRC_train(X_train_label, Y_train_label)
final_test_result = final_model.BRC_test(X_test)
final_eva = evaluate(final_test_result, Y_test)
print("\n最终测试评估:", final_eva)
