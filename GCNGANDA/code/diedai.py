from sklearn.model_selection import train_test_split
from u_mReadData import *
from u_base import *
import numpy as np
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *




# 读取数据
datasnames = ["20NG"]
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)
print("数据维度：", np.shape(X_train), np.shape(Y_train), np.shape(X_test), np.shape(Y_test))

# 初始划分：将训练数据分为小的有标签集和大量未标记集
X_train_label, X_unlabeled, Y_train_label, _ = train_test_split(
    X_train, Y_train, test_size=0.99, random_state=42)
print("初始有标签集大小：", np.shape(X_train_label), np.shape(Y_train_label))
print("初始未标记集大小：", np.shape(X_unlabeled))

# 设定迭代参数
n_iterations = 5  # 最大迭代次数
frac_to_add = 0.1  # 每次迭代从未标记集中加入的比例

# 迭代自训练过程
for i in range(n_iterations):
    print(f"\n====== 第 {i + 1} 轮迭代 ======")

    # 1. 用当前有标签数据训练模型
    model = BRclass()
    model.BRC_train(X_train_label, Y_train_label)

    # 2. 利用当前模型预测未标记数据的标签
    pseudo_labels = model.BRC_test(X_unlabeled)
    pseudo_labels = np.array(pseudo_labels)

    # 3. 从未标记集中选取部分数据（例如 10%）加入有标签集
    num_to_add = max(1, int(len(X_unlabeled) * frac_to_add))
    if num_to_add > len(X_unlabeled):
        num_to_add = len(X_unlabeled)

    X_batch = X_unlabeled[:num_to_add]
    Y_batch = pseudo_labels[:num_to_add]

    # 更新有标签数据集（增加伪标签数据）
    X_train_label = np.concatenate((X_train_label, X_batch), axis=0)
    Y_train_label = np.concatenate((Y_train_label, Y_batch), axis=0)

    # 从未标记集中剔除已加入的样本
    X_unlabeled = X_unlabeled[num_to_add:]

    print("当前有标签集大小：", np.shape(X_train_label))
    print("剩余未标记集大小：", np.shape(X_unlabeled))

    # 4. 选用当前模型对测试数据进行预测并评估
    test_result = model.BRC_test(X_test)
    eva = evaluate(test_result, Y_test)
    print(f"第 {i + 1} 轮迭代测试评估：", eva)

    # 如果未标记数据已用完，则退出迭代
    if len(X_unlabeled) == 0:
        print("所有未标记数据均已加入。")
        break

# 最终利用所有“标记”数据重新训练模型并测试
final_model = BRclass()
final_model.BRC_train(X_train_label, Y_train_label)
final_test_result = final_model.BRC_test(X_test)
final_eva = evaluate(final_test_result, Y_test)
print("\n最终测试评估：", final_eva)
