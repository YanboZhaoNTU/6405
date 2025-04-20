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
print(np.shape(X_train), np.shape(Y_train), np.shape(X_test), np.shape(Y_test))

# 迭代次数
num_iter = 5

for i in range(num_iter):
    print("Iteration:", i + 1)

    # 使用不同的 random_state 进行数据划分
    X_train_label, X_temp, Y_train_label, _ = train_test_split(
        X_train, Y_train, test_size=0.99, random_state=i)

    # 第一阶段：用少量标记数据训练并生成伪标签
    BRt = BRclass()
    BRt.BRC_train(X_train_label, Y_train_label)
    Plabel = BRt.BRC_test(X_temp)

    Y_train_label = np.array(Y_train_label)
    Plabel = np.array(Plabel)

    # 合并真实标签与伪标签
    Y_p_train = np.vstack([Y_train_label, Plabel])

    # 第二阶段：使用所有训练数据（真实标签 + 伪标签）训练并测试
    TRt = BRclass()
    TRt.BRC_train(X_train, Y_p_train)
    test_result = TRt.BRC_test(X_test)

    eva = evaluate(test_result, Y_test)
    print("Evaluation:", eva)
