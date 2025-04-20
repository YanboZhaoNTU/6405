from u_mReadData import *
from u_base import *
import numpy as np
from u_evaluation import *
from L2 import *
from weight import *
from CCclass import *
from CCnew import *




def all_label(list):
    datasnames = ["Yeast"]
    num = 0
    result = np.array([])
    Y_tr = np.array([])
    X_te = np.array([])
    Y_te = np.array([])
    w_save = np.array([])
    X_tr_list = []
    Y_tr_list = []
    X_te_list = []
    Y_te_list = []
    save = np.array([])
    rd = ReadData(datas=datasnames, genpath='data/')
    X_train, Y_train, X_test, Y_test = rd.readData(0)
    for i in range(1629):
        j = random.randint(0, 1628)

        X_tr_list.append(X_train[j])
        Y_tr_list.append(Y_train[j])
    for i in range(1629):
        j = random.randint(0, 1628)
        X_te_list.append(X_train[j])
        Y_te_list.append(Y_train[j])

    X_tr = np.array(X_tr_list)

    Y_tr = np.array(Y_tr_list)
    X_te = np.array(X_te_list)
    Y_te = np.array(Y_te_list)
##########################################

    newCC = CCclass()
    oldCC = CCclass()
    newCC.train(X_tr, Y_tr, list)

    print(list)
    Y_tr = np.array([])
    X_te = np.array([])
    Y_te = np.array([])
    X_tr_list = []
    Y_tr_list = []
    X_te_list = []
    Y_te_list = []
###########################################
    for i in range(788):
        j = random.randint(0, 787)

        X_tr_list.append(X_test[j])
        Y_tr_list.append(Y_test[j])
    Y_tr_list = np.array(Y_tr_list)
##########################################
    result = newCC.test_BRC_test(X_tr_list, 14)

    eva = evaluate(result, Y_tr_list)
    print(eva)


def process_permutation(perm):
    """
    这是一个示例处理函数，用于接收并处理每一个排列。
    你可以根据需要修改此函数，例如将排列保存到数据库、写入文件、做进一步计算等。
    """
    # 示例：打印排列
    all_label(perm)
    print(perm)

def generate_permutations(current, remaining, callback):
    """
    递归生成排列的生成器函数。

    参数：
    - current: 当前排列
    - remaining: 剩余可选元素
    - callback: 回调函数，用于处理每一个生成的排列
    """
    if not remaining:
        callback(current)
        return
    for i in range(len(remaining)):
        next_elem = remaining[i]
        new_current = current + [next_elem]
        new_remaining = remaining[:i] + remaining[i+1:]
        generate_permutations(new_current, new_remaining, callback)

def start_permutation_generation(n, callback):
    """
    启动排列生成的主函数。

    参数：
    - n: 要生成排列的数字范围上限（从 0 到 n-1）
    - callback: 回调函数，用于处理每一个生成的排列
    """

    elements = list(range(n))
    generate_permutations([], elements, callback)

if __name__ == "__main__":
    n = 14  # 输入的数字
    start_permutation_generation(n, process_permutation)




