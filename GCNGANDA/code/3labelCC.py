import numpy as np
from CCnew import *
from data import *
from CCclass import *

def normalCC(X_tr,Y_tr,X_te,Y_te):
    CC = CCclass()
    CC.train(X_tr, Y_tr, 0, 14)
    result = CC.test_BRC_test(X_te, 0,14)
    return result

def all_label(list,X_tr,Y_tr,X_te,Y_te):
    ##########################################

    newCC = newCCclass()

    newCC.train(X_tr, Y_tr, list)
    result = newCC.test_BRC_test(X_te, len(list))


    return result



def split_into_batches(total, batch_size, shuffle=False):
    """
    将数字从 1 到 total 分成多个批次，每个批次包含 batch_size 个数字。

    参数:
    - total (int): 总数字数量。
    - batch_size (int): 每个批次的数量。
    - shuffle (bool): 是否在分批前随机打乱数字顺序。

    返回:
    - List[List[int]]: 分批后的数字列表。
    """
    import random

    # 创建数字列表
    numbers = list(range(0, total))

    # 如果需要，打乱列表
    if shuffle:
        random.shuffle(numbers)
        print(f"打乱后的数字列表: {numbers}")

    # 分批
    batches = [numbers[i:i + batch_size] for i in range(0, total, batch_size)]

    return batches


# 示例使用
if __name__ == "__main__":
    total_number = 14  # 总数量
    draw_size = 3  # 每次抽取的数量
    shuffle_numbers = False  # 是否打乱数字顺序
    data = data()
    batches = split_into_batches(total_number, draw_size, shuffle_numbers)
    data.train_Data()
    data.test_Data()
    X_tr = data.TrainX()
    Y_tr = data.TrainY()
    X_te = data.TestX()
    Y_te = data.TestY()
    result  = np.array([])
    num = 0
    for idx, batch in enumerate(batches, 1):
#        Y_tr = Y_tr[:, batch]
#        Y_te = Y_te[:, batch]
        if num == 0:
            result = all_label(batch,X_tr,Y_tr,X_te,Y_te)
            num += 1
        else:

            print(f"第 {idx} 次抽取: {batch}")
            r = all_label(batch,X_tr,Y_tr,X_te,Y_te)
            print(f"第 {idx} 次抽取: {batch}")

            result = np.column_stack((result, r))
    eva = evaluate(result, Y_te)
    print(eva)

    normalresult = normalCC(X_tr,Y_tr,X_te,Y_te)
    eva = evaluate(normalresult, Y_te)
    print(eva)
#        Y_tr = data.TrainY()
#        Y_te = data.TestY()
