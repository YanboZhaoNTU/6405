from newData import *
from newLP import *



def chunk_list(num, n, randomize=False):
    """
    将 0 ~ num-1 的整数序列分割成若干个子列表，每个子列表长度为 n（最后一个子列表长度可能小于 n）。
    如果 randomize=True，会先对序列进行洗牌（乱序）。

    :param num:   生成的整数范围是 [0, num)
    :param n:     每个子列表中元素的个数
    :param randomize: 是否对生成的序列进行乱序，默认为 False
    :return:      一个由子列表组成的列表
    """
    # 生成 0 ~ num-1 的列表
    arr = list(range(num))

    # 如果需要乱序，则进行洗牌
    if randomize:
        random.shuffle(arr)

    # 分段
    result = []
    for i in range(0, num, n):
        result.append(arr[i: i + n])
    return result

def train(data):

    data.readData()
    LP = LabelPowersetLogistic()
    X_train = data.getoriginTrainData()
    Y_train = data.getoriginTrainLabel()
    X_test = data.getoriginTestData()
    Y_test = data.getoriginTestLabel()

    class_label_number = 9
    num = Y_train[0].shape[0]

    times = num / class_label_number
    times = round(times)
    star = 0
    end = star + class_label_number
    n = 0
    result = np.array([])

    list = chunk_list(14, 9, False)
 #   list = [[13,11,12],[0,5,6],[1,2,8],[7,3,10],[4,9]]

    for i in range(times):

        X = X_train
        # Y_train_selected = Y_train[:, cols_to_select]
        #   Y = Y_train[:,star:end]
        Y = Y_train[:, list[i]]

        LP.fit(X, Y)

        LP.shape_train(X_test)
        if n == 0:
            result = LP.predict(X_test, i)
            n = n + 1
        else:
            result = np.hstack([result, LP.predict(X_test, i)])
        star = star + class_label_number
        end = end + class_label_number

    real = np.array([])
    n = 0
    for i in range(times):
        if n == 0:
            real = Y_test[:, list[i]]
            n = n + 1
        else:
            real = np.hstack([real, Y_test[:, list[i]]])
    eva = evaluate(result, real)


    return [eva[0], list]
data = data()
result = []
nss = 0
for i in range(2):

    result = train(data)

    v = result[0]
 #   v = float(v[0])

    if v > nss:
        final = result
        nss = v

print(result)

