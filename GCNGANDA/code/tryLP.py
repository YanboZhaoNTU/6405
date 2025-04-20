import random
import numpy as np

from sklearn.linear_model import LogisticRegression


def generate_random_multilabel_data(num_samples=20, num_features=2, num_possible_labels=3):
    """
    随机生成多标签数据:
      X: 大小 [num_samples, num_features] 的特征向量 (numpy 数组)
      Y: list[set], 每个样本对应一个“标签集合”
    参数:
      - num_samples: 样本数
      - num_features: 每个样本的特征维度
      - num_possible_labels: 总共有多少种单独标签（从 0 到 num_possible_labels-1）
    返回:
      - X: np.ndarray, shape=[num_samples, num_features]
      - Y: list[set], 每个元素是一个标签集合
    """
    X = []
    Y = []
    for _ in range(num_samples):
        # 随机生成特征向量
        features = [random.random() * 10.0 for _ in range(num_features)]
        X.append(features)
        # 随机生成标签集合（每个标签以一定概率出现）
        labels = set()
        for label_id in range(num_possible_labels):
            # 这里设置一个简单概率阈值 0.5，决定是否包含某个标签
            if random.random() < 0.5:
                labels.add(label_id)
        Y.append(labels)
    return np.array(X), Y


def create_label_powerset_mapping(Y):
    """
    为 Label Powerset 构造双向映射:
    1) set_of_labels -> int
    2) int -> set_of_labels

    这里仅对在数据里实际出现过的标签集合进行编码。

    参数:
      - Y: list[set]，多标签集合列表
    返回:
      - set2int: dict, {frozenset(...) -> class_id}
      - int2set: dict, {class_id -> set(...)}
    """
    unique_label_sets = set(frozenset(lbls) for lbls in Y)  # 去重
    set2int = {}
    int2set = {}
    for idx, label_set in enumerate(unique_label_sets):
        set2int[label_set] = idx
        int2set[idx] = set(label_set)
    return set2int, int2set


def encode_label_powerset(Y, set2int):
    """
    根据 set2int 的映射，将多标签集合转换为整型 class_id
    """
    y_encoded = []
    for labels in Y:
        class_id = set2int[frozenset(labels)]  # frozenset 用于在 dict 中查找
        y_encoded.append(class_id)
    return np.array(y_encoded)  # 返回 numpy 数组


def decode_label_powerset(y_pred, int2set):
    """
    将模型预测得到的多分类结果 y_pred (class_id) 解码回原先的标签集合
    """
    decoded = []
    for class_id in y_pred:
        decoded.append(int2set[class_id])
    return decoded


# ---------------------------
#  主流程：LabelPowerset + LogisticRegression
# ---------------------------
if __name__ == "__main__":
    random.seed(0)

    # 1) 生成随机多标签数据
    X, Y = generate_random_multilabel_data(
        num_samples=2000,  # 样本数
        num_features=103,  # 特征维度
        num_possible_labels=14  # 标签总数(0, 1, 2)
    )


    # 2) 构建 Label Powerset 映射
    set2int, int2set = create_label_powerset_mapping(Y)
    print("\nLabel Powerset 映射:")
    for s, c_id in set2int.items():
        print(f"  标签集合 {set(s)} -> 类别ID {c_id}")

    # 3) 将多标签集合编码为单一多分类ID
    y_encoded = encode_label_powerset(Y, set2int)
    K = len(set2int)  # 不同“标签集合”的种类数

    # 4) 使用 sklearn 的 LogisticRegression 训练（多分类）
    #    solver 可根据特征维度和数据大小选择，这里用 lbfgs 即可
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    clf.fit(X, y_encoded)

    # 5) 预测并解码
    y_pred = clf.predict(X)  # 输出每个样本预测的类 ID
    Y_pred_decoded = decode_label_powerset(y_pred, int2set)

    # 打印结果
    print("\n预测结果 (前 5 条)：")
    for i in range(5):
        print(f"样本 {i}: 真实标签 {Y[i]}, 预测标签 {Y_pred_decoded[i]}")

    print("\n注意：本示例使用随机数据，仅作演示 LabelPowerset + LogisticRegression 的实现流程。")
