#先统计有相同特征的两标签数量
import itertools
from data import *
import pulp
from collections import defaultdict


import pulp
from collections import defaultdict
import numpy as np

import pulp
from collections import defaultdict

import pulp
from collections import defaultdict


def find_best_combinations(combinations_with_values):
    """
    找到一组不重叠的标签组合，使得这些组合的总值最大。

    参数:
    - combinations_with_values: List，按顺序包含组合和对应的值。
      例如: [ (0,1,2), 0.5571, (0,1,3), 0.5678, ... ]

    返回:
    - selected_combinations: List，选中的最佳组合，每个组合是标签的元组。
    - selected_values: List，对应每个选中组合的值。
    """

    # 检查输入列表长度是否为偶数
    if len(combinations_with_values) % 2 != 0:
        raise ValueError("输入列表长度必须为偶数，组合和对应的值应成对出现。")

    # 处理组合和对应的值
    processed_combinations = []
    for i in range(0, len(combinations_with_values), 2):
        combo = combinations_with_values[i]
        value = combinations_with_values[i + 1]

        # 确保组合是一个包含3个标签的元组
        if not (isinstance(combo, tuple) and len(combo) == 7):
            print(f"忽略无效的组合: {combo}（必须是包含3个标签的元组）")
            continue

        # 确保标签唯一
        if len(set(combo)) != 7:
            print(f"忽略包含重复标签的组合: {combo}")
            continue

        # 添加到处理后的组合列表中，确保标签排序一致
        processed_combinations.append((tuple(sorted(combo)), value))

    if not processed_combinations:
        raise ValueError("没有找到包含3个不重复标签的有效组合。")

    # 建立标签到组合的映射
    label_to_combos = defaultdict(list)
    for idx, (combo, _) in enumerate(processed_combinations):
        for label in combo:
            label_to_combos[label].append(idx)

    # 创建一个线性规划问题，目标是最大化总值
    prob = pulp.LpProblem("Maximize_Total_Value", pulp.LpMaximize)

    # 创建决策变量
    decision_vars = [pulp.LpVariable(f"Combo_{i}", cat='Binary') for i in range(len(processed_combinations))]

    # 设置目标函数：最大化所有选中组合的总值
    prob += pulp.lpSum(
        [processed_combinations[i][1] * decision_vars[i] for i in range(len(processed_combinations))]), "Total_Value"

    # 添加约束：每个标签最多出现在一个组合中
    for label, combos in label_to_combos.items():
        prob += pulp.lpSum([decision_vars[i] for i in combos]) <= 1, f"Label_{label}_constraint"

    # 选择求解器并求解问题
    solver = pulp.PULP_CBC_CMD(msg=False)  # 设置 msg=True 可以看到求解过程
    prob.solve(solver)

    # 检查求解状态
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise Exception(f"求解失败，状态: {pulp.LpStatus[prob.status]}")

    # 提取被选中的组合及其值
    selected_combinations = []
    selected_values = []
    for i in range(len(processed_combinations)):
        if pulp.value(decision_vars[i]) == 1:
            selected_combinations.append(processed_combinations[i][0])
            selected_values.append(processed_combinations[i][1])

    return selected_combinations, selected_values


def generate_combinations(n, k):
    """
    生成所有从n个元素中选取k个元素的组合。

    :param n: 总元素数量
    :param k: 每个组合的元素数量
    :return: 生成器，包含所有组合
    """
    elements = list(range(0, n))  # 生成元素列表，例如 [1, 2, ..., 14]
    combinations = itertools.combinations(elements, k)
    return combinations

def all_label(list,X_tr,Y_tr,X_te):
    ##########################################

    newCC = newCCclass()

    newCC.train(X_tr, Y_tr, list)
    result = newCC.test_BRC_test(X_te, len(list))


    return result

def find(list, Y_te):
    selected_columns = Y_te[:, list]
    return selected_columns

def print_combinations(combinations,X_tr,Y_tr,X_pr,Y_pr,X_te,Y_te):
    """
    打印所有组合，并统计总数。

    :param combinations: 组合生成器
    """
    li = []

    count = 0
    for combo in combinations:
        result = all_label(combo,X_tr,Y_tr,X_te)
        real = find(combo, Y_te)
        eva = evaluate(result, real)

        li.append(combo)
        li.append(eva[1])

    best = find_best_combinations(li)
    print(best)
    v = 0
    re = np.array([])
    for i in best[0]:
        print(i)
        if v == 0:
            re = all_label(i,X_tr,Y_tr,X_te)
            v = v+1
        else:
            res = all_label(i,X_tr,Y_tr,X_te)
            re = np.column_stack((re, res))

    return re

if __name__ == "__main__":
    n = 14
    k = 7

    combinations = generate_combinations(n, k)
    data = data()
    data.train_Data()
    data.test_Data()
    X_tr = data.TrainX()
    Y_tr = data.TrainY()
    X_pr = data.PredX()
    Y_pr = data.PredY()
    X_te = data.TestX()
    Y_te = data.TestY()
    r = print_combinations(combinations,X_tr,Y_tr,X_pr,Y_pr,X_te,Y_te)

    eva = evaluate(r, Y_te)
    print(eva)
