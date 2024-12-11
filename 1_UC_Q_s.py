# Author: PAFF
# CreatTime: 2024/12/4
# FileName: 1_UC_Q_s
import kaiwu as kw
import pandas as pd
import numpy as np  # 用于生成随机浮动
import os
import gurobipy as gp
from gurobipy import GRB

generators = 5
periods = 24
maxstart0 = 0

# 原始 L 值
L_base = [2500, 2500, 2500, 2500, 2500, 2500,
          10000, 10000, 10000,
          4000, 4000, 4000, 4000, 4000, 4000,
          13000, 13000, 13000,
          4500, 4500, 4500, 4500, 4500, 4500]
P_min = [850, 1250, 1500, 3000, 6000]
P_max = [2000, 1750, 4000, 9000, 18000]
A = [1000, 2600, 3000, 3000, 3000]
B = [2, 1.3, 3, 2.3, 3]
C = [2, 1.3, 3, 2.3, 3]
PB = 200
startup_cost = [2000, 1000, 500, 1000, 2000]
N = 10

# 创建输出目录
out_dir = 'data_uc'
os.makedirs(out_dir, exist_ok=True)

# 循环生成 100 个算例
for case_id in range(1, 101):
    print(f"正在生成第 {case_id} 个算例...")

    # 为当前算例生成新的 L 值（基于原始 L 值，按上下 30% 浮动）
    L = [int(l * np.random.uniform(0.7, 1.3)) for l in L_base]

    h = []
    s = {}

    for i in range(len(P_min)):
        h.append((P_max[i] - P_min[i]) / N)

    for i in range(len(P_min)):
        for k in range(1, N + 1):
            for t in range(len(L)):
                s[(f'bs{i}_{k}_{t}')] = kw.qubo.Binary(f's{i}_{k}_{t}')

    con = 0
    for t in range(len(L)):
        for i in range(len(P_min)):
            for k in range(1, N + 1):
                con += A[i] * (s[(f'bs{i}_{k}_{t}')] + 1) / 2 + B[i] * (P_min[i] + (k - 1) * h[i]) * (s[(f'bs{i}_{k}_{t}')] + 1) / 2
        con += PB * (L[t] - kw.qubo.quicksum([kw.qubo.quicksum([(P_min[i] + (k - 1) * h[i]) * (s[(f'bs{i}_{k}_{t}')] + 1) / 2
                                                              for k in range(1, N + 1)]) for i in range(len(P_min))])) ** 2
    obj = con
    data = obj['coefficient']

    variables_order = list(s.keys())

    # 创建空矩阵
    df_matrix = pd.DataFrame(0, index=variables_order, columns=variables_order)
    df_row = pd.DataFrame(0, index=variables_order, columns=['Value'])

    # 处理数据
    for key, value in data.items():
        if len(key) == 2:
            row_var, col_var = key
            if row_var in df_matrix.index and col_var in df_matrix.columns:
                df_matrix.loc[row_var, col_var] = value
            else:
                print(f"警告: 变量 '{row_var}' 或 '{col_var}' 不存在于变量顺序中。")
        elif len(key) == 1:
            var = key[0]
            if var in df_row.index:
                df_row.loc[var, 'Value'] += value
            else:
                print(f"警告: 变量 '{var}' 不存在于变量顺序中。")
        else:
            print(f"警告: 键 {key} 的长度不符合预期。")

    # 保存每个算例的 CSV 文件
    csv_matrix_filename = os.path.join(out_dir, f'Q_UC_N10_matrix_case{case_id}.csv')
    csv_row_filename = os.path.join(out_dir, f'Q_UC_N10_row_case{case_id}.csv')

    model = gp.Model()
    b_uc = {}
    for i in range(len(P_min)):
        for k in range(1, N + 1):
            for t in range(len(L)):
                b_uc[(f'bs{i}_{k}_{t}')] = model.addVar(vtype = GRB.BINARY, name = f'b{i}_{k}_{t}')

    model.addConstrs(gp.quicksum((P_min[i] + (k - 1) * h[i]) * b_uc[(f'bs{i}_{k}_{t}')] for k in range(1, N + 1)
                     for i in range(len(P_min))) >= L[t] for t in range(len(L)))
    obj_gp = 0
    for t in range(len(L)):
        for i in range(len(P_min)):
            for k in range(1, N + 1):
                obj_gp += A[i] * b_uc[(f'bs{i}_{k}_{t}')] + B[i] * (P_min[i] + (k - 1) * h[i]) * b_uc[(f'bs{i}_{k}_{t}')]
    model.setObjective(obj_gp)
    model.optimize()

    # 检查模型是否找到最优解
    if model.status == GRB.OPTIMAL:
        # 获取变量解
        solution = {var.varName: var.x for var in model.getVars()}

        # 将解保存为 DataFrame
        solution_df = pd.DataFrame(solution.items(), columns=["Variable", "Value"])

        # 保存解为 CSV 文件
        solution_df.to_csv(f"{out_dir}/solution_case_{case_id}.csv", index=False)
        print(f"最优解已保存为 solution_case_{case_id}.csv")
    else:
        print(f"未找到最优解，状态码: {model.status}")


    df_matrix.to_csv(csv_matrix_filename)
    df_row.to_csv(csv_row_filename)

    print(f"第 {case_id} 个算例矩阵已成功保存为 '{csv_matrix_filename}'。")
    print(f"第 {case_id} 个算例单行矩阵已成功保存为 '{csv_row_filename}'。")

print("所有算例已生成完成！")

