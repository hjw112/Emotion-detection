import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

# 读取 CSV 文件并转换为 Pandas 数据框
df1 = pd.read_csv('data1.csv')
df2 = pd.read_csv('data2.csv')

# 数据清理和预处理
df1 = df1.dropna() # 删除缺失值
df2 = df2.dropna()

# 定义逻辑回归模型
model = LogisticRegression()

# 使用 10 折交叉验证计算得分
scores1 = cross_val_score(model, df1, y1, cv=10)
scores2 = cross_val_score(model, df2, y2, cv=10)

# 计算交叉验证得分的平均值和标准差
mean1, std1 = scores1.mean(), scores1.std()
mean2, std2 = scores2.mean(), scores2.std()

# 输出结果
print("数据集 1 的交叉验证得分: %0.2f (+/- %0.2f)" % (mean1, std1 * 2))
print("数据集 2 的交叉验证得分: %0.2f (+/- %0.2f)" % (mean2, std2 * 2))

