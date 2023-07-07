import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'data1/features1.csv'

all_data = []
labels = []
data = pd.read_csv(data_path)
data = data[data['0'].isin(['HAP', 'DIS'])]
all_data.append(data.iloc[:, 1:])
labels.extend(data.iloc[:, 0])

X = pd.concat(all_data).values
y = np.array(labels)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 定义 SVM 模型
svm = SVC(probability=True)

# 定义要搜索的超参数组合
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],  # 仅在 kernel='poly' 时有效
    'gamma': ['scale', 'auto', 0.1, 1, 10],  # 仅在 kernel='rbf', 'poly', 'sigmoid' 时有效
    'coef0': [0.0, 0.1, 0.5, 1]  # 仅在 kernel='poly', 'sigmoid' 时有效
}


# 使用 SMOTE 对数据进行过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 使用 GridSearchCV 对超参数进行搜索
#stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#grid_search = GridSearchCV(svm, param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=-1, verbose=2)

####################
scaler = StandardScaler()
svm = SVC(kernel='poly')
pipeline = Pipeline([('scaler', scaler), ('svm', svm)])

param_grid = {'svm__C': np.logspace(-3, 2, 6), 'svm__gamma': np.logspace(-3, 2, 6)}
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 从 GridSearchCV 结果中提取需要的信息
cv_results = pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score']]

# 将 'params' 字典列拆分为单独的列
params_df = cv_results['params'].apply(pd.Series)
cv_results = pd.concat([params_df, cv_results['mean_test_score']], axis=1)

# 绘制准确率图表
fig, ax = plt.subplots(figsize=(12, 6))

# 在 x 轴上遍历 C 值，y 轴上遍历 gamma 值
for gamma in cv_results['svm__gamma'].unique():
    subset = cv_results[cv_results['svm__gamma'] == gamma]
    ax.plot(subset['svm__C'], subset['mean_test_score'], label=f'Gamma: {gamma}', marker='o')

ax.set_xscale('log')
ax.set_xlabel('C')
ax.set_ylabel('Mean Test Score')
ax.set_title('Accuracy for Different Parameter Combinations')
ax.legend()
plt.show()
####################
'''
grid_search.fit(X_resampled, y_resampled)

# 获取参数组合和对应的平均交叉验证准确率
parameters = grid_search.cv_results_['params']
mean_test_scores = grid_search.cv_results_['mean_test_score']

# 将参数和准确率整合为一个 DataFrame
result_df = pd.DataFrame(parameters)
result_df['mean_test_score'] = mean_test_scores

# 可视化，将参数列表转换为多列
result_df = pd.concat([result_df.drop('kernel', axis=1), pd.get_dummies(result_df['kernel'], prefix='kernel')], axis=1)

# 绘制热力图
plt.figure(figsize=(12, 6))
sns.heatmap(result_df.pivot_table(values='mean_test_score', index=['C'], columns=['kernel_linear', 'kernel_poly', 'kernel_rbf']), annot=True, fmt=".3f", cmap='viridis')
plt.xlabel('Kernel')
plt.ylabel('C')
plt.title('Grid Search Results (Accuracy)')
plt.show()
'''

# 输出最佳超参数组合和相应的准确率
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# 使用最佳参数重新训练 SVM 模型
best_svm = grid_search.best_estimator_

# 在测试集上评估模型
accuracy = best_svm.score(X_test, y_test)
print("Test set accuracy: ", accuracy)
