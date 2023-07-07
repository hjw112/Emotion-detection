import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

filepath = 'data1/features1.csv'
all_data = []
labels = []
data = pd.read_csv(filepath)
all_data.append(data.iloc[:, 1:])
labels.extend(data.iloc[:, 0])
# 合并数据和标签
X = pd.concat(all_data).values
y = np.array(labels)

# 对标签进行编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

###############################使用默认的决策树参数
'''
# 创建决策树分类器
clf = DecisionTreeClassifier(random_state=42)
# 训练模型
clf.fit(X_train, y_train)
# 预测测试集的标签
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("决策树模型在测试集上的准确率：", accuracy)
'''
###############################

###############################参数网络
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
}
clf = DecisionTreeClassifier(random_state=42)

#创建并执行网格搜索
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
#输出最佳参数组合
print("最佳参数组合：", grid_search.best_params_)
#使用最佳参数训练决策树模型
best_clf = grid_search.best_estimator_
joblib.dump(best_clf, 'emotion_classifier_dt.pkl')

#提取结果并转化为DataFrame
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results[['params', 'mean_test_score']]

#为每个参数创建新列
cv_results['criterion'] = cv_results['params'].apply(lambda x: x['criterion'])
cv_results['max_depth'] = cv_results['params'].apply(lambda x: x['max_depth'])
cv_results['min_samples_split'] = cv_results['params'].apply(lambda x: x['min_samples_split'])
cv_results['min_samples_leaf'] = cv_results['params'].apply(lambda x: x['min_samples_leaf'])
cv_results.drop('params', axis=1, inplace=True)

#将DataFrame转化为热力图
heatmap_data = cv_results.pivot_table(
    index=['criterion', 'max_depth'],
    columns=['min_samples_split', 'min_samples_leaf'],
    values='mean_test_score'
)

#使用seaborn画图
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f')
plt.title('Cross-validation')
plt.xlabel('min_samples_split and min_samples_leaf')
plt.ylabel('criterion and max_depth')
plt.show()
