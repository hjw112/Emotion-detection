from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class EmotionClassifierSVM:
    def __init__(self, data_path):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True)

    def load_data(self):
        all_data = []
        labels = []

        # 读取所有 CSV 文件

        data = pd.read_csv(self.data_path)
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

        return X_train, X_test, y_train, y_test

    def train(self):
        X_train, X_test, y_train, y_test = self.load_data()
        self.svm.fit(X_train, y_train)
        y_pred = self.svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def predict(self, feature_vector):
        prediction = self.svm.predict_proba(feature_vector.reshape(1, -1))
        class_index = np.argmax(prediction)
        return self.label_encoder.inverse_transform([class_index])[0]


data_path = 'data1/features.csv'
# 创建 EmotionClassifierSVM 的实例®
emotion_classifier_svm = EmotionClassifierSVM(data_path)
# 训练 SVM 分类器
accuracy = emotion_classifier_svm.train()
print(f"Accuracy on test data: {accuracy * 100:.2f}%")
# 使用训练好的 SVM 分类器进行预测
