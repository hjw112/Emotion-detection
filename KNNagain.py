import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

class EmotionClassifierKNN:
    def __init__(self,  k=10):
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.label_encoder = LabelEncoder()

        # Specify the category order for the LabelEncoder
        self.label_order = ['ANG', 'FEA', 'NEU', 'DIS', 'HAP', 'SAD']
        self.label_encoder.fit(self.label_order)

    def load_data(self,data_path):
        # Load the dataset from the .csv file
        data = pd.read_csv(data_path)

        all_data = []
        labels = []

        all_data.append(data.iloc[:, 1:])
        labels.extend(data.iloc[:, 0])

        # 合并数据和标签
        X = pd.concat(all_data).values
        y = np.array(labels)

        # Encode the labels using the fitted LabelEncoder
        y_encoded = self.label_encoder.transform(y)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def preprocess_data(self, X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def train(self, csv_file):
        X_train, X_test, y_train, y_test = self.load_data(csv_file)
        X_train, X_test = self.preprocess_data(X_train, X_test)
        self.knn.fit(X_train, y_train)
        y_pred = self.knn.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        return accuracy

    def predict(self, feature_vector):
        return self.knn.predict(feature_vector)

path = 'data1/features1.csv'
knn = EmotionClassifierKNN()
accuracy = knn.train(path)


