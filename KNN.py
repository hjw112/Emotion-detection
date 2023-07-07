import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

class EmotionClassifierKNN:
    def __init__(self, data_path):
        self.data_path = data_path
        self.knn_pipeline = None
        self.label_encoder = LabelEncoder()

        # Specify the category order for the LabelEncoder
        self.label_order = ['ANG', 'FEA', 'NEU', 'DIS', 'HAP', 'SAD']
        self.label_encoder.fit(self.label_order)

    def load_data(self):
        # Load the dataset from the .csv file
        data = pd.read_csv(self.data_path)

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

        return X_train, X_test, y_train, y_test, self.label_encoder

    def build_and_train_model(self, n_neighbors=10):
        # Create a pipeline for preprocessing and classification
        self.knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Scale features to have zero mean and unit variance
            ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors))  # Use KNN with specified number of neighbors
        ])

        # Load the dataset
        X_train, X_test, y_train, y_test, _ = self.load_data()

        # Train the KNN model
        self.knn_pipeline.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = self.knn_pipeline.predict(X_test)

        # Calculate and print the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Test accuracy: {accuracy:.2f}')

    def save_model(self, file_name='emotion_classifier_knn.pkl'):
        # Save the trained model
        joblib.dump(self.knn_pipeline, file_name)

'''
data_path = 'data1/features.csv'
emotion_classifier_knn = EmotionClassifierKNN(data_path)

# Train the KNN model
emotion_classifier_knn.build_and_train_model(n_neighbors=3)

# Save the trained model
emotion_classifier_knn.save_model('emotion_classifier_knn.pkl')
'''
