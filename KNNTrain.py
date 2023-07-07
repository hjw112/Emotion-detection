import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from KNN import EmotionClassifierKNN

def plot_knn_accuracy(emotion_classifier_knn, max_neighbors=10):
    _, X_test, _, y_test, _ = emotion_classifier_knn.load_data()

    accuracies = []
    k_values = list(range(1, max_neighbors + 1))

    for k in k_values:
        emotion_classifier_knn.build_and_train_model(n_neighbors=k)
        y_pred = emotion_classifier_knn.knn_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    plt.plot(k_values, accuracies)
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Test Set Accuracy')
    plt.title('KNN Test Set Accuracy for Different K Values')
    plt.show()

data_path = 'data1/features.csv'
emotion_classifier_knn = EmotionClassifierKNN(data_path)
plot_knn_accuracy(emotion_classifier_knn, max_neighbors=100)
