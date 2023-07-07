import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam

class EmotionClassifierRNN:
    def __init__(self, data_file):
        self.data_file = data_file
        self.label_encoder = LabelEncoder()
        self.model = None

    def load_data(self):
        all_data = []
        labels = []

        data = pd.read_csv(self.data_file)
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

        return X_train, X_test, y_train, y_test, label_encoder
    '''
    def build_model(self, input_shape, num_classes):
        self.model = Sequential()
        self.model.add(GRU(128, input_shape=input_shape))  # Add GRU layer with 128 hidden units
        self.model.add(Dropout(0.5))  # Add Dropout layer with 0.5 dropout rate to prevent overfitting
        self.model.add(Dense(num_classes, activation='softmax'))  # Add output Dense layer with softmax activation for multi-class classification
        optimizer = Adam(learning_rate=0.001)  # Define the optimizer with a custom learning rate
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # Compile the model with the custom optimizer
    '''

    def build_model(self, input_shape, num_classes):
        self.model = Sequential()
        self.model.add(GRU(128, input_shape=input_shape, return_sequences=True))  # Add first GRU layer with 128 hidden units and return full sequence
        self.model.add(Dropout(0.5))  # Add Dropout layer with 0.5 dropout rate to prevent overfitting
        self.model.add(GRU(64, return_sequences=False))  # Add second GRU layer with 64 hidden units and return only the last output
        self.model.add(Dropout(0.5))  # Add Dropout layer with 0.5 dropout rate to prevent overfitting
        self.model.add(Dense(num_classes, activation='softmax'))  # Add output Dense layer with softmax activation for multi-class classification
        optimizer = Adam(learning_rate=0.0005)  # Define the optimizer with a custom learning rate
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # Compile the model with the custom optimizer

    def train(self, epochs=100, batch_size=32):
        X_train, X_test, y_train, y_test, _ = self.load_data()  # Load the data
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)  # Train the model
        return history

    def save_model(self, model_file):
        self.model.save(model_file)  # Save the model to a file



def plot_training_history(history):
    # 获取训练过程中的accuracy和loss值
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(accuracy) + 1)

    # 绘制accuracy曲线
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # 绘制loss曲线
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


data_file = 'data1/features.csv'
emotion_classifier_rnn = EmotionClassifierRNN(data_file)

X_train, _, _, _, label_encoder = emotion_classifier_rnn.load_data()

input_shape = (X_train.shape[1], 1)
num_classes = len(label_encoder.classes_)

emotion_classifier_rnn.build_model(input_shape, num_classes)
history = emotion_classifier_rnn.train(epochs=500, batch_size=32)
plot_training_history(history)
emotion_classifier_rnn.save_model('emotion_classifier_rnn_upgrade.h5')
