import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout

import matplotlib.pyplot as plt

class EmotionClassifierRNN:
    def __init__(self, data_file):
        self.data_file = data_file
        self.model = None

    def load_data(self):
        data = pd.read_csv(self.data_file)

        # 计算每个类别的样本数量
        class_counts = data.iloc[:, -1].value_counts()

        # 筛选出样本数量大于等于2的类别
        valid_classes = class_counts[class_counts >= 2].index
        data = data[data.iloc[:, -1].isin(valid_classes)]

        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        return X_train, X_test, y_train, y_test, label_encoder

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

    def build_model(self, input_shape, num_classes):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, epochs=100, batch_size=32):
        X_train, X_test, y_train, y_test, _ = self.load_data()
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
        return history

    def save_model(self, model_file):
        self.model.save(model_file)

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
history = emotion_classifier_rnn.train(epochs=150, batch_size=32)
plot_training_history(history)
emotion_classifier_rnn.save_model('emotion_classifier_rnn.h5')
