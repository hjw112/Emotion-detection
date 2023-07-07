import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop, Adagrad

class EmotionClassifierCNN:

    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.model = None

    def load_data(self):
        all_data = []
        labels = []

        # 读取所有 CSV 文件
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.data_folder, filename)
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

        return X_train, X_test, y_train, y_test, label_encoder


    def build_model(self, input_shape, num_classes):
        self.model = Sequential()
        #激活函数选择relu
        self.model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(128, kernel_size=3, activation='relu'))

        #在卷积层和全连接层之间添加归一化层，加快收敛速度
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        #添加随机dropout
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))

        #调整其它优化器如optimizer = RMSprop(lr=0.001)或者adagrad, 可以有更快的收敛速度, 但是收敛后准确率较低
        #另外，更小的学习率可以提高模型收敛速度
        optimizer = Adam(learning_rate=0.0001)
        #optimizer = RMSprop(learning_rate=0.0001)
        #optimizer = Adagrad(learning_rate=0.0001)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=1000, batch_size=32):
        if self.model is None:
            raise ValueError("Model is not built yet. Call 'build_model' first.")

        X_train, X_test, y_train, y_test, label_encoder = self.load_data()

        #将数据调整为卷积层所需的形状
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)


        #将标签转化为分类编码
        y_train_categorical = to_categorical(y_train)
        y_test_categorical = to_categorical(y_test)

        history = self.model.fit(X_train, y_train_categorical, validation_data=(X_test, y_test_categorical),
                       epochs=epochs, batch_size=batch_size)

        return history

    def savemodel(self, model_path):
        if self.model is None:
            raise ValueError("the model is not built yet. Please build the model first, call the 'build model' function first")
        self.model.save(model_path)
