from CNN import EmotionClassifierCNN
import matplotlib.pyplot as plt


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

data_folder = 'data1'
classifier = EmotionClassifierCNN(data_folder)

X_train, _, _, _, label_encoder = classifier.load_data()
input_shape = (X_train.shape[1], 1)
num_classes = len(label_encoder.classes_)

classifier.build_model(input_shape, num_classes)
history = classifier.train(epochs=3000, batch_size=32)
plot_training_history(history)
classifier.savemodel('emotion_classifier3.h5')
