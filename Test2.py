import os
import numpy as np
from Extract import SpeechFeatureExtractor
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 加载模型
model = load_model('emotion_classifier.h5')

# 创建标签编码器
label_encoder = LabelEncoder()
emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] # 使用适当的情感标签替换此列表
label_encoder.fit(emotion_labels)

label_mapping = {
    'anger': 'ANG',
    'disgust': 'DIS',
    'fear': 'FEA',
    'happy': 'HAP',
    'neutral': 'NEU',
    'sad': 'SAD',
    # 如果测试集中没有 'surprise' 标签，将其映射到一个存在的标签，例如 'neutral'
    'surprise': 'NEU',
}
'''
label_encoder = LabelEncoder()
emotion_labels = ['ANG', 'DiS', 'FEA', 'HAP', 'NEU', 'SAD']  # 按照训练集的顺序列出所有的情绪标签
label_encoder.fit(emotion_labels)
'''

# 测试集文件夹路径
test_folder = '/Users/hujiawei/Downloads/AudioWAV'

# 获取测试集中的所有文件名
file_names = os.listdir(test_folder)

# 初始化一个空列表，用于存储预测结果和真实标签
predictions = []
true_labels = []

# 遍历测试集中的所有文件
for file_name in file_names:
    # 提取情绪标签

    file_path = os.path.join(test_folder, file_name)
    print(file_path)
    true_label = file_name.split('_')[2]
    true_labels.append(true_label)
    #print(true_label)

    speech_extractor = SpeechFeatureExtractor(audio_file=file_path)
    features = speech_extractor.features
    features = np.expand_dims(features, axis=-1)

    # 使用模型进行预测
    prediction = model.predict(np.array([features]))
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]
    predictions.append(label_mapping[predicted_label])


# 计算混淆矩阵
cm = confusion_matrix(true_labels, predictions)

# 为了在矩阵图中显示类别标签，我们需要对类别进行编码
le = LabelEncoder()
le.fit(np.unique(true_labels))
labels = le.classes_

# 绘制混淆矩阵图
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
'''
# 计算准确率
accuracy = np.sum(np.array(predictions) == np.array(true_labels)) / len(true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
'''
