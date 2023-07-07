import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model = load_model('emotion_classifier3.h5')

label_encoder = LabelEncoder()
emotion_labels = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'] # 使用适当的情感标签替换此列表

label_encoder.fit(emotion_labels)


csv_files = [
    'data/ANG_features.csv',
    'data/DIS_features.csv',
    'data/FEA_features.csv',
    'data/HAP_features.csv',
    'data/NEU_features.csv',
    'data/SAD_features.csv'
]

dataframes = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dataframes.append(df)

combined_df = pd.concat(dataframes)

X_test = combined_df.iloc[:, 1:].values  # 选取除第一列（索引为 0）之外的所有列作为特征
y_test = label_encoder.transform(combined_df.iloc[:, 0].values)  # 选取第一列（索引为 0）作为标签

X_test = np.expand_dims(X_test, axis=-1)

confidence_matrix = model.predict(X_test)
y_pred = np.argmax(confidence_matrix, axis=1)
confusion_mat = confusion_matrix(y_test, y_pred)

################
num_classes = len(emotion_labels)

# 取置信度矩阵的平均值
mean_confidence = np.mean(confidence_matrix, axis=0)

fig, ax = plt.subplots(figsize=(8, 6))
bar_positions = np.arange(num_classes)
bars = ax.bar(bar_positions, mean_confidence, align='center', alpha=0.7)

ax.set_xticks(bar_positions)
ax.set_xticklabels(emotion_labels)
ax.set_xlabel('Emotion')
ax.set_ylabel('Confidence')
ax.set_title('Confidence per Emotion')
plt.show()

#######################
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.title('Confusion Matrix')
plt.show()
