import os
from Extract import SpeechFeatureExtractor
import pandas as pd

#第一个数据集文件路径与标签
'''
folder_path = ['/Users/hujiawei/Desktop/Final_Project/OriginalDataset/sad','/Users/hujiawei/Desktop/Final_Project/OriginalDataset/neutral','/Users/hujiawei/Desktop/Final_Project/OriginalDataset/happy'
               ,'/Users/hujiawei/Desktop/Final_Project/OriginalDataset/fear','/Users/hujiawei/Desktop/Final_Project/OriginalDataset/disgust','/Users/hujiawei/Desktop/Final_Project/OriginalDataset/anger']
label = ['SAD', 'NEU', 'HAP', 'FEA', 'DIS', 'ANG']
'''

#第二个数据集文件路径与标签
destination_folder = 'data1'
folder_path = '/Users/hujiawei/Downloads/AudioWAV'
label = ['sad', 'neutral', 'happy', 'fear', 'disgust', 'anger']


destination_folder = 'data1'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

#提取第一个数据集特征值，每个标签存一个csv文件
'''
for i in range(len(folder_path)):
    all_features = []
    for filename in os.listdir(folder_path[i]):

        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path[i], filename)
            # 提取特征
            extractor = SpeechFeatureExtractor(file_path)
            features = extractor.features
            features = [str(feature) for feature in features]
            features = [label[i]] + features
            # 将特征添加到列表中
            all_features.append(features)

    # 将特征列表转换为 DataFrame
    all_features_df = pd.DataFrame(all_features)
    output_file = f'data/{label[i]}_features.csv'
    # 保存 DataFrame 为 CSV 文件
    all_features_df.to_csv(output_file, index=False)
'''

#提取audioWav文件夹中音频特征值，提取第二个数据集特征值，将所有特征存一个csv
all_features = []
for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        file_path = os.path.join(folder_path, filename)
        # 提取特征
        label = filename.split('_')[2]
        extractor = SpeechFeatureExtractor(file_path)
        features = extractor.features
        features = [str(feature) for feature in features]
        features = [label] + features
        # 将特征添加到列表中
        all_features.append(features)

# 将特征列表转换为 DataFrame
all_features_df = pd.DataFrame(all_features)
output_file = 'data1/features1.csv'
# 保存 DataFrame 为 CSV 文件
all_features_df.to_csv(output_file, index=False)

