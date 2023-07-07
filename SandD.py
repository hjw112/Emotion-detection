#import Similarity
import Discrimination
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset1 = np.genfromtxt('/Users/hujiawei/Desktop/Final_Project/data/anger_feature.csv', delimiter=',')
dataset2 = np.genfromtxt('/Users/hujiawei/Desktop/Final_Project/data/sad_feature.csv', delimiter=',')
dataset3 = np.genfromtxt('/Users/hujiawei/Desktop/Final_Project/data/happy_feature.csv', delimiter=',')
dataset4 = np.genfromtxt('/Users/hujiawei/Desktop/Final_Project/data/fear_feature.csv', delimiter=',')
dataset5 = np.genfromtxt('/Users/hujiawei/Desktop/Final_Project/data/neutral_feature.csv', delimiter=',')
#dataset6 = np.genfromtxt('/Users/hujiawei/Desktop/Final_Project/data/surprise_feature.csv', delimiter=',')
#dataset7 = np.genfromtxt('/Users/hujiawei/Desktop/Final_Project/data/disgust_feature.csv', delimiter=',')

dataset1 = dataset1[1:,1:]
dataset2 = dataset2[1:,1:]
dataset3 = dataset3[1:,1:]
dataset4 = dataset4[1:,1:]
dataset5 = dataset5[1:,1:]
#dataset6 = dataset6[1:,1:]
#dataset7 = dataset7[1:,1:]

dataset = [dataset1,dataset2,dataset3,dataset4,dataset5]
#dataset = [dataset2,dataset3,dataset5]

data = np.zeros((5, 5))
data1 = np.zeros((5,5))

for i in range(5):
    for j in range(5):
        data[i, j] = Discrimination.Discrimination(dataset[i], dataset[j]).Euclidean_Distance()
        #data[i, j] =Discrimination.Discrimination(dataset[i],dataset[j]).mahalanobis()*10000

for i in range(5):
    for j in range(5):
        data1[i, j] =Discrimination.Discrimination(dataset[i],dataset[j]).mahalanobis()*10000



fig, ax = plt.subplots()
im = ax.imshow(data, cmap='viridis')

cbar = ax.figure.colorbar(im, ax=ax)
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))

ax.set_xticklabels(['anger', 'sad', 'fear', 'happy', 'neutral'])
ax.set_yticklabels(['anger', 'sad', 'fear', 'happy', 'neutral'])

plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        text = ax.text(j, i, round(data[i, j], 2), ha='center', va='center', color='w')

ax.set_title("Discriminability")

plt.show()
