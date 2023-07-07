import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import mahalanobis

class Discrimination:

    def __init__(self, date1, date2):
        self.date1 = date1
        self.date2 = date2

    def Euclidean_Distance(self):
        dataset1_center = np.mean(self.date1, axis=0)
        dataset2_center = np.mean(self.date2, axis=0)

        # 每个数据集中的样本与中心的欧几里得距离
        dataset1_distances = cdist(self.date1, [dataset1_center], 'euclidean')
        dataset2_distances = cdist(self.date2, [dataset2_center], 'euclidean')

        # 区分度
        discriminability = np.abs(np.mean(dataset1_distances) - np.mean(dataset2_distances))

        print("discriminability：", discriminability)
        return discriminability

    def mahalanobis(self):
        dataset1_center = np.mean(self.date1, axis=0)
        dataset2_center = np.mean(self.date2, axis=0)

        covariance1 = np.cov(self.date1, rowvar=False)
        covariance2 = np.cov(self.date2, rowvar=False)

        print(np.linalg.det(covariance1))
        print(covariance1)


        md1 = [mahalanobis(x, dataset1_center, np.linalg.inv(covariance1)) for x in self.date1]
        md2 = [mahalanobis(x, dataset2_center, np.linalg.inv(covariance2)) for x in self.date2]

        # 区分度
        separation = np.abs(np.mean(md1) - np.mean(md2)) / np.sqrt(np.var(md1) + np.var(md2))
        return(separation)

