import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
sns.set()  # for plot styling
from sklearn.cluster import KMeans

from sklearn.datasets.samples_generator import make_blobs
from config import cfg, annotation_dir, repo_dir
from yolov3.data import convert_annotation


def test_of_k_means():
    # 创建测试点，X是数据，y是标签，X:(300,2), y:(300,)
    os.chdir(annotation_dir)
    image_ids = os.listdir('.')
    image_ids = glob.glob(str(image_ids) + '*.xml')
    os.chdir(repo_dir)
    coord_arr = []
    for train_instance in image_ids[:]:
        image_id = train_instance.split('.xml')[0]
        xywhc = convert_annotation(annotation_dir, image_id, None)
        if not xywhc:
            continue
        coord = np.reshape(xywhc, [30, 5])
        # print(image_id, 'coord', coord)
        for row in range(np.size(coord,0)):
            if np.sum(coord[row, :]) > 0:
                coord_arr.append(coord[row, 2:4])
        X = np.asarray(coord_arr)
    # X, y_true = make_blobs(n_samples=300, centers=9, cluster_std=0.60, random_state=0)
    kmeans = KMeans(n_clusters=9)  # 将数据聚类
    kmeans.fit(X)  # 数据X
    y_kmeans = kmeans.predict(X)  # 预测

    # 颜色范围viridis: https://matplotlib.org/examples/color/colormaps_reference.html
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='viridis')  # c是颜色，s是大小

    centers = kmeans.cluster_centers_  # 聚类的中心
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=40, alpha=0.5)  # 中心点为黑色
    arr = sorted(centers, key=lambda x: x[0]*x[1])
    print(arr)
    print("Accuracy:", kmeans.inertia_/(22000/9)*416)

    plt.show()  # 展示
    for num in range(len(arr)):
        for num_sub in range(len(arr[num])):
            arr[num][num_sub] *= 416
            arr[num][num_sub] = int(arr[num][num_sub])
# [[0.02076107, 0.01293223], [0.0617669 , 0.02770938], [0.04374917, 0.13792588], [0.29036524, 0.11192373], [0.08295782, 0.40606537], [0.71478164, 0.06639978], [0.08983734, 0.80475301], [0.69282471, 0.41136212], [0.72597913, 0.81407759]]
# [[0.02037891, 0.01285352], [0.06018678, 0.02693154], [0.04385148, 0.13835239], [0.28709339, 0.11099135], [0.08399807, 0.41270674], [0.71451175, 0.06639165], [0.08975563, 0.80695904], [0.69282471, 0.41136212], [0.72597913, 0.81407759]]


if __name__ == '__main__':
    test_of_k_means()