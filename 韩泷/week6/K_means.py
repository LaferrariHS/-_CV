import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('lenna.png')

rows, cols,ch = img.shape
data = img.reshape((rows*cols,3))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #停止条件

K = 4 # number of clusters

flags = cv2.KMEANS_RANDOM_CENTERS #随机初始化中心点

compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags) #K-means聚类

centers2 = np.uint8(centers)
res = centers2[labels.flatten()]
print(res)

dst = res.reshape((img.shape))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
images = [img, dst]

for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray') # 使用gray进行灰度区分
    plt.title(['Original', 'K-means'][i])
    plt.xticks([]), plt.yticks([])

plt.show()


