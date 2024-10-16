# Canny 算法
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# 导入图像
graygraph = cv2.imread('lenna.png',0)
# 灰度化

# 高斯滤波去噪
sigma = 0.5
kernel_size = int(np.round(6*sigma)+1)
if kernel_size % 2 == 0:
    kernel_size += 1
blur_gray = cv2.GaussianBlur(graygraph, (kernel_size, kernel_size), sigma)

plt.figure(1)
plt.imshow(blur_gray.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
plt.axis('off')

# 计算梯度
gx = cv2.Sobel(blur_gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(blur_gray, cv2.CV_32F, 0, 1, ksize=3)
gx[gx == 0] = 0.00000001
# 计算梯度幅值和方向
angle = gy/gx
angle = (angle - np.min(angle)) / (np.max(angle) - np.min(angle))
dx,dy = graygraph.shape
img_tidu = np.zeros((dx,dy))
for i in range(dx):
    for j in range(dy):
        img_tidu[i, j] = np.sqrt(gx[i, j]**2 + gy[i, j]**2)

plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
plt.axis('off')

# 非极大值抑制
# 使用梯度幅值和方向确定边缘线性插值
img_yizhi = np.zeros(img_tidu.shape)
for i in range(1, dx-1):
    for j in range(1, dy-1):
        flag = True  # 在8邻域内是否要抹去做个标记
        temp = img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
        if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        if flag:
            img_yizhi[i, j] = img_tidu[i, j]
plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

# 4、双阈值检测
lower_boundary = img_tidu.mean() * 0.5
high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
zhan = []
for i in range(1, img_yizhi.shape[0]-1):  
    for j in range(1, img_yizhi.shape[1]-1):
        if img_yizhi[i, j] >= high_boundary: 
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        elif img_yizhi[i, j] <= lower_boundary: 
            img_yizhi[i, j] = 0

while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop() 
    a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1-1, temp_2-1] = 255 
        zhan.append([temp_1-1, temp_2-1])
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        img_yizhi[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        img_yizhi[temp_1, temp_2 + 1] = 255
        zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])

for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0

# 绘图
plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')  # 关闭坐标刻度值


# 简化方法
cannyimg = cv2.Canny(blur_gray, 60,180,apertureSize=3)
plt.figure(5)
plt.imshow(cannyimg, cmap='gray')
plt.axis('off')  # 关闭坐标刻度值
plt.show()



