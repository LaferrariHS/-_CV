import cv2
import numpy as np

#均值Hash算法
def mean_hash(img):
    hash_str=''
    # 将图像缩放到8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    # 转化为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算图像的均值
    mean = np.mean(gray)
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > mean:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    # 返回hash值
    return hash_str

# 读取图片
img = cv2.imread('lenna.png')
print(mean_hash(img))
