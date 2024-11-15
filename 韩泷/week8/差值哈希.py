import cv2
import numpy as np

#均值Hash算法
def Chazhi_hash(img):
    hash_str=''
    # 将图像缩放到8*8
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    # 转化为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将当前和后一个做对比，每行最后一个不对比
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    # 返回hash值
    return hash_str

# 读取图片
img = cv2.imread('lenna.png')
print(Chazhi_hash(img))