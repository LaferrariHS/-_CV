import numpy as np
import cv2

def perspective_transform(src, dst):
    assert src.shape == dst.shape and src.shape[0] >= 4
    
    A = np.zeros((2*src.shape[0],8))
    B = np.zeros((2*src.shape[0],1))
    
    for i in range(src.shape[0]):
        A_i = src[i]
        B_i = dst[i]
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,-A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1,-A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]
        
    A = np.mat(A)
    warp_matrix = A.I * np.mat(B)
    warp_matrix = np.array(warp_matrix).T[0]
    warp_matrix = np.insert(warp_matrix,warp_matrix.shape[0], values=1.0, axis=0)
    warp_matrix = warp_matrix.reshape((3, 3))
    return warp_matrix
    
if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)
    
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    
    warp_matrix = perspective_transform(src, dst)
    print(warp_matrix)
    img = cv2.imread('photo1.jpg')

    result1 = img.copy()
    src = np.float32([[205, 151], [517, 285], [16, 602], [342, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    m = perspective_transform(src, dst)
    # m = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(result1, m, (337, 488))
    cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)
