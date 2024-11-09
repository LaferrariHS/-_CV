import numpy as np
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram, linkage , fcluster

# 生成示例数据
data =np.random.rand(10,2)

# 层次聚类
Z = linkage(data,'ward')

# 绘制层次聚类树
plt.figure(figsize=(10, 7))
dendrogram(Z)
f = fcluster(Z,4,'distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

clusters = fcluster(Z, t=2, criterion='maxclust')  # 选择2个簇
print(clusters)
print(Z)
