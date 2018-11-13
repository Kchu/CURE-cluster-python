# -*- coding:utf-8 -*-

from CURE import *
import sys,time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# The number of representative points
numRepPoints = 5
# Shrink factor
alpha = 0.1
# Desired cluster number
numDesCluster = 3

start = time.clock()
data_set = np.loadtxt('E:/Kchu/Research/CURE/CURE-python/Data/3clus.txt')
data = data_set[:,0:2]
Label_true = data_set[:,2]
print("Please wait for CURE clustering to accomplete...")
Label_pre = runCURE(data, numRepPoints, alpha, numDesCluster)
print("The CURE clustering is accompleted!!\n")
end = time.clock()
print("The time of CURE algorithm is", end - start, "\n")
# Compute the NMI
nmi = metrics.v_measure_score(Label_true, Label_pre)
print("NMI =", nmi)
# Plot the result
plt.subplot(121)
plt.scatter(data_set[:, 0], data_set[:, 1], marker='.')
plt.text(0, 0, "origin")
plt.subplot(122)
scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown', 'cyan', 'brown',
                 'chocolate', 'darkgreen', 'darkblue', 'azure', 'bisque']
for i in range(data_set.shape[0]):
    color = scatterColors[Label_pre[i]]
    plt.scatter(data_set[i, 0], data_set[i, 1], marker='o', c=color)
plt.text(0, 0, "clusterResult")
plt.show()

