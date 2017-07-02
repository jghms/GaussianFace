"""
This is a demo to test the output of LDA and PCA using a small dataset
Requires 'iris.data'

Run:

$ cd GaussianFace/
$ python -m demos.testLDAPCA

"""
import numpy as np
from numpy import linalg as la

from PIL import Image
import sys # TODO remove
import os.path # TODO remove

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.gaussianface import *


F1 = np.ones((50, 4)) # JxR
F2 = np.ones((50, 4)) # JxR
F3 = np.ones((50, 4)) # JxR

ii = 0;
with open('iris.data') as fp:
     for line in fp:
        l = line.split(',')
        if (ii < 50):
            F1[ii][0] = float(l[0])
            F1[ii][1] = float(l[1])
            F1[ii][2] = float(l[2])
            F1[ii][3] = float(l[3])
        elif (ii < 100):
            F2[ii - 50][0] = float(l[0])
            F2[ii - 50][1] = float(l[1])
            F2[ii - 50][2] = float(l[2])
            F2[ii - 50][3] = float(l[3])
        else:
            F3[ii - 100][0] = float(l[0])
            F3[ii - 100][1] = float(l[1])
            F3[ii - 100][2] = float(l[2])
            F3[ii - 100][3] = float(l[3])

        ii += 1; 

dataSize = 100
features = 4
F = np.vstack((F1, F2))
F = np.vstack((F, F3))

labels = np.hstack((np.ones(50) , np.ones(50) * 2,)) 
labels = np.hstack((labels, np.ones(50)*3))

w = LDA(F, labels)

sys.exit(0)

import matplotlib.pyplot as plt
idx = np.fromfunction(lambda i, j: j, (1, dataSize/2), dtype=int)

D1 = F1.dot(w)
D2 = F2.dot(w)
D3 = F3.dot(w)

plt.plot(D1[:, 0], D1[:, 1], 'ro')
plt.plot(D2[:, 0], D2[:, 1], 'bo')
plt.plot(D3[:, 0], D3[:, 1], 'go')
plt.show()

w = PCA(F)

D1 = F1.dot(w)
D2 = F2.dot(w)
D3 = F3.dot(w)

plt.plot(D1[:, 0], D1[:, 1], 'ro')
plt.plot(D2[:, 0], D2[:, 1], 'bo')
plt.plot(D3[:, 0], D3[:, 1], 'go')
plt.show()

w2 = LDA(F.dot(w), labels)

plt.plot(idx[0], F1.dot(w).dot(w2), 'ro')
plt.plot(idx[0]+dataSize/2, F2.dot(w).dot(w2), 'bo')
plt.plot(idx[0]+dataSize, F3.dot(w).dot(w2), 'go')
plt.show()