"""
This is a code to train the LDA transformation matrix on the normalised FERET images.
Requires normalized feret images

Run:

$ cd GaussianFace/
$ python -m demos.trainWLDA

"""
import cv2
import numpy as np
from src.gaussianface import *

R = 10 # 10
P = 8 # 8
k = 11 # k 3 - 16
J = k*k 

pcaAcc = 0.92

# Collect the names of images in training set
filelist = createFileList("training", False)

# Create lists of labels and images
images = []
labels = []
for img in filelist:
	labels.append(img[20:25])
for file in filelist:
	img = cv2.imread(file)
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	images.append(grayImg)
images = np.array(images)

# Create MLPB index
I = createIndex()

# Create of load the Fj matrices
# Train LDA space transformation Matrix and save.
for j in range(0, J):
	Wj = WJlda(images, labels, j, R, P, k, I, pcaAcc)
	np.save('savedMatrix/Wj' + str(j) + "k" + str(k), Wj)