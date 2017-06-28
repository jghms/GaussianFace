import cv2
import numpy as np
from src.gaussianface import *


filelist = createFileList("training", False)

images = []

labels = []
for img in filelist:
	labels.append(img[20:25])

R = 10 # 10
P = 8 # 8
#PCAAccuracy = 0.98 # 0.98
k = 3 # J 3 - 11
J = k*k 

for file in filelist:
	img = cv2.imread(file)
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	images.append(grayImg)

images = np.array(images)

I = createIndex()

for j in range(0, J):
	Wj = WJlda(images, labels, j, R, P, k, I)
	np.save('savedMatrix/Wj' + str(j), Wj)