import cv2
import numpy as np
import src.gaussianface as gf
import matplotlib.pyplot as plt

img = np.load('./tmp.npy')
L = np.load('./labels.npy')

image = img[0,:,:]
image = image.astype(np.uint8)

k = 3
i = 2
R = 2
P = 8
I = gf.createIndex()
J = k*k
while i < J:
    patch = gf.extractPatches(image, k, i)
    f = gf.F(patch,  R, P, I)
