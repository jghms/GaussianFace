import cv2
import numpy as np

img = np.load('./tmp.npy')
L = np.load('./labels.npy')

image = img[0,:,:]
image = image.astype(np.uint8)
cv2.imshow("Image 0", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
