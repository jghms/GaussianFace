import cv2
import sys
import numpy as np
from src.align import GFAlign
from src.gaussianface import mLBP, createIndex

imagePath = sys.argv[1]

image = cv2.imread(imagePath, 0)
# rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

aligner = GFAlign(None)
rects = aligner.detectAll(image)

(x, y, w, h) = aligner.rect2BoundingBox(rects[0])

thumbnail = image[y:y+h,x:x+w]
thumbnail = cv2.resize(thumbnail, (142, 120))

cv2.imwrite('out.png', image)
cv2.imwrite('thumbnail.png', thumbnail)

R = 2
P = 8
print thumbnail.shape
lbpimage = np.empty_like(thumbnail)
index = createIndex()
for x in range(thumbnail.shape[0]):
    for y in range(thumbnail.shape[1]):
        mlbp = mLBP(thumbnail, R, P, x, y, index)
        lbpimage[x, y] =  mlbp

print lbpimage
cv2.imshow('image', lbpimage*4)
cv2.waitKey(0)
cv2.destroyAllWindows()
