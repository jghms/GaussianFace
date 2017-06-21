"""
Detects a face in an image and saves it to a file.

Run:

$ cd GaussianFace/
$ python -m demos.facedetect <image_path>

"""

import cv2
import sys
import numpy as np
from src.align import GFAlign

imagePath = sys.argv[1]

image = cv2.imread(imagePath)
rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

aligner = GFAlign('res/shape_predictor_68_face_landmarks.dat')
rects = aligner.detectAll(rgbImg)

(x, y, w, h) = aligner.rect2BoundingBox(rects[0])

grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thumbnail = grayImg[y:y+h,x:x+w]
thumbnail = cv2.resize(thumbnail, (142, 120))

cv2.imwrite('out.png', image)
cv2.imwrite('thumbnail.png', thumbnail)
