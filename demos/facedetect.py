"""
Detects a face in an image and saves it to a file.

Run:

$ cd GaussianFace/
$ python -m demos.facedetect <image_path>

"""

import cv2
import sys
from src.align import GFAlign

imagePath = sys.argv[1]

image = cv2.imread(imagePath)
rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

aligner = GFAlign('res/shape_predictor_68_face_landmarks.dat')
rects = aligner.detectAll(rgbImg)

(x, y, w, h) = aligner.rect2BoundingBox(rects[0])

shape = aligner.landmarkPrediction(rgbImg, rects[0])
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
shape = aligner.landmarkPrediction(rgbImg, rects[0])

for (x, y) in shape:
	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imwrite('out.png', image)
