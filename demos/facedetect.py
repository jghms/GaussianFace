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

aligner = GFAlign(None)
rects = aligner.detectAll(rgbImg)

(x, y, w, h) = aligner.rect2BoundingBox(rects[0])

cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imwrite('out.png', image)
