import cv2
import sys
from src.gaussianface import extractPatches

imagePath = sys.argv[1]
image = cv2.imread(imagePath)
if image == None:
    print "No image found"
patches = extractPatches(image)

for i, patch in enumerate(patches):
    cv2.imwrite('patches/patch{}.png'.format(i), patch)
