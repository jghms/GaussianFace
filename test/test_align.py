import unittest
import dlib
import cv2
from src.align import GFAlign

class GFALignTest(unittest.TestCase):

    def setUp(self):
        image = cv2.imread('test/faceimage.jpg')
        self.rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def test_constructorNoShapePrediction(self):
        aligner = GFAlign(None)

    def test_detectAllWithFace(self):
        aligner = GFAlign(None)
        rects = aligner.detectAll(self.rgbImg)

        self.assertEqual(len(rects), 1)
        self.assertEqual(rects[0].left(), 45)
        self.assertEqual(rects[0].right(), 107)
        self.assertEqual(rects[0].top(), 52)
        self.assertEqual(rects[0].bottom(), 114)

    def test_rect2bb(self):
        aligner = GFAlign(None)
        rect = dlib.rectangle(left=2, right=10, top=4, bottom=14)
        (x, y, w, h) = aligner.rect2BoundingBox(rect)

        self.assertEqual(x, 2)
        self.assertEqual(y, 4)
        self.assertEqual(w, 8)
        self.assertEqual(h, 10)

    def test_landmarkPredict(self):
        aligner = GFAlign('res/shape_predictor_68_face_landmarks.dat')
        rect = dlib.rectangle(left=45, right=107, top=53, bottom=115)
        shape = aligner.landmarkPrediction(self.rgbImg, rect)

        self.assertEqual(shape.shape, (68, 2))
