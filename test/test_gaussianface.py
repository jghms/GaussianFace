import unittest
import numpy as np
import cv2
from src.gaussianface import *

class GFLBPTest(unittest.TestCase):

    def setUp(self):
        self.im = np.array([[26, 129, 232],[121, 150, 211], [50, 89, 215]])

    def test_LBP(self):

        res = LBP(self.im, 1, 8, 1, 1)

        self.assertEqual(res[0], 131)

    def test_createIndex(self):

        I = createIndex()

        self.assertEqual(I(5), 0)
        self.assertEqual(I(6), 1)
        self.assertEqual(I(5), 0)

    def test_uniformity(self):

        res, gc, gp = LBP(self.im, 1, 8, 1, 1)

        self.assertTrue(uniformity(gc, gp) <= 2)
