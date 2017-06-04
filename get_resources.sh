#!/bin/bash

cd res/

# GET haarcascade
curl -O https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml

# GET face landmarks
curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2


cd ../test
curl -o faceimage.jpg http://vis-www.cs.umass.edu/lfw/images/Aaron_Eckhart/Aaron_Eckhart_0001.jpg
