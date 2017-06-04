#!/bin/bash
# GET haarcascade
curl -O https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml

# GET face landmarks
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
tar xvjf shape_predictor_68_face_landmarks.dat.bz2
