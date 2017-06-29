import cv2
import numpy as np
from src.gaussianface import *
import sys

filelist = createFileList("fa")
filelist2 = createFileList("fb")
filelist3 = createFileList("dup1")
filelist4 = createFileList("dup2")
trainList = createFileList("training")

images = []
fa = []
fb = []
dup1 = []
dup2 = []
testImages = []

R = 10 # 10
P = 8 # 8
PCAAccuracy = 0.98 # 0.98
k = 11 # J 3 - 16
J = k*k 

for file in filelist:
    img = cv2.imread(file)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(grayImg)
    fa.append(grayImg)
for file in filelist2:
    img = cv2.imread(file)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(grayImg)
    fb.append(grayImg)
for file in filelist3:
    img = cv2.imread(file)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(grayImg)
    dup1.append(grayImg)
for file in filelist4:
    img = cv2.imread(file)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(grayImg)
    dup2.append(grayImg)

I = createIndex()

# TODO
labels = []
faLabels = []
for img in filelist:
    faLabels.append(img[20:25])

testLabelsFB = []
for img in filelist2:
    testLabelsFB.append(img[20:25])

testLabelsDup1 = []
for img in filelist3:
    testLabelsDup1.append(img[20:25])

testLabelsDup2 = []
for img in filelist4:
    testLabelsDup2.append(img[20:25])

# Old code for generating F
# 
#for j in range(0, J):
#    Wj = WJlda(fa, labels, j, R, P, k, I, 'testingFA')

# 
#for j in range(0, J): 
#   Wj = WJlda(fb, labels, j, R, P, k, I, 'testingFB')

#
#for j in range(0, J):
#   Wj = WJlda(dup1, labels, j, R, P, k, I, 'testingDup1')

#sys.exit(0)

#for j in range(0, J):
#   Wj = WJlda(dup2, labels, j, R, P, k, I, 'testingDup2')

wLDA = []
for j in range(0, J):
    # Load Wlda j
    wLDA.append(np.load('savedMatrix/Wj' + str(j) + '.npy'))
    print wLDA[j].shape


Dj = []
for j in range(0, J):
    fj = np.array(FJ(fa, j, R, P, k, I, 'testingFA'))
    Dj.append(DJ(wLDA[j], fj))

print("Starting testing")

totaltotal = 0
totalerror = 0

errors = []
totals = []

#setsToTest = [(fb, 'testingFB', testLabelsFB), (dup1, 'testingDup1', testLabelsDup1), (dup2, 'testingDup2', testLabelsDup2), (fa, 'testingFA', faLabels)]
setsToTest = [(dup1, 'testingDup1', testLabelsDup1)]

for testset in setsToTest:
    #if totalerror < 3: totalerror += 1; continue;
    testDj = []
    for j in range(0, J):
        fj = np.array(FJ(np.array(testset[0]), j, R, P, k, I, testset[1]))
        testDj.append(DJ(wLDA[j], fj)) 

    total = 0
    error = 0
    for idx, image in enumerate(testset[0]):
        simStorage = []
        total += 1
        for img in range(0, len(faLabels)):
            sim = 0
            for i in range(0, J-1):
                if not (Dj[i] == 0).all():
                    sim += np.dot(testDj[i][idx], np.array([Dj[i][img]]).T)  / (la.norm(testDj[i][idx]) *la.norm(Dj[i][img]))
            simStorage.append(tuple((sim, faLabels[img])))
            
        # Find best SIM and compare Labels
        sorted_sim = sorted(simStorage, key=lambda tup: tup[0],  reverse=True)
        errorString = ''
        if not str(testset[2][idx]) == str(sorted_sim[0][1]):
            errorString = "\t Not equal"
            error += 1
        print("Test Label: " + str(testset[2][idx]) + " Best label: " + str(sorted_sim[0][1])) + errorString + " " + testset[1]

    totaltotal += total
    totalerror += error
    print error 
    print total
    errors.append(error)
    totals.append(total)


print totalerror
print totaltotal


print errors
print totals


# .90 .98 k=3
#[156, 722, 234, 0]
#[1195, 722, 234, 1196]


# .92 1 - 1E-6 k=3
#[150, 721, 234, 0]
#[1195, 722, 234, 1196]