import cv2
import numpy as np
from src.gaussianface import *

R = 10 # 10
P = 8 # 8
k = 3 # k 3 - 16
J = k*k 

# Collect images and Labels for all iamge sets
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

# Create MLPB index
I = createIndex()

# Generate F matrixes if not already created
for j in range(0, J):
    FJ(fa, j, R, P, k, I, 'testingFAk' + str(k))
for j in range(0, J): 
    FJ(fb, j, R, P, k, I, 'testingFBk' + str(k))
for j in range(0, J):
   FJ(dup1, j, R, P, k, I, 'testingDup1k' + str(k))
for j in range(0, J):
   FJ(dup2, j, R, P, k, I, 'testingDup2k' + str(k))

# Load the LDA space transformation Matrices
wLDA = []
for j in range(0, J):
    # Load Wlda j
    wLDA.append(np.load('savedMatrix/Wj' + str(j) + "k" + str(k) + '.npy'))
    print "Wlda " + str(j) + str(wLDA[j].shape)

# Calcualte the Dj matrices for set "fa"
Dj = []
for j in range(0, J):
    fj = np.array(FJ(fa, j, R, P, k, I, 'testingFAk' + str(k)))
    Dj.append(DJ(wLDA[j], fj))

print("Starting testing")

totaltotal = 0
totalerror = 0

errors = []
totals = []

# Choose which sets to test on
setsToTest = [(fb, 'testingFB', testLabelsFB), (dup1, 'testingDup1', testLabelsDup1), (dup2, 'testingDup2', testLabelsDup2)]
#setsToTest = [(fb, 'testingFB', testLabelsFB)]
#setsToTest = [(dup1, 'testingDup1', testLabelsDup1)]
#setsToTest = [(dup2, 'testingDup2', testLabelsDup2)]

for testset in setsToTest:
    # Create Dj matrices for testset
    testDj = []
    for j in range(0, J):
        fj = np.array(FJ(np.array(testset[0]), j, R, P, k, I, testset[1] + "k"+str(k)))
        testDj.append(DJ(wLDA[j], fj)) 

    total = 0
    error = 0
    for idx, image in enumerate(testset[0]):
        # Calculate similariy and find the best match to testimage from set "fa"
        simStorage = []
        total += 1
        for img in range(0, len(faLabels)):
            sim = 0
            for i in range(0, J):
                if not (Dj[i] == 0).all():
                    sim += np.dot(testDj[i][idx], np.array([Dj[i][img]]).T)  / (la.norm(testDj[i][idx]) * la.norm(Dj[i][img]))
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

print "Set: " + str([x for _,x,_ in setsToTest])
print "Error: " + str(errors)
print "Total: " + str(totals)