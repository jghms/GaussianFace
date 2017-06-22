import numpy as np
from numpy import linalg as la
from src.gaussianface import *

#from PIL import Image
import sys # TODO remove
import os.path # TODO remove

import cv2
from src.align import GFAlign

R = 10 # 10
P = 8 # 8
PCAAccuracy = 0.98 # 0.98
k = 3 # J 3 - 11
J = k*k 
featuresSize = 58
trainingSetSize = 495 # 501 (Some images does not exist)
resizeWidth = 120 # 120
resizeHeight = 142 # 142

def PCA(data):

    # Covariance matrix
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    cov = np.cov(data, rowvar=False)
    #mu = np.mean(data, axis=0)
    #cov = np.cov((data - mu), rowvar=False)

    # Compute eigenvectors
    eig, eigV = la.eig(cov)
    eigSize = eigV.shape
    normV = eig / np.sum(eig)
    
    # Sort the eigenvectors
    sortedV = np.argsort(normV)
    sortedV = sortedV[: :-1]

    # Find the most important eigenvectors
    important = normV[sortedV[0]]
    w = eigV[sortedV[0]];
    i = 1;
    while (important < PCAAccuracy and i < eigSize[0]) or i < 2:
        important += normV[sortedV[i]]
        w = np.vstack((w, eigV[sortedV[i]]));
        i += 1;

    print("PCA Eigenvectors " + str(i))

    return w.T

def LDA(data, labels):
    feat = data.shape[1]

    # Calculate mean

    mu = np.mean(data, axis=0)

    sw = np.zeros((feat, feat))
    sb = np.zeros((feat, feat))

    muClass = np.zeros((1, feat))
    dataClass = np.zeros((0, feat))
    label = 1
    objects = 0
    for idx in range(0, data.shape[0]):
        if not label == labels[idx]:
            label = labels[idx]
            muClass = muClass / objects
            sw += ((dataClass - muClass).T.dot((dataClass - muClass)))
            sb += objects * (muClass - mu) * (muClass - mu).reshape(feat, 1);
            dataClass = np.zeros((0, feat))
            muClass = np.zeros((1, feat))
            objects = 0
        objects += 1

        muClass += data[idx]
        dataClass = np.vstack((dataClass, data[idx]))

    eig, eigV = la.eig(la.inv(sw).dot(sb))

    # Get one Eigenvector
    #w = eigV.T[np.argmax(eig)]

    # Get more Eigenvectors
    # Sort the eigenvectors
    normV = eig / np.sum(eig)
    eigSize = eigV.shape
    sortedV = np.argsort(normV)
    sortedV = sortedV[: :-1]

    # Find the most important eigenvectors
    important = normV[sortedV[0]]
    w = eigV[sortedV[0]];
    i = 1;
    while (important < PCAAccuracy and i < eigSize[0]) or i < 2:
        important += normV[sortedV[i]]
        w = np.vstack((w, eigV[sortedV[i]]));
        i += 1;

    print "LDA Eigenvectors  " + str(i)

    return w.T

def LDAOld(dataClass1, dataClass2, F3=0):
    feat = dataClass1.shape[1]

    # Calculate mean
    mu1 = np.mean(dataClass1, axis=0)
    mu2 = np.mean(dataClass2, axis=0)
    mu = (mu1 + mu2) / 2;
    #mu3 = np.mean(F3, axis=0)
    #mu = (mu1 + mu2 + mu3) / 3;

    #print "mu"
    #print mu1
    #print mu2
    #print mu3

    # Within-class Scatter matrix
    s1 = ((dataClass1 - mu1).T.dot((dataClass1 - mu1)))
    s2 = ((dataClass2 - mu2).T.dot((dataClass2 - mu2)))
    sw = s1 + s2;
    #s3 = ((F3 - mu3).T.dot((F3 - mu3)))
    #sw = s1 + s2 + s3;

    # Small round errors so the matrix is not symmetric and gives imaginary parts
    #c1 = np.cov((dataClass1 - mu1), rowvar=False)
    #c2 = np.cov((dataClass2 - mu2), rowvar=False)
    #c3 = np.cov((F3 - mu3), rowvar=False)


    #print "sw"
    #print sw
    #print "scw"
    #print scw

    # Between class
    sb1 = dataSize/2 * (mu1 - mu) * (mu1 - mu).reshape(feat, 1);
    sb2 = dataSize/2 * (mu2 - mu) * (mu2 - mu).reshape(feat, 1);
    sb = sb1 + sb2
    #sb3 = dataSize/2 * (mu3 - mu) * (mu3 - mu).reshape(feat, 1);
    #sb = sb1 + sb2 + sb3;

    #print "sb"
    #print sb

    eig, eigV = la.eig(la.inv(sw).dot(sb))

    #print "Eig"
    #print eig
    #print ""
    #print eigV

    w = eigV.T[np.argmax(eig)]

    #print w

    return w.T

def readFeretFiles():
    partitionsPath = './colorferet/colorferet/colorferet/dvd1/doc/partitions/'
    imagePath1 = './colorferet/colorferet/colorferet/dvd1/data/images/'
    imagePath2 = './colorferet/colorferet/colorferet/dvd2/data/images/'

    dup1 = partitionsPath + 'dup1.txt' # 736
    dup2 = partitionsPath + 'dup2.txt' # 994
    fa = partitionsPath + 'fa.txt'     # 228
    fb = partitionsPath + 'fb.txt'     # 992
    # fc = open(partitionsPath + 'fc.txt', 'r')

    training = 'feret_training.srt'

    I = np.zeros((trainingSetSize, resizeWidth, resizeHeight))
    L = np.zeros((trainingSetSize))

    i = 0
    imgCount = 0
    label = 0
    with open(training, 'r') as trainingFiles:
        for line in trainingFiles:
            image = line.split()
            label += 1
            for img in image:
                part1 = img[0:5]
                part2 = img[5:7]
                part3 = img[7:10] # 010 or Something else
                part4 = img[11:17]
                part5 = ''
                if img[10] == 'a':
                    part4 = img[12:18]
                    part5 = '_' + img[10]
                elif img[10] == 'd':
                    part4 = img[12:18]
                imgFile = part1 + '/' + part1 + '_' + part4 + '_fa' + part5 + '.ppm'
                imgFile2 = part1 + '/' + part1 + '_' + part4 + '_fa' + '_a' + part5 + '.ppm'

                
                if os.path.exists(imagePath1 + imgFile):
                    imagePath = imagePath1 + imgFile
                elif os.path.exists(imagePath1 + imgFile2):
                    imagePath = imagePath1 + imgFile2
                elif os.path.exists(imagePath2 + imgFile):
                    imagePath = imagePath2 + imgFile
                elif os.path.exists(imagePath2 + imgFile2):
                    imagePath = imagePath2 + imgFile2
                else:
                    print(img)
                    print(imgFile)
                    print("File not found")
                    #raise IOError("File")
                    continue

                print("Found: " + imgFile + " Progress: " +str(round(100.0*imgCount/495, 1)) + "%")

                image = cv2.imread(imagePath)

                rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                aligner = GFAlign(None)
                rects = aligner.detectAll(rgbImg)

                (x, y, w, h) = aligner.rect2BoundingBox(rects[0])

                grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                thumbnail = grayImg[y:y+h,x:x+w]
                thumbnail = cv2.resize(thumbnail, (resizeHeight, resizeWidth))

                #print thumbnail
                I[imgCount][:][:] = thumbnail
                L[imgCount] = label
                imgCount += 1

                #if imgCount >= 10:
                #    return I, L

    print("imgCount: " + str(imgCount))
    return I, L

def readAllFeretFiles():
    imgCount = 0
    for filename in [dup1, fa, dup2, fb]: # Training
        partitionCount = 0
        print filename
        with open(filename, 'r') as fp:
            for line in fp:
                l = line.split()

                if os.path.exists(imagePath1 + l[0] +  '/' + l[1]):
                    with open(imagePath1 + l[0] +  '/' + l[1]) as img:
                        #print("open" + l[1])
                        imgCount += 1
                        partitionCount += 1
                elif os.path.exists(imagePath2 + l[0] +  '/' + l[1]):
                    with open(imagePath2 + l[0] +  '/' + l[1]) as img:
                        #print("open" + l[1])
                        imgCount += 1
                        partitionCount += 1
                else:
                    print(l)
                    raise IOError("File")

        print(partitionCount)
    print imgCount

    print img_two
    img_two = np.asarray(img_two)
    print img_two



def testData():
    F1 = np.ones((50, 4)) # JxR
    F2 = np.ones((50, 4)) # JxR
    F3 = np.ones((50, 4)) # JxR


    ii = 0;
    with open('iris.data') as fp:
         for line in fp:
            l = line.split(',')
            if (ii < 50):
                F1[ii][0] = float(l[0])
                F1[ii][1] = float(l[1])
                F1[ii][2] = float(l[2])
                F1[ii][3] = float(l[3])
            elif (ii < 100):
                F2[ii - 50][0] = float(l[0])
                F2[ii - 50][1] = float(l[1])
                F2[ii - 50][2] = float(l[2])
                F2[ii - 50][3] = float(l[3])
            else:
                F3[ii - 100][0] = float(l[0])
                F3[ii - 100][1] = float(l[1])
                F3[ii - 100][2] = float(l[2])
                F3[ii - 100][3] = float(l[3])

            ii += 1; 

    dataSize = 100
    features = 4

    w = LDA(F1, F2, F3)

    import matplotlib.pyplot as plt
    idx = np.fromfunction(lambda i, j: j, (1, dataSize/2), dtype=int)

    plt.plot(idx[0], F1.dot(w), 'ro')
    plt.plot(idx[0]+dataSize/2, F2.dot(w), 'bo')
    plt.plot(idx[0]+dataSize, F3.dot(w), 'go')
    plt.show()


    F = np.vstack((F1, F2))
    F = np.vstack((F, F3))

    w = PCA(F)

    plt.plot(idx[0], F1.dot(w)[:, 1], 'ro')
    plt.plot(idx[0]+dataSize/2, F2.dot(w)[:, 1], 'bo')
    plt.plot(idx[0]+dataSize, F3.dot(w)[:, 1], 'go')
    plt.show()

    w2 = LDA(F1.dot(w), F2.dot(w), F3.dot(w))

    plt.plot(idx[0], F1.dot(w).dot(w2), 'ro')
    plt.plot(idx[0]+dataSize/2, F2.dot(w).dot(w2), 'bo')
    plt.plot(idx[0]+dataSize, F3.dot(w).dot(w2), 'go')
    plt.show()


mode = 3

if mode == 1:

    img, L = readFeretFiles()


    np.save('savedMatrix/tmp', img)
    np.save('savedMatrix/labels', L)
    
    print L

    FJ = np.zeros((J, trainingSetSize, featuresSize)) # OBS 59!?

    for j  in range(0, J):
        images = 10

        for i in range(0, images):
            imgS = img[i][:][:]

            # Create regions
            patch = extractPatches(imgS, k, j)

            index = createIndex()

            f = F(patch, R, P, index)[:][0]

            #print "Patch Shape: " + str(patch.shape)
            #print "f shape: " + str(f.shape)
            #print "f avg: " + str(np.average(f))

            FJ[j][i][:] = f.reshape(featuresSize)
        print("FJ: " + str(j) + " done")

    R = 100 # Number of Scales
    P = [1 , 2 ,3 ,4 ,5] # 1xR Number of pixels in neighbourhood
    J = 10 # Amount of subregions

    dataSize = 10
    features = 58

    print FJ.shape
    print FJ

    np.save('savedMatrix/FJ', FJ)
elif mode == 2:
    FJ = np.load('savedMatrix/FJ.npy')
    L = np.load('savedMatrix/labels.npy')

elif mode == 3:

    for j in range(0, J-1):
        wPCA = PCA(FJ[j][:][:])
        dPCA = np.dot(FJ[j][:][:], wPCA)

        wLDA = LDA(dPCA, L)
        DJ = np.dot(dPCA, wLDA)

        #wLDA = LDA(FJ[j][:][:], L)
        #DJ = np.dot(FJ[j][:][:], wLDA)

        print "New DJ"
        #print DJ

        np.save('savedMatrix/wLDA'+str(j), DJ)
        np.save('savedMatrix/DJ'+str(j), DJ)

elif mode == 4:
    # Load DJ

    DJ = array()
    wLDA = array()

    for i in range(0, 9-1):
        DJ[i] = np.load('savedMatrix\DJ'+str(i))
        wLDA[i] = np.load('savedMatrix/wLDA'+str(j))

# Load test image/s


# Get FJ for picture to test


    # Calculate Similarity


















def oldCode():
    mu_vec1 = np.zeros(features)
    cov_mat1 = np.identity(features)
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, dataSize/2).T

    mu_vec2 = np.ones(features)*4
    cov_mat2 = np.identity(features)
    cov_mat2[2][2] = 4
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, dataSize/2).T

    all_samples = np.concatenate((class1_sample, class2_sample), axis=1)

    class1_sample = class1_sample.T
    class2_sample = class2_sample.T
    all_samples = all_samples.T

    print("All samples")
    print(all_samples)

    w = PCA(all_samples)

    print("PCA w " + str(w.shape))
    print(str(w))

    w=np.identity(features)

    class1_samplePCA = np.dot(class1_sample, w)
    class2_samplePCA = np.dot(class2_sample, w)
    all_samplesPCA = np.dot(all_samples, w)

    # TODO w = LDA(class1_samplePCA, class2_samplePCA)
    w = LDA(class1_sample, class2_sample)

    print("LDA w " + str(w.shape))
    print(str(w))

    class1_sampleLDA = np.dot(class1_samplePCA, w)
    class2_sampleLDA = np.dot(class2_samplePCA, w)
    all_samplesLDA = np.dot(all_samplesPCA, w)
    #print("Result")
    #print(all_samples)

    import matplotlib.pyplot as plt
    idx = np.fromfunction(lambda i, j: j, (1, dataSize/2), dtype=int)

    plt.plot(idx[0], class1_sampleLDA, 'ro')
    plt.plot(idx[0]+dataSize/2, class2_sampleLDA, 'bo')
    plt.show()

