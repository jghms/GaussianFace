import numpy as np
from numpy import linalg as la

#from PIL import Image
import sys # TODO remove
import os.path # TODO remove
#try:
from src.gaussianface import *
import cv2
from src.align import GFAlign
#except Exception as e:
#pass

R = 10 # 10
P = 8 # 8
PCAAccuracy = 0.98 # 0.98
k = 3 # J 3 - 11
J = k*k 
featuresSize = 58
trainingSetSize = 495 # 501 (Some images does not exist)
resizeWidth = 130 # 120
resizeHeight = 150 # 142

def PCA(data):

    # Covariance matrix
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0) # TODO Check
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
    w = eigV[:, sortedV[0]];
    i = 1;
    while (important < PCAAccuracy and i < eigSize[0]) or i < 2:
        important += normV[sortedV[i]]
        w = np.vstack((w, eigV[:, sortedV[i]]));
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
        if (not label == labels[idx]) or idx == data.shape[0]-1:
            if idx == data.shape[0]-1:
                muClass += data[idx]
                dataClass = np.vstack((dataClass, data[idx]))
                objects += 1

            label = labels[idx]
            muClass = muClass / objects
            print str(label) + " Mu" 
            print muClass
            sw += ((dataClass - muClass).T.dot((dataClass - muClass)))
            sb += objects * (muClass - mu) * (muClass - mu).reshape(feat, 1);
            dataClass = np.zeros((0, feat))
            muClass = np.zeros((1, feat))
            objects = 0
        objects += 1

        muClass += data[idx]
        dataClass = np.vstack((dataClass, data[idx]))

    #print "Sb"
    #print sb
    #print "Sw"
    #print sw
    #print "Mu"
    #print mu

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
    w = eigV[:, sortedV[0]];
    i = 1;
    while (important < PCAAccuracy and i < eigSize[0]) or i < 2:
        important += normV[sortedV[i]]
        w = np.vstack((w, eigV[:, sortedV[i]]));
        i += 1;

    print "LDA Eigenvalues  " + str(i)
    #print eig
    #print "Eigenvectors"
    #print eigV
    #print "W"
    #print w.T

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

def readSet(partitionName):
    partitionsPath = './colorferet/colorferet/colorferet/dvd1/doc/partitions/'
    imagePath1 = './colorferet/colorferet/colorferet/dvd1/data/images/'
    imagePath2 = './colorferet/colorferet/colorferet/dvd2/data/images/'

    #dup1 = partitionsPath + 'dup1.txt' # 736
    #dup2 = partitionsPath + 'dup2.txt' # 228
    #fa = partitionsPath + 'fa.txt'     # 994
    #fb = partitionsPath + 'fb.txt'     # 992
    # fc = open(partitionsPath + 'fc.txt', 'r')

    filename = partitionsPath + partitionName + '.txt'

    imgCount = 0
    print filename
    num_lines = sum(1 for line in open(filename))

    limit = 5
    num_lines = min(limit, num_lines)


    I = np.zeros((num_lines, resizeWidth, resizeHeight))
    L = np.zeros((num_lines))
    print(I.shape)
    with open(filename, 'r') as fp:

        for line in fp:
            l = line.split()

            # TODO remove LIMIT test SIZE
            if limit and imgCount >= limit:
                print("imgCount (with limit): " + str(imgCount))
                return I, L

            if os.path.exists(imagePath1 + l[0] +  '/' + l[1]):
                imagePath = imagePath1 + l[0] +  '/' + l[1]
            elif os.path.exists(imagePath2 + l[0] +  '/' + l[1]):
                imagePath = imagePath2 + l[0] +  '/' + l[1]
            else:
                print(l)
                print("File not found")
                continue

            print("Found: " + l[0] +  '/' + l[1] + " Progress: " +str(round(100.0*imgCount/num_lines, 1)) + "%")


            image = cv2.imread(imagePath)

            rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            aligner = GFAlign(None)
            rects = aligner.detectAll(rgbImg)

            try:
                (x, y, w, h) = aligner.rect2BoundingBox(rects[0])
            except Exception as e:
                print("Rect error " + str(l))
                continue;

            grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thumbnail = grayImg[y:y+h,x:x+w]
            thumbnail = cv2.resize(thumbnail, (resizeHeight, resizeWidth))

            #print thumbnail
            I[imgCount][:][:] = thumbnail
            L[imgCount] = l[0]
            imgCount += 1

    print("imgCount: " + str(imgCount))
    return I, L

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
    F = np.vstack((F1, F2))
    F = np.vstack((F, F3))

    labels = np.hstack((np.ones(50) , np.ones(50) * 2,)) 
    labels = np.hstack((labels, np.ones(50)*3))

    w = LDA(F, labels)

    import matplotlib.pyplot as plt
    idx = np.fromfunction(lambda i, j: j, (1, dataSize/2), dtype=int)

    D1 = F1.dot(w)
    D2 = F2.dot(w)
    D3 = F3.dot(w)

    plt.plot(D1[:, 0], D1[:, 1], 'ro')
    plt.plot(D2[:, 0], D2[:, 1], 'bo')
    plt.plot(D3[:, 0], D3[:, 1], 'go')
    plt.show()

    sys.exit(0)

    w = PCA(F)

    D1 = F1.dot(w)
    D2 = F2.dot(w)
    D3 = F3.dot(w)

    plt.plot(D1[:, 0], D1[:, 1], 'ro')
    plt.plot(D2[:, 0], D2[:, 1], 'bo')
    plt.plot(D3[:, 0], D3[:, 1], 'go')
    plt.show()

    w2 = LDA(F.dot(w), labels)

    plt.plot(idx[0], F1.dot(w).dot(w2), 'ro')
    plt.plot(idx[0]+dataSize/2, F2.dot(w).dot(w2), 'bo')
    plt.plot(idx[0]+dataSize, F3.dot(w).dot(w2), 'go')
    plt.show()

def createFJ(img):
    FJ = np.zeros((J, img.shape[0], featuresSize)) # OBS 59!?

    for j  in range(0, J):

        for i in range(0, img.shape[0]):
            imgS = img[i][:][:]

            # Create regions
            patch = extractPatches(imgS, k, j)

            index = createIndex()

            f = F(patch, R, P, index)[:][0]

            FJ[j][i][:] = f.reshape(featuresSize)
        if (img.shape[0] > 10):
            print("FJ: " + str(j) + " done")
    if (img.shape[0] < 10):
        print("FJ: done")
    return FJ

def OLDcreateFileList(partitionName, type = 2, printFound=False):
    partitionsPath1 = './colorferet/colorferet/colorferet/dvd1/doc/partitions/'
    partitionsPath2 = './colorferet/'
    imagePath1 = './colorferet/colorferet/colorferet/dvd1/data/images/'
    imagePath2 = './colorferet/colorferet/colorferet/dvd2/data/images/'

    #dup1 = partitionsPath + 'dup1.txt' # 736
    #dup2 = partitionsPath + 'dup2.txt' # 228
    #fa = partitionsPath + 'fa.txt'     # 994
    #fb = partitionsPath + 'fb.txt'     # 992
    # fc = open(partitionsPath + 'fc.txt', 'r')

    if type == 1:
        filename = partitionsPath1 + partitionName + '.txt'
    elif type == 2:
        filename = partitionsPath2 + partitionName + '.txt'

    imgCount = 0
    print filename
    num_lines = sum(1 for line in open(filename))

    with open(filename, 'r') as fp:

        for line in fp:

            if (type == 1):
                l = line.split()

                if os.path.exists(imagePath1 + l[0] +  '/' + l[1]):
                    imagePath = imagePath1 + l[0] +  '/' + l[1]
                elif os.path.exists(imagePath2 + l[0] +  '/' + l[1]):
                    imagePath = imagePath2 + l[0] +  '/' + l[1]
                else:
                    print(str(l)  + " File not found")
                    continue

                if printFound:
                    print("Found: " + l[0] +  '/' + l[1] + " Progress: " +str(round(100.0*imgCount/num_lines, 1)) + "%")
                imgCount += 1

                image = cv2.imread(imagePath)
            if (type == 2):
                image = line.split()
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
                        print(str(imgFile) + " File not found")
                        continue
                    if printFound:
                        print("Found: " + imgFile + " Progress: " +str(round(100.0*imgCount/495, 1)) + "%")
                    imgCount += 1

                    image = cv2.imread(imagePath)
    print "Imgeas found " + partitionName + " " +  str(type), str(imgCount)


def training(J, R, P):
    img, L = readFeretFiles()
    np.save('savedMatrix/tmp', img)
    np.save('savedMatrix/labels', L)

    wLDA = []

    I = createIndex();

    for j in range(0, J-1):
        wLDA.append(WJlda(img, labels, j, R, P, I))

    return wLDA



try:
    mode = int(sys.argv[1])
except:
    mode = 3

if mode == 1:

    #img, L = readFeretFiles()
    #np.save('savedMatrix/tmp', img)
    #np.save('savedMatrix/labels', L)

    img = np.load('savedMatrix/tmp.npy')
    L = np.load('savedMatrix/labels.npy')
    print L

    FJ = createFJ(img)

    print FJ.shape
    print FJ

    np.save('savedMatrix/FJ', FJ)
elif mode == 2:
    FJ = np.load('savedMatrix/FJ.npy')
    L = np.load('savedMatrix/labels.npy')

    for j in range(0, J-1):
        wPCA = PCA(FJ[j][:][:])
        dPCA = np.dot(FJ[j][:][:], wPCA)

        print dPCA

        wLDA = LDA(dPCA, L)
        DJ = np.dot(dPCA, wLDA)

        #wLDA = LDA(FJ[j][:][:], L)
        #DJ = np.dot(FJ[j][:][:], wLDA)

        print "New DJ"
        #print DJ

        np.save('savedMatrix/wPCA'+str(j), wPCA)
        np.save('savedMatrix/wLDA'+str(j), wLDA)
        np.save('savedMatrix/DJ'+str(j), DJ)

elif mode == 3:
    # Load DJ

    DJ = []
    wLDA = []
    wPCA = []
    L = np.load('savedMatrix/labels.npy')

    for j in range(0, J-1):
        DJ.append(np.load('savedMatrix/DJ'+str(j) + '.npy'))
        wLDA.append(np.load('savedMatrix/wLDA'+str(j) + '.npy'))
        wPCA.append(np.load('savedMatrix/wPCA'+str(j) + '.npy'))

    loadFA = False
    if loadFA:
        faD = []
        for j in range(0, J-1):
            faD.append(np.load('savedMatrix/faD'+str(j)+ '.npy'))         
        faLabels = np.load('savedMatrix/faLabels'+ '.npy')

    else:
        # LOAD template set 'FA'
        faImages, faLabels = readSet('fa');

        index = createIndex()
        # Create F(J) for FA
        faFJ = createFJ(faImages)
        print faFJ
        for j in range(0, J-1):
            tmptmp = FJ(faImages, j, R, P, k, index)
            print tmptmp

        sys.exit(0)

        # Calculate DJ for FA using trainging wLDA and wPCA
        faD = []
        for j in range(0, J-1):
            faD.append(faFJ[j].dot(wPCA[j]).dot(wLDA[j]))
            # STORE Matices DJ
            np.save('savedMatrix/faD'+str(j), faD[j])
        np.save('savedMatrix/faLabels', faLabels)

    for testSet in ['fa', 'fb', 'dup1', 'dup2']: # 'fc' does not exist

        print testSet

        # LOAD testset
        testImages, testLabels = readSet(testSet)

        for idx, image in enumerate(testImages):
            
            # Calc F and D

            testD = createFJ(np.array([testImages[idx]]), 0)
            testDj = []
            for j in range(0, J-1):
                testDj.append(testD.dot(wPCA[j]).dot(wLDA[j])) 

            simStorage = []

            for img in range(0, faLabels.size):
                sim = 0
                for i in range(0, J-1):
                    sim += np.dot(testDj[i][i], np.array([faD[i][img]]).T)  / (la.norm(testDj[i]) *la.norm(faD[i][img]))
                #print("Sim: (img:" + str(img) + ", score:" + str(sim))
                simStorage.append(tuple((sim, faLabels[img])))
                
            print simStorage
            # Find best SIM and compare Labels
            sorted_sim = sorted(simStorage, key=lambda tup: tup[0],  reverse=True)
            print("Best result: " + str(sorted_sim[0]))
            print("Test Label: " + str(testLabels[idx]) + " Best label: " + str(sorted_sim[0][1]))

elif mode == 4:
    np.set_printoptions(suppress=True, threshold=(120*142))

    FJ = np.load('savedMatrix/FJ.npy')
    L = np.load('savedMatrix/labels.npy')

    print(FJ)


elif mode == 5:
    np.set_printoptions(suppress=True, threshold=(120*142))

    img = np.load('savedMatrix/tmp.npy')
    L = np.load('savedMatrix/labels.npy')

    print img[0].shape
    print img[0]

    imgplot = plt.imshow(img)

elif mode == 6:
    testData()
elif mode == 7:

    for setname in ['training', 'fa', 'fb', 'dup1', 'dup2']: # fc
        filelist = createFileList(setname)
        print(filelist)


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


# 00095_930128_fa.ppm Renamed from 00095_940128_fa.ppm
# 00086_930422_fa.ppm