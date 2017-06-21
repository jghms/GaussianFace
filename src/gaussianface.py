import numpy as np

def extractPatches(rgbImg, k, i):
    """
    Extracts patches of 25pxx25px from a face with a stride of 2.
    """
    imgWidth = rgbImg.shape[0]
    imgHeight = rgbImg.shape[1]

    xSize = int(np.ceil(imgWidth / (k * 1.0)))
    ySize = int(np.ceil(imgHeight / (k * 1.0)))

    x = (i % k ) * xSize
    y = int(i/k) * ySize
    xend = min(imgWidth, x+xSize)
    yend = min(imgHeight, y+ySize)

    return rgbImg[x:xend, y:yend]

def depricated_extractPatches(rgbImg):
    width = k
    height = k

    patches = []
    while y+height < rgbImg.shape[0]:
        x = 0
        while x+width < rgbImg.shape[1]:
            patches.append(rgbImg[x:x+width, y:y+height])
            x = x + 2
        y = y + 2

    return patches

def extractFeature(patch):
    """
    Extract patch feature using multi-scale LBP descriptor.
    """
    pass

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def s(x):
    if x >= 0:
        return 1
    else:
        return 0

def uniformity(gc, gp):

    return abs(s(gp[-1]-gc) - s(gp[0] - gc)) + sum([abs(s(gp[i] - gc) - s(gp[i-1]-gc)) for i in range(1, len(gp))])

def LBP(greyImg, R, P, x, y):
    """
    Extracts the LBP transform for a pixel in the image.
    """
    gc = greyImg[x, y]

    coords = np.array([ pol2cart(R, i*2*np.pi/P) for i in range(P) ], dtype=np.float32)
    coords = np.rint(coords)
    coords = coords.astype(int)

    res = 0
    gp = []
    for i, px in enumerate(coords):
        # Looks like this is flipped but it actually needs to be like this.
        try:
            gp.append(greyImg[x+px[1], y+px[0]])
            res = res + s(greyImg[x+px[1], y+px[0]] - gc)*2**i
        except Exception as e:
            gp.append(0)

    return (res, gc, gp)

def createIndex():
    index = {}
    i = 0

    def I(x):
        if x in index:
            return index[x]
        else:
            index[x] = len(index)
            return index[x]

    return I

def mLBP(greyImg, R, P, x, y, I):
    lbp, gc, gp = LBP(greyImg, R, P, x, y)
    if uniformity(gc, gp) <= 2:
        return I(lbp)
    else:
        return (P-1)*P+2


def H(m, R, P, I):

    h = np.zeros((1, (P-1)*P+2), dtype=int)

    for i, row in enumerate(m):
        for j, px in enumerate(row):
            res = mLBP(m, R, P, i, j, I)
            if not res == 58:
                h[0][res] = h[0][res] + 1
    return h

def F(m, R, P, I):
    f = np.zeros((R, (P-1)*P+2))

    for r in range(1, R+1):
        h = H(m, r, P, I)
        f[r-1] = h
    return f
