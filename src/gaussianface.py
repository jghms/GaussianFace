

def extractPatches(rgbImg):
    """
    Extracts patches of 25pxx25px from a face with a stride of 2.
    """
    x = 0
    y = 0
    width = 25
    height = 25

    patches = []
    while y+height < rgbImg.shape[0]:
        x = 0
        while x+width < rgbImg.shape[1]:
            patches.append(rgbImg[x:x+width, y:y+height])
            x = x + 2
        y = y + 2

    return patches
