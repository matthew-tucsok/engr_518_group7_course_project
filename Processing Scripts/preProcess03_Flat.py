from PIL import Image, ImageOps
import numpy as np
import glob

k = 0

desiredWidth = 64
desiredHeight = 64

image_list = []
for filename in glob.glob('../RawImages/Flat/*.JPG'):  # assuming gif
    img = Image.open(filename)
    width, height = img.size
    print(img.size)
    for i in range(desiredWidth, width, desiredWidth):
        for j in range(desiredHeight, height, desiredHeight):
            area = (i - desiredWidth, j - desiredHeight, i, j)
            # image[i*(width-desiredWidth)+j]=img.crop(area)
            tmpImg = img.crop(area)
            grayTmpImg = ImageOps.grayscale(tmpImg)
            # grayTmpImg.save(('../processedImages01/'+str(int((i-desiredWidth)/desiredWidth*int(width/desiredWidth)+(j-desiredHeight)/desiredHeight))+'.JPG'))
            # image.append(img.crop(area))
            grayTmpImg.save(('../processedImagesFlat/F_' + str(1000 + k) + '.JPG'))
            k += 1

print(k)
