import numpy as np
import ImageProcess as imgp


def doTheThing(case,num):
    filename = filenameOrigin[case] + str(num) + '.jpg'
    print(filename)
    print("----")
    image = imgp.readImg(filename)
    yMin, yMax, xMin, xMax = imgp.findNail(image, scale_factor=0.15, gammaFactor=1, threshold=70)
    imageCropped = imgp.cropImage(image, yMin, yMax, xMin, xMax, extraHeight=0.05, extraWidth=0.05)
    error = imgp.lookForDefects(imageCropped, gammaCorFactor=1, threshold=50)


filenameOrigin = ['img/flatHead_','img/bent_','img/head_','img/pointCut_','img/undercut_','img/control_']

for i in range(1,7,1):  #flathead
    doTheThing(0,i)

for i in range(1,4,1):  #bent
    doTheThing(1,i)

for i in range(1,6,1):  #head
    doTheThing(2,i)

for i in range(1,5,1):  #pointCut
    doTheThing(3,i)

for i in range(1,13,1): #undercut
    doTheThing(4,i)

for i in range(1,10,1):  #control
    doTheThing(5,i)


