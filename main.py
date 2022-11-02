import numpy as np

import ImageProcess as imgp
import cv2 as cv
#__all__ = []

filename = 'undercut_1.jpg'

image = imgp.readImg(filename)

yMin, yMax, xMin, xMax = imgp.findNail(image, scale_factor=0.15, gammaFactor=1,threshold=70)

imageCropped = imgp.cropImage(image, yMin, yMax, xMin, xMax, extraHeight=0.05, extraWidth=0.05)

error = imgp.lookForDefects(imageCropped,gammaCorFactor=1, threshold=50)


