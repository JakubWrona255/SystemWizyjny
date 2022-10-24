import ImageProcess as imgp
import cv2 as cv
__all__ = []

filename = 'nail_4.jpg'

image = imgp.readImg(filename)

relHighOfCenter, relWidthOfBeginning, relWidthOfEnd = imgp.findNail(image, scale_factor=0.15)

imageCropped = imgp.cropImage(image,relHighOfCenter, relWidthOfBeginning, relWidthOfEnd,hightParam=0.1,widthParam=0.05)

imgp.lookForDefects(imageCropped)
