import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt


def readImg(filename):
    image = cv.imread(filename)
    if image is None:
        sys.exit("Could not read the image.")
    return image


def imgPreliminaryProcessing(image,gammaCorFactor,threshold):

    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # convert to grayscale
    imageGray = gammaCorrection(imageGray, gammaCorFactor)  # correct lighting 1.2

    (_, imageBinary) = cv.threshold(imageGray, threshold, 255, cv.THRESH_BINARY_INV)  # convert to binary #40

    imgDilate = cv.dilate(imageBinary, None, iterations=1)
    imgDilateAndEroded = cv.erode(imgDilate, None, iterations=1)

    cv.imshow("original", image)
    cv.imshow("imgDilate", imgDilate)
    cv.imshow("imgDilateAndEroded", imgDilateAndEroded)
    cv.imshow("imageGray", imageGray)
    cv.imshow("binary", imageBinary)

    return imgDilateAndEroded


def findNail(image,scale_factor):

    image2 = cv.resize(image, (0,0), fx=scale_factor, fy=scale_factor) #make image smaller

    imgDilateAndEroded = imgPreliminaryProcessing(image2,gammaCorFactor=1.2,threshold=50)

    highHistogram,widthHistogram = lateralHistogram( imgDilateAndEroded)
    relativeHighOfNailCenter = np.argmin(highHistogram) / len(highHistogram)
    relativeWidthOfNailBeginning = np.argmin(widthHistogram) / len(widthHistogram)

    thresholdMultiplier = 0.99  #detect end of nail when histogram values return to 0.99 of max values
    threshold = thresholdMultiplier * np.max(widthHistogram)
    relativeWidthOfNailEnd = 0

    for i in range(np.argmin(widthHistogram),len(widthHistogram)): #scan histogram from beginning of nail to the end of image width
        if widthHistogram[i] > threshold:
            relativeWidthOfNailEnd = i / len(widthHistogram)
            break  #get out of the scan
    return relativeHighOfNailCenter, relativeWidthOfNailBeginning, relativeWidthOfNailEnd


def lateralHistogram(image):
    high, width = image.shape
    highHistogram = np.zeros(high)
    widthHistogram = np.zeros(width)

    for i in range(0,high):
        for k in range(0,width):
            highHistogram[i] = highHistogram[i] + image[i,k]
            widthHistogram[k] = widthHistogram[k] + image[i,k]

    plt.subplot(121)
    plt.plot(highHistogram)
    plt.title('high histogram')

    plt.subplot(122)
    plt.plot(widthHistogram)
    plt.title('width histogram')

    plt.suptitle('lateral histograms')
    plt.show()
    return highHistogram,widthHistogram


def gammaCorrection(src, gamma):
    invGamma =  1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv.LUT(src, table)


def cropImage(image,relHighOfCenter, relWidthOfBeginning, relWidthOfEnd,hightParam,widthParam):

    orgHigh, orgWidth, colorsNum = image.shape
    highCropStart = int(orgHigh * (relHighOfCenter - hightParam))
    highCropStop = int(orgHigh * (relHighOfCenter + hightParam))
    widthCropStart = int(orgWidth * (relWidthOfBeginning - widthParam))
    widthCropStop = int(orgWidth * (relWidthOfEnd + widthParam))
    imageCrop = image[highCropStart:highCropStop,widthCropStart:widthCropStop].copy()
    cv.imshow('Cropped',imageCrop)
    cv.waitKey(0)
    return imageCrop


def lookForDefects(image):
    checkHead(image)


def checkHead(image):
    imageTransformed = imgPreliminaryProcessing(image,gammaCorFactor=1.1,threshold=50)
    highHistogram, widthHistogram = lateralHistogram(imageTransformed)
    #for i in range ()








#cv.waitKey(0)
    #cv.destroyAllWindows()

    #edgesDet = cv.Canny(imageGray, 30, 100, None, 3)
    #linesP = cv.HoughLinesP(edgesDet, 1, np.pi / 180, 50, None, 50, 10)
    #if linesP is not None:
    #    for i in range(0, len(linesP)):
    #        l = linesP[i][0]
    #        cv.line(output, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv.LINE_AA)