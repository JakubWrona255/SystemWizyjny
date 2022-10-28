import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt


def readImg(filename):
    image = cv.imread(filename)
    if image is None:
        sys.exit("Could not read the image.")
    return image


def imgPreliminaryProcessing(image,gammaCorFactor,threshold,showImg=False):

    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # convert to grayscale
    imageGray = gammaCorrection(imageGray, gammaCorFactor)  # correct lighting 1.2
    (_, imageBinary) = cv.threshold(imageGray, threshold, 255, cv.THRESH_BINARY_INV)  # convert to binary #40

    imgDilate = cv.dilate(imageBinary, None, iterations=1)
    imgDilateAndEroded = cv.erode(imgDilate, None, iterations=1)
    if showImg:
        cv.imshow("imgDilateAndEroded", imgDilateAndEroded)
        cv.waitKey(0)
    return imgDilateAndEroded


def findNail(image,scale_factor,gammaFactor,threshold):

    image2 = cv.resize(image, (0,0), fx=scale_factor, fy=scale_factor)  #make image smaller
    imgDilateAndEroded = imgPreliminaryProcessing(image2,gammaCorFactor=gammaFactor,threshold=threshold,showImg=True)
    highHistogram,widthHistogram = lateralHistogram(imgDilateAndEroded)

    xMin,xMax = scanLateralHistogram(widthHistogram,0.05)
    yMin, yMax = scanLateralHistogram(highHistogram, 0.05)

    return yMin, yMax, xMin, xMax


def scanLateralHistogram(lateralHist, changeFactor):

    detection = 0
    varMin = varMax = 0

    threshold = (np.max(lateralHist) - np.min(lateralHist)) * changeFactor
    k = 0

    for i in range(0,len(lateralHist)):  #scan histogram from beginning of nail to the end of image width
        if detection == 0:
            if lateralHist[i] < np.max(lateralHist) - threshold:
                k += 1
                if k > 4:
                    varMin = i / len(lateralHist)
                    detection = 1
                    k = 0
            else:
                k = 0

        if detection == 1:
            if lateralHist[i] > np.max(lateralHist) - threshold:
                k += 1
                if k > 4:
                    varMax = i / len(lateralHist)
                    break
            else:
                k = 0

    return varMin, varMax


def lateralHistogram(image, showPlot=False):
    high, width = image.shape
    highHistogram = np.zeros(high)
    widthHistogram = np.zeros(width)

    for i in range(0,high):
        for k in range(0,width):
            highHistogram[i] = highHistogram[i] + image[i,k]
            widthHistogram[k] = widthHistogram[k] + image[i,k]

    if showPlot:

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


def cropImage(image,yMin, yMax, xMin, xMax, extraHeight, extraWidth,showImg=False):

    orgHigh, orgWidth, colorsNum = image.shape
    highCropStart = int(orgHigh * (yMin - extraHeight))
    highCropStop = int(orgHigh * (yMax + extraHeight))
    widthCropStart = int(orgWidth * (xMin - extraWidth))
    widthCropStop = int(orgWidth * (xMax + extraWidth))

    imageCrop = image[highCropStart:highCropStop,widthCropStart:widthCropStop].copy()
    if showImg:
        cv.imshow('Cropped',imageCrop)
        cv.waitKey(0)
    return imageCrop


def lookForDefects(imageRaw, gammaCorFactor, threshold):
    errors = np.zeros(shape=5,dtype=bool)

    imageTransformed = imgPreliminaryProcessing(imageRaw, gammaCorFactor=gammaCorFactor, threshold=threshold, showImg=True)
    highHistogram, widthHistogram = lateralHistogram(imageTransformed, showPlot=True)

    errors[0] = checkIfBent(imageRaw,imageTransformed,maxAngDiff=5)        #works

    printErrors(errors)


def printErrors(errors):
    errorTypes = ['Bent nail','Problem with head','Cut point','Extra material on point','Smashed point']

    for i in range(0,len(errors)):
            print(errorTypes[i],' : ',errors[i])


def checkIfBent(imageRaw,imageTransformed,maxAngDiff):

    output = imageRaw.copy()
    edgesDet = cv.Canny(imageTransformed, 1, 1, None, 3)
    linesP = cv.HoughLinesP(edgesDet, 1, np.pi / 180, 20, None, 80, 30)

    longestLineLen = 0.0
    longestIndex = 0

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            x1,y1,x2,y2 = l[0],l[1],l[2],l[3]
            lineLen = np.sqrt((x1-x2)**2 + (y1-y2)**2)

            if lineLen > longestLineLen:    #find longest line among found lines
                longestLineLen = lineLen
                longestIndex = i

    angles = []
    detectedError = False
    if linesP is not None:
        for i in range(0, len(linesP)):
            longest = linesP[longestIndex][0]
            l = linesP[i][0]

            x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
            xl1, yl1, xl2, yl2 = longest[0], longest[1], longest[2], longest[3]

            angle1 = np.arctan(np.absolute(y1-y2)/np.absolute(x1-x2))/np.pi * 180
            angle2 = np.arctan(np.absolute(yl1 - yl2) / np.absolute(xl1 - xl2))/np.pi * 180
            angles.append(np.absolute(angle2-angle1))

    for i in range (0,len(angles)):
        if angles[i] > maxAngDiff:
            detectedError = True
            l = linesP[i][0]
            cv.line(output, (l[0], l[1]), (l[2], l[3]), (255, 0, 255), 2, cv.LINE_AA)
            l = linesP[longestIndex][0]
            cv.line(output, (l[0], l[1]), (l[2], l[3]), (255, 0, 255), 2, cv.LINE_AA)

    if detectedError:
        cv.imshow("Window", output)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return detectedError


def getLines(imageRaw, gammaCorFactor, threshold):

    imageTransformed = imgPreliminaryProcessing(imageRaw, gammaCorFactor=gammaCorFactor, threshold=threshold)
    output = imageRaw.copy()
    edgesDet = cv.Canny(imageTransformed, 1, 1, None, 3)

    linesP = cv.HoughLinesP(edgesDet, 1, np.pi / 180, 20, None, 80, 30)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(output, (l[0], l[1]), (l[2], l[3]), (255,0, 255), 2, cv.LINE_AA)

    cv.imshow("Window", output)
    cv.waitKey(0)
    cv.destroyAllWindows()
