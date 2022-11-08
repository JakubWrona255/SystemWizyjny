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
    imgDilateAndEroded = imgPreliminaryProcessing(image2,gammaCorFactor=gammaFactor,threshold=threshold,showImg=False)
    highHistogram,widthHistogram = lateralHistogram(imgDilateAndEroded,showPlot=False)

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
    errors = [None,None,None,None,None]

    imageTransformed = imgPreliminaryProcessing(imageRaw, gammaCorFactor=gammaCorFactor, threshold=threshold, showImg=False)
    highHistogram, widthHistogram = lateralHistogram(imageTransformed, showPlot=False)

    for i in range(0,1):
        errors[0] = checkIfBent(imageRaw,imageTransformed,maxAngDiff=5)     # works

        #if errors[0]:
        #    break

        imageHead, imagePoint = separatePointAndHead(imageRaw, highHistogram, widthHistogram)
        #cv.imshow("uszkodzone",imageHead)
        #cv.waitKey(0)


        imageHeadTransformed = imgPreliminaryProcessing(imageHead, gammaCorFactor=gammaCorFactor, threshold=threshold, showImg=False)
        imagePointTransformed = imgPreliminaryProcessing(imagePoint, gammaCorFactor=gammaCorFactor, threshold=threshold, showImg=False)

        #cv.imshow("Actual processed line on image", imagePointTransformed)

        #errors[2],errors[3] = checkIfPointCut(imagePoint,imagePointTransformed,threshold=25)

        #errors[4] = checkIfFlatHead(imagePoint,imagePointTransformed,thresholdOfArea=8500)

        #cv.imshow("głowka",imageHeadTransformed)
        checkIfWrongHead(imageHead,imageHeadTransformed)



    printErrors(errors)

#Mariusz's doing
def checkIfPointUnderCut(imagePoint,imagePointTransformed,threshold, showImg):
    #it is unused, because this functionality is implemented in function below
    pass


def checkIfPointCut(imagePoint, imagePointTransformed, threshold):
    output = imagePoint.copy()

    edgesDet = cv.Canny(imagePointTransformed, 1, 1, None, 3)
    linesP = cv.HoughLinesP(edgesDet, 1, np.pi / 180, 15, None, 0, 20)#changed from 20 -> 15

    longestLineLen = 0.0
    longestIndex = 0
    #print("Number of found lines on point cut")
    #print(len(linesP))
    #print("---------------")
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            #show actual processed line on image
            #imgToShowWithLine = output.copy()
            #cv.line(imgToShowWithLine, (l[0], l[1]), (l[2], l[3]), (255, 0, 255), 2, cv.LINE_AA)
            #cv.imshow("Actual processed line on image", imgToShowWithLine)
            #cv.waitKey(0)

            x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
            lineLen = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            if lineLen > longestLineLen:  # find longest line among found lines
                longestLineLen = lineLen
                longestIndex = i

    angles = []
    detectedErrorCutPoint = False
    detectedErrorUnderCut = False

    if linesP is not None:
        for i in range(0, len(linesP)):
            longest = linesP[longestIndex][0]
            l = linesP[i][0]

            x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
            xl1, yl1, xl2, yl2 = longest[0], longest[1], longest[2], longest[3]

            angle1 = np.arctan(np.absolute(y1 - y2) / np.absolute(x1 - x2)) / np.pi * 180
            angle2 = np.arctan(np.absolute(yl1 - yl2) / np.absolute(xl1 - xl2)) / np.pi * 180
            angles.append(np.absolute(angle2 - angle1))

    for i in range(0, len(angles)):
        #print("Difference between angles")
        #print(np.absolute(angles[i] - 90))
        if np.absolute(angles[i] - 90) < threshold:
            detectedErrorCutPoint = True
            l = linesP[i][0]
            cv.line(output, (l[0], l[1]), (l[2], l[3]), (255, 0, 255), 2, cv.LINE_AA)
            l = linesP[longestIndex][0]
            cv.line(output, (l[0], l[1]), (l[2], l[3]), (255, 0, 255), 2, cv.LINE_AA)

    if detectedErrorCutPoint:
        if (len(linesP) >=8):
            detectedErrorUnderCut = True
        pass
        #cv.imshow("Window", output)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
    return detectedErrorCutPoint,detectedErrorUnderCut


def separatePointAndHead(image,highHistogram, widthHistogram):

    xMin, xMax = scanLateralHistogram(widthHistogram,changeFactor=0.05)
    yMin, yMax = scanLateralHistogram(highHistogram, changeFactor=0.05)

    headLen = 0.02
    pointXMin = measurePoint(widthHistogram,xMin,initialBuffer=0.3,maxDeviation=0.2)

    imageHead = cropImage(image,yMin,yMax,xMin,xMin + headLen,extraHeight=0.2,extraWidth=0.02,showImg=False)
    imagePoint = cropImage(image, yMin, yMax, pointXMin, xMax, extraHeight=0.1, extraWidth=0.05, showImg=False)

    return imageHead, imagePoint


def measurePoint(histogram,xMin,initialBuffer,maxDeviation):

    pointXMin = 0.0
    startScan = int((xMin + initialBuffer)*len(histogram))
    deviation = maxDeviation * np.absolute(np.max(histogram) - np.min(histogram))
    #print(startScan)
    #print(len(histogram))
    #print(deviation)

    for i in range(startScan,len(histogram)):
        movingAverage = np.average(histogram[i-150:i-1])
        if np.absolute(movingAverage - histogram[i]) > deviation:
            pointXMin = (i-30) / len(histogram)
            #print(i)
            #print(len(histogram))
            #print(pointXMin)
            return pointXMin


def printErrors(errors):
    errorTypes = ['Bent nail','Problem with head','Cut point','Extra material on point','Smashed point']

    for i in range(0,len(errors)):
            print(errorTypes[i],' : ',errors[i])
    print("--------------------------------")


def checkIfBent(imageRaw,imageTransformed,maxAngDiff):

    output = imageRaw.copy()

    edgesDet = cv.Canny(imageTransformed, 1, 1, None, 3)
    linesP = cv.HoughLinesP(edgesDet, 1, np.pi / 180, 20, None, 100, 30)#from 80->100

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


def checkIfFlatHead(imageRaw, imagePointTransformed,thresholdOfArea):
    errorFlatHead = False
    invertedImagePointTransformed = 255 - imagePointTransformed

    contours,a = cv.findContours(invertedImagePointTransformed,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)
        if area > thresholdOfArea:
            errorFlatHead = True
            cv.drawContours(imageRaw, contour, -1, (0, 255, 0), 3)
            cv.imshow("withContour", imageRaw)
            cv.waitKey(0)
            print(area)
    return errorFlatHead

def checkIfWrongHead(imageRaw,imageHeadTransformed):
    lateralHistogram(imageHeadTransformed,True)
    output = imageRaw.copy()

    edgesDet = cv.Canny(imageHeadTransformed, 1, 1, None, 3)
    linesP = cv.HoughLinesP(edgesDet, 1, np.pi / 180, 20, None, 0, 20)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            #show actual processed line on image
            imgToShowWithLine = output.copy()
            cv.line(imgToShowWithLine, (l[0], l[1]), (l[2], l[3]), (255, 0, 255), 2, cv.LINE_AA)
            cv.imshow("Actual processed line on image", imgToShowWithLine)
            cv.waitKey(0)

