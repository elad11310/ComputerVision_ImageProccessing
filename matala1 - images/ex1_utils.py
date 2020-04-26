"""
         '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""

from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
NORMALIZE = 1 / 255
DNORMALIZE = 255


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 203306014


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    m = plt.imread(filename)
    ##d = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  ## reading with cv2 - greyScale - BGR
    ##d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)  ## convert BGR to RGB - because its GreyScae all 3 layers are the same.

    ## for greyScale
    if (representation == LOAD_GRAY_SCALE):
        temp = np.dot(m[..., :3], [0.299, 0.587, 0.114])
    ## for RGB
    else:
        temp = np.copy(m).astype(float)

    ## Normalize
    temp = temp.dot(NORMALIZE)
    return temp


def imDisplay(filename: str, representation: int):
    img = imReadAndConvert(filename, representation)
    ## if we need to display in grey
    if (representation == LOAD_GRAY_SCALE):
        plt.gray()
    plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
   # rgbToYiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
   # #imgIYQ = imgRGB.copy()
   # imgYIQ = imgRGB.dot(rgbToYiq)

    y1y = (0.299 * imgRGB[:,:,0] + 0.587 * imgRGB[:,:,1] + 0.114 * imgRGB[:,:,2])
    y1i = (0.596 * imgRGB[:,:,0] - 0.275 * imgRGB[:,:,1] - 0.321 * imgRGB[:,:,2])
    y1q = (0.212 * imgRGB[:,:,0] - 0.523 * imgRGB[:,:,1] + 0.311 * imgRGB[:,:,2])
    #
    imgYIQ = np.zeros_like(imgRGB)
    imgYIQ[:,:,0]=y1y
    imgYIQ[:, :, 1] = y1i
    imgYIQ[:, :, 2] = y1q

    return imgYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
   # yiqToRgb = np.array([[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]])
   # ##imgRGB = imgYIQ.copy()
   # imgRGB = imgYIQ.dot(yiqToRgb)

    rgb1 = (1 * imgYIQ[:, :, 0] + 0.956 * imgYIQ[:, :, 1] + 0.619 * imgYIQ[:, :, 2])
    rgb2 = (1 * imgYIQ[:, :, 0]  -0.272 * imgYIQ[:, :, 1] -0.647 * imgYIQ[:, :, 2])
    rgb3 = (1 * imgYIQ[:, :, 0] -1.106 * imgYIQ[:, :, 1] +  1.703 * imgYIQ[:, :, 2])

    imgRGB=np.zeros_like(imgYIQ)

    imgRGB[:, :, 0] = rgb1
    imgRGB[:, :, 1] = rgb2
    imgRGB[:, :, 2] = rgb3

    return imgRGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):

    ## if the img is RGB taking only the y channel
    if imgOrig.ndim == 3:
        ## transform from rgb to yiq
        imgYIQ = transformRGB2YIQ(imgOrig)
        yChannel = imgYIQ[:, :, 0]

    else:
        yChannel = imgOrig
    ## taking shape of the photo
    w, h, = yChannel.shape
    ## making histogram - counting how many pixels from each intensity from 0-255
    hist, bin_edges = np.histogram(yChannel, 256)

    ## now converting the histogram into cumulative histogram
    cumulativeHist = calCumSum(hist, w, h)
    ##cumulativeHist = np.cumsum(hist)

    ## now make the LUT - probability

    LUT = [cumulativeHist[i] / np.size(yChannel) for i in range(0, len(cumulativeHist))]
    ## scaling
    yChannel = (yChannel - yChannel.min()) * 255 / (yChannel.max() - yChannel.min())
    ## round all values
    yChannel = np.round(yChannel)
    ## mapping the intensity values of the new image
    for i in range(0, w):
        for j in range(0, h):
            yChannel[i][j] = LUT[(int)(round(yChannel[i][j]))]

    ## attaching the new y channel
    if imgOrig.ndim == 3:
        imgYIQ[:, :, 0] = yChannel
        imgEq = transformYIQ2RGB(imgYIQ)
        histEQ, y = np.histogram(yChannel, 256)
    else:
        imgEq = yChannel
        histEQ, y = np.histogram(imgEq, 256)

    return imgEq, hist, histEQ


def calCumSum(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    ## this function computes the cumulative histogram based on the given histogram.

    cum_sum = np.zeros(len(arr))  ## new arr of zeros in size of arr
    cum_sum[0] = arr[0]  ## copy the first element that indicates the number of pixels with inensity 0

    arr_len = len(arr)

    ## now summing untill the current index level of intesity pixels and normalize
    for index in range(1, arr_len):
        cum_sum[index] = arr[index] + cum_sum[index - 1]

    return cum_sum


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    ## if the img is RGB
    if imOrig.ndim == 3:
         ## transform from rgb to yiq
        imgYIQ = transformRGB2YIQ(imOrig)
        ## taking only the y channel
        yChannelOrig = imgYIQ[:, :, 0]
        yChannel = imgYIQ[:, :, 0]

    ## if its Grey scale
    else:
         yChannelOrig = imOrig
         yChannel = imOrig

    ## setting histogram for y channel
    hist, bin_edges = np.histogram(yChannelOrig, 255)

    # numbers of borders to split
    K = nQuant
    # borders spliting
    borders = [i for i in range(0, len(hist) + 1, (int)(255 / K))]
    if borders[len(borders) - 1] != 255:
        borders[len(borders) - 1] = 255


    Q = np.zeros(K)
    errorArr = np.ndarray((nIter), float) ## error list
    imgList = np.ndarray((nIter), np.ndarray) ## img list

    for k in range(0, nIter):
        for i in range(0, K):
            totalPixels = 0
            weightedAvg = 0
            yChannel = np.copy(yChannelOrig)

            if i != K:
                for j in range(borders[i], borders[i + 1]):
                    totalPixels += hist[j]  # how many pixels
                    weightedAvg += hist[j] * j
                Q[i] = weightedAvg / totalPixels

        # multiply and scale the y channel to 0-255
        yChannel = (yChannel - yChannel.min()) * 255 / (yChannel.max() - yChannel.min())
        yChannel = np.round(yChannel)
        for i in range(0, K):
            if i != K:
                ## changing the segments intensities to selected q
                yChannel[np.where(
                    np.logical_and(np.greater_equal(yChannel, borders[i]), np.less_equal(yChannel, borders[i + 1])))] = \
                Q[i]

            ##checking MSE

        ## if its RGB
        if imOrig.ndim == 3:
            yChannel = yChannel.dot(NORMALIZE)
            imgYIQ[:, :, 0] = yChannel
            errorArr[k] = np.sqrt(np.power(np.sum(imOrig - imgYIQ), 2)) / np.size(imOrig)
            imgYIQ = transformYIQ2RGB(imgYIQ)
            imgList[k] = imgYIQ
            imgYIQ = transformRGB2YIQ(imgYIQ)

        ## if its Grey Scale
        else:
            imgList[k] = yChannel
            errorArr[k] = np.sqrt(np.power(np.sum(yChannel - yChannelOrig), 2)) / np.size(yChannel)

        #changing the borders
        for b in range(1, K):
            borders[b] = (int)((Q[b - 1] + Q[b]) / 2)


    return imgList, errorArr
