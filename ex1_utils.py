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
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np


LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
RGB_YIQ_TRANSITION_MAT = np.array([[0.299, 0.587, 0.114],
                                     [0.596, -0.275, -0.321],
                                     [0.212, -0.523, 0.311]])




def myID() -> int:
    """
    Return my ID
    :return: int
    """
    return 204155311


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Reads the Image
    im_cv = cv2.imread(filename).astype(np.float32)
    
    # Make sure there is an image
    if im_cv is None:
        raise Exception("Cannot load image!\n")

    # Convert the image
    if representation == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)
    elif representation == LOAD_RGB:
        img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    else:
        raise Exception("Please choose RGB or GRAY SCALE!\n")
    # Return a normalize image to [0,1]
    return np.multiply(img, 1 / 255)


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    if representation == LOAD_RGB:
        plt.imshow(imReadAndConvert(filename, representation))
    elif representation == LOAD_GRAY_SCALE:
        plt.imshow(imReadAndConvert(filename, representation), cmap="gray")
    else:
        raise Exception("Please choose RGB or GRAY SCALE!\n")
    plt.show()



def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    return np.dot(imgRGB.reshape(-1, 3), RGB_YIQ_TRANSITION_MAT.transpose()).reshape(imgRGB.shape)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    return np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(RGB_YIQ_TRANSITION_MAT).transpose()).reshape(imgYIQ.shape)


# Auxiliary function for 'hsitogramEqualize' func
def collect_data(imgOrig: np.ndarray) -> np.ndarray:
    """
    If the image is RGB, get the Y channel (from YIQ).
    :param imgOrig:
    :return:Y channel or Gray Scale respectively
    """
    return imgOrig.copy() if len(imgOrig.shape) != 3 else transformRGB2YIQ(imgOrig)[:, :, 0]


# Auxiliary function for 'hsitogramEqualize' func
def back_to_image(imgOrig: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    After proccesing on Y channel or GRAY SCALE, convert it back to image in RGB or Gray Scale
    :param imgOrig:
    :param data:
    :return: image on RGB or Gray Scale respectively
    """
    if len(imgOrig.shape) == 3:
        tmp = transformRGB2YIQ(imgOrig).dot(255).astype(int)
        tmp[:, :, 0] = data
        newImg = transformYIQ2RGB(tmp).astype(int)
    else:
        newImg = data

    return newImg


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # Collect the data and denormalize it
    data = collect_data(imgOrig)
    data = np.dot(data, 255).astype(int)

    # Create histogram and get the number of total pixels (the last bin from cumsum)
    histOrg, binsOrig = np.histogram(data, np.arange(257))
    arr_cumsum = np.cumsum(histOrg)
    NORMALIZE_COMP = arr_cumsum[-1]

    # Build the Look-Up Table
    LUT = np.array([int(x * (255 / NORMALIZE_COMP)) for x in arr_cumsum])

    # For each intense in the Y channel, replace by LUT
    y_channel_equalized = np.array([LUT[i] for i in data.flatten()]).reshape(data.shape)

    # Convert back to image
    imgEq = back_to_image(imgOrig, y_channel_equalized)

    # Get the histogram of equalized image
    histEq, binsEq = np.histogram(y_channel_equalized, np.arange(257))
    imgEq = np.multiply(imgEq, 1 / 255)

    return imgEq, histOrg, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
