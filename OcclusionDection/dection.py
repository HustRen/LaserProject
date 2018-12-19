# coding:utf-8
import os
import sys
import math
import cv2
import numpy as np
import abc
import matplotlib.pyplot as plt
from random import randint, gauss
from math import floor
from skimage import io, feature, transform 
import scipy.stats as st #for gaussian kernel

def test():
    I = cv2.imread('D:/LaserData/plane/paper.PNG', cv2.IMREAD_GRAYSCALE)  
    cx = 169 #centerCol
    cy = 93 #centerRow 
    px = 254
    py = 70
    patch = I[py - 7 : py + 8, px - 7 : px + 8]
    dx = 2 * abs(px - cx) + 1
    dy = 2 * abs(py - cy) + 1
    Ifx = np.zeros((1, dx), dtype=int)
    Ify = np.zeros((dy,1), dtype=int)
    Ifx[0, 0] = 1
    Ify[dy-1, 0] = 1
    result = cv2.matchTemplate(I, patch, cv2.TM_CCORR_NORMED)
    result = np.power(result, 3)
    result = cv2.filter2D(result, -1, Ifx)
    result = cv2.filter2D(result, -1, Ify)
    #cv2.imshow('result', result)
    plt.imshow(result)
    #cv2.waitKey()
    plt.waitforbuttonpress()

if __name__ == "__main__":
    test()