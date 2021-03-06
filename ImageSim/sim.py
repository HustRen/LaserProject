# coding:utf-8
import math
import os
import sys
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(__file__) + os.sep + '../ImageSim')
sys.path.append('D:/工作/研究生/激光干扰/LaserInterEval')
from LaserInterSim import file_name, mkdir, traversalDir_FirstDir


def main():
    group = 'level9'
    targetPath = 'D:/LaserData/plane/plane1.png'
    scrPath    = 'D:/LaserData/background/320X256/1.png'
    laserPath  = 'D:/LaserData/LaserSpot/' + group + '.bmp'
    ansPath    = 'D:/LaserData/ans/' + 'template' + '/'
    target = cv2.imread(targetPath, cv2.IMREAD_GRAYSCALE)
    scr    = cv2.imread(scrPath, cv2.IMREAD_GRAYSCALE)
    laser  = cv2.imread(laserPath, cv2.IMREAD_GRAYSCALE)   
    for angle in range(0, 360, 45):
        for r in range(0, 120, 20):
            filename = 'template' + '_' + str(angle) + '_' + str(r) + '.png'
            print(filename)
            simImg   = addTarget(scr, laser, target, r, angle)
            cv2.imwrite(ansPath + filename, simImg)
    '''
    cv2.imwrite(ansPath, simImg)
    cv2.imshow('sim', simImg) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
def addTarget(imgScr, laser, target, r, angle): 
    """完成激光干扰仿真
    
    Arguments:
        imgScr {[image]} -- 背景图像
        laser {[image]} -- 干扰光斑
        r {[int]} -- 光斑位置半径，以背景图的中心点的极坐标
        angle {[int]} -- 光斑位置角度（0-360），以背景图的中心点的极坐标
    """
    imgRow = imgScr.shape[0]
    imgCol = imgScr.shape[1]
    sptRow = target.shape[0]
    sptCol = target.shape[1]
    startRow, startCol = GetSatrtXYFromPolar(imgScr.shape, target.shape, r, angle)
    imgAns = np.array(imgScr)
    '''for row in range(0, imgRow):
        for col in range(0, imgCol):
            gay = int(imgScr[row][col]) + int(laserD[row][col])
            imgAns[row][col] = min(gay, int(255))'''
    for row in range(0, sptRow):
        for col in range(0, sptCol):
            gay = int(target[row][col]) + int(imgAns[startRow + row][startCol + col])
            imgAns[startRow + row][startCol + col] = min(gay, int(255))
    return imgAns

def GetSatrtXYFromPolar(imageshape, targetshape, radius, angle):
    imgRow = imageshape[0]
    imgCol = imageshape[1]
    targetRow = targetshape[0]
    targetCol = targetshape[1]

    imgCenRow = imgRow / 2
    imgCenCol = imgCol / 2
    targetCenRow = targetRow / 2
    targetCenCol = targetCol / 2
    startRow  = int(min(max(imgCenRow - radius * math.sin((angle / 180) * math.pi) - targetCenRow, 0), imgRow - targetRow))
    startCol  = int(min(max(imgCenCol + radius * math.cos((angle / 180) * math.pi) - targetCenCol, 0), imgCol - targetCol))
    return [startRow, startCol]

def cutImage():
    scrPath = 'D:/LaserData//background/640X512/area0.png'
    ansPath = 'D:/LaserData//background/320X256/'
    scrImg = cv2.imread(scrPath, cv2.IMREAD_GRAYSCALE)
    ansImg = scrImg[20:276, 30:350] #row col
    cv2.imwrite(ansPath + '1.png', ansImg)

if __name__ == '__main__':
    main()
    #cutImage()
