# coding:utf-8
import os
import sys
import math
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
sys.path.insert(0, 'D:/工作/研究生/激光干扰/LaserInterEval')
from LaserInterSim import traversalDir_FirstDir, mkdir, file_name
from evalAlgorithm import AlgorContext, AlgorSuper, AlgorIQA, AlgorFeature
from feature import FeatureSuper, FeatureHog, FeatureRaw

def main():
    files = file_name('D:/LaserData/ans/level7','.png', True)
    plt.figure(1)
    plt.title("Evaluation Result")
    plt.xlabel('Radius(pixel)')
    plt.ylabel('Correlation')
    for angle in range(0, 360, 45):
        data = getEvalData(files, angle)
        plt.plot(data[0], data[1], label='angle'+str(angle))
    plt.legend(loc="upper left")
    plt.show()
    

def getEvalData(fileList, angle):
    """获取每一组干扰仿真图像的评估结果
    Arguments:
        fileList {str} -- 文件夹路径如'D:/LaserData/ans/level0'
        angle {int} -- 角度 我们以一个角度为一组
    Returns:
        [list] --  [listX, listY]该组数据的自变量目标离光斑距离listX;干扰评估结果listY    
    """
    listX = []
    listY = []
    context = AlgorContext(AlgorIQA())
    #context = AlgorContext(AlgorFeature(FeatureRaw()))
    plane = cv2.imread('D:/LaserData/plane/plane1.png', cv2.IMREAD_GRAYSCALE)
    for file in fileList:
        (filepath,tempfilename) = os.path.split(file) #文件路径、文件名+后缀名
        (shotname,extension) = os.path.splitext(tempfilename)#文件名、后缀名
        splitStr = shotname.split('_')
        if(len(splitStr) == 3 and angle == int(splitStr[1])):
            patch = getPlanePatch(file)
            listX.append(int(splitStr[2])) #半径
            listY.append(context.eval(plane, patch))
    indexs = np.argsort(listX)
    X = []
    Y = []
    for index in indexs:
        X.append(listX[index])
        Y.append(listY[index])
    return [X, Y]                  


def getPlanePatch(filename):
    """获得仿真图像中的目标块，返回目标快的numpy数组
    Arguments: 
        filename {str} -- 仿真图像文件路径
    Returns:
        [image] -- 目标图像块数据
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    (filepath,tempfilename) = os.path.split(filename) #文件路径、文件名+后缀名
    (shotname,extension) = os.path.splitext(tempfilename)#文件名、后缀名
    splitStr = shotname.split('_')
    r = int(splitStr[2])
    angle = int(splitStr[1])
    imgRow = img.shape[0]
    imgCol = img.shape[1]
    sptRow = 107
    sptCol = 99
    imgCenRow = imgRow / 2
    imgCenCol = imgCol / 2
    sptCenRow = sptRow / 2
    sptCenCol = sptCol / 2
    startRow  = int(min(max(imgCenRow - r * math.sin((angle / 180) * math.pi) - sptCenRow, 0), imgRow - sptRow))
    startCol  = int(min(max(imgCenCol + r * math.cos((angle / 180) * math.pi) - sptCenCol, 0), imgCol - sptCol))
    patch = img[startRow:startRow+sptRow, startCol:startCol+sptCol]
    return patch

if __name__ == '__main__':
    main()