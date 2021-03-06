# coding:utf-8
import os
import sys

import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'D:/工作/研究生/激光干扰/LaserInterEval')
#sys.path.append(os.path.dirname(__file__) + os.sep + '../Estimate')
from textureSynthesis import OcclusionEstimateImage
from textureSynthesis import GridImage
import WFSIM
import WMSSIM

def detect_circles_demo(image):
    dst = cv2.pyrMeanShiftFiltering(image, 10, 100)   #边缘保留滤波EPF
    cv2.imshow('test', dst)
    cimage = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 20, param1=60, param2=20, minRadius=5, maxRadius=100)
    if type(circles) != None:
        for i in circles[0, : ]:
            i = np.uint16(np.around(i)) #把circles包含的圆心和半径的值变成整数
            #if image[i[0]][i[1]][0] > 200:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  #画圆
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 2)  #画圆心
    cv2.imshow("circles", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

class SpotEstimate():
    def __init__(self, scrImage):
        if scrImage.ndim == 3:
            self.occlusionImage = cv2.cvtColor(scrImage, cv2.COLOR_RGB2GRAY)
        else:
            self.occlusionImage = scrImage
        self.mask = self.getMask(self.occlusionImage)

    def getEstimateImg(self):
        eImage = OcclusionEstimateImage(self.occlusionImage, self.mask, 15)
        eImage.debugModel('D:/LaserData//plane/estimate/secondySpot')
        return eImage.textureSynthesis()

    @staticmethod
    def getMask(occlusionImage, dtype = 'circles'):
        if dtype == 'circles':
            image = cv2.cvtColor(occlusionImage, cv2.COLOR_GRAY2RGB)
            dst = cv2.pyrMeanShiftFiltering(image, 10, 100)
            cimage = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
            circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0) 
            circle = circles[0, 0, :]
            mask3 = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask3, (circle[0], circle[1]), circle[2], (255,255,255), thickness=cv2.FILLED)
            mask = cv2.cvtColor(mask3, cv2.COLOR_RGB2GRAY)
        else:
            rStr, rowStr, colStr=dtype.split('_')
            image = cv2.cvtColor(occlusionImage, cv2.COLOR_GRAY2RGB)
            mask3 = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask3, (int(rStr), int(colStr)), int(rStr) + 2, (255,255,255), thickness=cv2.FILLED)
            mask = cv2.cvtColor(mask3, cv2.COLOR_RGB2GRAY)
        return mask

class VarMeanEstimate(object):
    def __init__(self, image, grid, dtype = 'circles'):
        mask = SpotEstimate.getMask(image, dtype)
        self.gridImage = GridImage(image, grid)
        size = self.gridImage.meanImage.shape
        scaleMask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        menEst = OcclusionEstimateImage(self.gridImage.meanImage, scaleMask, 3)
        self.meanImg = menEst.textureSynthesis()
        varEst = OcclusionEstimateImage(self.gridImage.varImage, scaleMask, 3)
        self.varImg = varEst.textureSynthesis()

    def show(self):
        plt.subplot(2,2,1)
        plt.title('scrmean')
        plt.imshow(self.gridImage.meanImage)

        plt.subplot(2,2,2)
        plt.title('scrVar')
        plt.imshow(self.gridImage.varImage)

        plt.subplot(2,2,3)
        plt.title('estMean')
        plt.imshow(self.meanImg)

        plt.subplot(2,2,4)
        plt.title('estVar')
        plt.imshow(self.varImg)

        plt.show()
        plt.waitforbuttonpress()

def getfilename(fullpath):
    tname = os.path.basename(fullpath) 
    (shotname,extension) = os.path.splitext(tname)#文件名、后缀名
    return shotname

def MFSIMforMatlab(im1Path, im2Path):
    im1 = cv2.imread(im1Path, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(im2Path, cv2.IMREAD_GRAYSCALE)
    score = WFSIM.MFSIM(im1, im2)
    return score * 100

def getEstimateFeature(fullfilepath):
    image = cv2.imread(fullfilepath, cv2.IMREAD_GRAYSCALE)
    name = getfilename(fullfilepath)
    girdEst = GridImage(image, 3)
    match = re.search(r'[0-9]{1,2}_[0-9]{1,3}_[0-9]{1,3}', name)
    if match:
        vm = VarMeanEstimate(image, 3, name)
        f1 = WFSIM.SNR(girdEst.meanImage)
        f2 = WFSIM.SNR(vm.meanImg)
        f3 = WMSSIM.SSIM(vm.meanImg, girdEst.meanImage)
        f4 = WMSSIM.SSIM(vm.varImg, girdEst.varImage)
    else:
        f1 = 0
        f2 = 0
        f3 = 0
        f4 = 0
    image[image < 255] = 0
    f5 = np.sum(np.sum(image, axis=1),axis=0) / (image.shape[0] * image.shape[1] * 255)
    return [f1, f2, f3, f4, f5]

def main():
    '''scr = cv2.imread('D:/LaserData/background/1024X1024/sim/P0783__1__0___0/50_416_72.png', cv2.IMREAD_GRAYSCALE)
    scr[scr < 255] = 0
    scr = np.repeat(scr[:,:,np.newaxis], 3, axis=2)
    detect_circles_demo(scr)
    cv2.waitKey()
    cv2.destroyAllWindows()'''
    scrImage = cv2.imread('D:/LaserData/plane/path.png', cv2.IMREAD_GRAYSCALE)
    maskImage = cv2.imread('D:/LaserData/plane/estimate/mask.png', cv2.IMREAD_GRAYSCALE)
    disImage = cv2.imread('D:/LaserData/plane/estimate/occlusion.png', cv2.IMREAD_GRAYSCALE)
    estImage = cv2.imread('D:/LaserData/plane/estimate/secondySpot/5789.png', cv2.IMREAD_GRAYSCALE)
    print('-------------------------scrimage-----------------------------')
    print('scrImage SNR: %f', (WFSIM.SNR(scrImage)))
    print('disImage SNR: %f', (WFSIM.SNR(disImage)))
    print('estImage SNR: %f', (WFSIM.SNR(estImage)))
    print('scrImage and disImage MFSIM: %f', (WFSIM.MFSIM(scrImage, disImage)))
    print('estImage and disImage MFSIM: %f', (WFSIM.WFSIM(estImage, disImage)))
    print('scrImage and disImage WFSIM: %f', (WFSIM.WFSIM(scrImage, disImage)))
    print('estImage and disImage WFSIM: %f', (WFSIM.WFSIM(estImage, disImage)))
    print('scrImage and disImage WMS_SSIM: %f', (WMSSIM.WMS_SSIM(scrImage, disImage)))
    print('estImage and disImage WMS_SSIM: %f', (WMSSIM.WMS_SSIM(estImage, disImage)))
    print('scrImage and disImage SSIM: %f', (WMSSIM.SSIM(scrImage, disImage)))
    print('estImage and disImage SSIM: %f', (WMSSIM.SSIM(estImage, disImage)))
    print('-------------------------gridimage mean----------------------------')
    gridScr = GridImage(scrImage, 3)
    gridDist = GridImage(disImage, 3)
    vm = VarMeanEstimate(disImage, 3, dtype='circles')
    print('Scr mean SNR: %f', (WFSIM.SNR(gridScr.meanImage)))
    print('dist mean SNR: %f', (WFSIM.SNR(gridDist.meanImage)))
    print('gird mean SNR: %f', (WFSIM.SNR(vm.meanImg)))
    print('gird scrImage and disImage SSIM: %f', (WMSSIM.SSIM(gridScr.meanImage, gridDist.meanImage)))
    print('gird estImage and disImage SSIM: %f', (WMSSIM.SSIM(vm.meanImg, gridDist.meanImage)))
    print('gird scrImage and disImage lumComp: %f, conComp: %f', (WFSIM.lumComp(gridScr.meanImage, gridDist.meanImage), WFSIM.conComp(gridScr.meanImage, gridDist.meanImage)))
    print('gird estImage and disImage lumComp: %f, conComp: %f', (WFSIM.lumComp(vm.meanImg, gridDist.meanImage), WFSIM.conComp(vm.meanImg, gridDist.meanImage)))
    print('-------------------------gridimage var-----------------------------')
    print('gridScr SNR: %f', (WFSIM.SNR(gridScr.varImage)))
    print('gridDist SNR: %f', (WFSIM.SNR(gridDist.varImage)))
    print('gird SNR: %f', (WFSIM.SNR(vm.varImg)))
    print('gird scrImage and disImage SSIM: %f', (WMSSIM.SSIM(gridScr.varImage, gridDist.varImage)))
    print('gird estImage and disImage SSIM: %f', (WMSSIM.SSIM(vm.varImg, gridDist.varImage)))

if __name__ == "__main__":
    path0 = 'D:/LaserData/background/1024X1024/sim/P0783__1__0___0/50_416_72.png'
    path1 = 'D:/LaserData/background/1024X1024/resize/P0022__1__0___0.png'
    print(getEstimateFeature(path0))


    #g = GridImage(maskImage, 10)
    #g.show('mean')
    #g.show('var')
