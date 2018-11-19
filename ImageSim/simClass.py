# coding:utf-8
import os
import sys
import math
import cv2
import numpy as np
import abc

class SimAlgorSuper(metaclass = abc.ABCMeta):
    def __init__(self, im_size):
        self._data = np.zeros(im_size)

    @abc.abstractclassmethod
    def data(self):
        pass

class Noise(SimAlgorSuper):
    def __init__(self, im_size, laser):
        super().__init__(im_size)
        self.__laser = laser
    
    def data(self):
        imgRow = self._data.shape[0]
        imgCol = self._data.shape[1]
        for row in range(0, imgRow):
            for col in range(0, imgCol):
                gay =  int(self.__laser[row][col])
                self._data[row][col] = min(gay, int(255))
        return self._data

class Target(SimAlgorSuper):
    def __init__(self, im_size, target, r, angle):
        super().__init__(im_size)
        self.__target = target
        self.__r = r
        self.__angle = angle
    
    def data(self):
        imgRow, imgCol= self._data.shape
        targetRow, targetCol = self.__target.shape

        imgCenRow = imgRow / 2
        imgCenCol = imgCol / 2
        targetCenRow = targetRow / 2
        targetCenCol = targetCol / 2

        startRow  = int(min(max(imgCenRow - self.__r * math.sin((self.__angle / 180) * math.pi) - targetCenRow, 0), imgRow - targetRow))
        startCol  = int(min(max(imgCenCol + self.__r * math.cos((self.__angle / 180) * math.pi) - targetCenCol, 0), imgCol - targetCol))

        for row in range(0, targetRow):
            for col in range(0, targetCol):
                gay = int(self.__target[row][col])
                self._data[startRow + row][startCol + col] = min(gay, int(255))
        return self._data

class Background(SimAlgorSuper):
    def __init__(self, im_size, bk_image):
        super().__init__(im_size)
        self.__bk = bk_image

    def data(self):
        self._data =self.__bk
        return self._data

class SimImage():
    def __init__(self, im_size):
        self.__image = np.zeros(im_size)
        self.__datas = []
    
    def add(self, sim):
        self.__datas.append(sim.data())

    def image(self):
        for data in self.__datas:
            self.__image = self.__image + data
            self.__image[self.__image > 255] = 255
        return self.__image.astype(np.uint8)    
                
def main():
    group = 'level9'
    targetPath = 'D:/LaserData/plane/plane1.png'
    scrPath    = 'D:/LaserData/background/320X256/1.png'
    laserPath  = 'D:/LaserData/LaserSpot/' + group + '.bmp'
    ansPath    = 'D:/LaserData/ans/' + 'template' + '/'
    target = cv2.imread(targetPath, cv2.IMREAD_GRAYSCALE)
    scr    = cv2.imread(scrPath, cv2.IMREAD_GRAYSCALE)
    laser  = cv2.imread(laserPath, cv2.IMREAD_GRAYSCALE) 

    im = SimImage(scr.shape)
    im.add(Background(scr.shape, scr))
    #im.add(Noise(scr.shape, laser))
    im.add(Target(scr.shape, target, 100, 180))
    image = im.image()
    cv2.imshow('test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()