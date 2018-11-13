# coding:utf-8
import abc
import os
import sys
import math
import cv2
import numpy as np
sys.path.insert(0, 'D:/工作/研究生/激光干扰/LaserInterEval')
from WMSSIM import SSIM, WMS_SSIM
from WFSIM  import WFSIM, MFSIM
from feature import FeatureSuper, FeatureHog
class AlgorContext():
    def __init__(self, algor):
        self.__algor = algor

    def eval(self, template, patch):
        return self.__algor.eval(template, patch) 


class AlgorSuper(metaclass = abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractclassmethod
    def eval(self, template, patch):
        pass


class AlgorIQA(AlgorSuper):
    def __init__(self, type = 'SSIM'):
        super().__init__()
        self.__type = type

    def eval(self, template, patch):
        ans = 0.0
        if self.__type == 'WFSIM': #基于光斑性质加权的特征相识度算法
            ans = WFSIM(template, patch)
        elif self.__type == 'WMS_SSIM': #基于小波加权的SSIM评估算法
            ans = WMS_SSIM(template, patch)
        elif self.__type == 'MFSIM': #基于小波加权的SSIM评估算法
            ans = MFSIM(template, patch)
        else:
            ans = SSIM(template, patch)
        return ans


class AlgorFeature(AlgorSuper):
    def __init__(self, featuresuper):
        super().__init__()
        self.__feature = featuresuper

    def eval(self, template, patch):
        v1 = self.__feature.vector(template)
        v2 = self.__feature.vector(patch)
        dist = np.linalg.norm(v1 - v2)
        return dist
