# coding:utf-8
import abc
import os
import sys
import math
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog

class FeatureSuper(metaclass = abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractclassmethod
    def vector(self, image):
        pass

class FeatureHog(FeatureSuper):
    def __init__(self):
        super().__init__()

    def vector(self, image):
        ans = hog(image)
        return ans
