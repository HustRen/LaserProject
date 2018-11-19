# coding:utf-8
import abc
import tensorflow as tf
import os
import sys
import math
import cv2
import numpy as np
sys.path.insert(0, 'D:/工作/研究生/激光干扰/LaserInterEval')
sys.path.insert(0, 'D:/ProgramData/project/tensorflow-vgg-master/tensorflow-vgg-master')
import vgg16
import utils
from WMSSIM import SSIM, WMS_SSIM
from WFSIM  import WFSIM, MFSIM
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from PIL import Image
from skimage.feature import hog
import skimage
import skimage.io
import skimage.transform


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

class AlgorVGG16(AlgorSuper):
    sess = tf.Session()
    images = tf.placeholder("float", [2, 224, 224, 3])
    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    def __init__(self):
        super().__init__()

    def eval(self, template, patch):
        dist = 0.0
        img1 = utils.resizeImg(template)
        img2 = utils.resizeImg(patch)

        batch1 = img1.reshape((1, 224, 224, 3))
        batch2 = img2.reshape((1, 224, 224, 3))
        batch = np.concatenate((batch1, batch2), 0)
        feed_dict = {AlgorVGG16.images: batch}
        feature = AlgorVGG16.sess.run(AlgorVGG16.vgg.fc8, feed_dict=feed_dict)
        dist = np.linalg.norm(feature[0,:] - feature[1,:])
        return dist

class AlgorFeature(AlgorSuper):
    def __init__(self, featuresuper):
        super().__init__()
        self.__feature = featuresuper

    def eval(self, template, patch):
        v1 = self.__feature.vector(template)
        v2 = self.__feature.vector(patch)
        dist = np.linalg.norm(v1 - v2)
        return dist

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

class FeatureRaw(FeatureSuper):
    def __init__(self):
        super().__init__()

    def vector(self, image):
        return image.flatten()




def mian():
    base1 = 'D:/LaserData/plane/plane1.png'
    base2 = 'D:/LaserData/ans/level7/patch/level7_0_100_patch.png'
    img1 = utils.load_image(base1)
    img2 = utils.load_image(base2)

    batch1 = img1.reshape((1, 224, 224, 3))
    batch2 = img2.reshape((1, 224, 224, 3))

    batch = np.concatenate((batch1, batch2), 0)

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    fig,ax = plt.subplots()
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [2, 224, 224, 3])
            feed_dict = {images: batch}

            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            feature = sess.run(vgg.fc8, feed_dict=feed_dict)
            ax.hist(feature[0,:], label='plane1')
            ax.hist(feature[1,:], label='level7_0_100_patch')
            ax.legend()
            plt.waitforbuttonpress()
           # draw(data)
            #print(prob)

def draw(data):
    sc = plt.imshow(data)
    sc.set_cmp('gray')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    mian()