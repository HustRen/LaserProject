import os
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, 'D:/ProgramData/project/tensorflow-vgg-master/tensorflow-vgg-master')
import vgg16
import utils
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

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
            ax.hist(feature[0,:], bins=1000, histtype='stepfilled', label='plane1')
            ax.hist(feature[1,:], bins=1000, histtype='stepfilled', label='level7_0_100_patch')
            ax.legend()
            plt.show()
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