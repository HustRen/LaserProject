# coding:utf-8
import abc
import math
import os
import shutil
import sys
from math import floor
from random import gauss, randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st  # for gaussian kernel
from skimage import feature, io, transform


class GridImage(object):
    def __init__(self, img, grid):
        rows, cols = img.shape
        rows = int(rows / grid)
        cols = int(cols / grid)
        self.meanImage = np.zeros((rows, cols))
        self.varImage = np.zeros((rows, cols))
        self.scrImage = img
        self.computGrid(grid)
    
    def computGrid(self, grid):
       for row in range(0, self.meanImage.shape[0]):
           for col in range(0, self.meanImage.shape[1]):
                startRow = grid * row
                endRow   = grid * (row + 1) if row < self.meanImage.shape[0] - 1 else self.scrImage.shape[0]
                startCol = grid * col
                endCol   = grid *(col + 1) if col < self.meanImage.shape[1] - 1 else self.scrImage.shape[1]
                self.meanImage[row][col] = np.mean(self.scrImage[startRow:endRow, startCol:endCol])
                self.varImage[row][col] = np.var(self.scrImage[startRow:endRow, startCol:endCol])

    def show(self, dtype):
        if dtype == 'mean':
            plt.title('mean')
            plt.imshow(self.meanImage)
        else:
            plt.title('var')
            plt.imshow(self.varImage)
        plt.show()
        plt.waitforbuttonpress()


class OcclusionEstimateImage(object):
    def __init__(self, occlussionMap, maskMap, searchKernelSize, attenuation = 80, truncation = 0.8):
        """评估光斑遮挡的数据 
     Arguments：
         occlussionMap {image} -- 有遮挡的图像数据（单通道，或者二维数组）
         maskMap {[type]} -- 遮挡模板，有遮挡的地方为255，无遮挡的为0
         searchKernelSize {[int]} -- patch的大小
     Keyword Arguments:
         attenuation {int} -- [description] (default: {80})
         truncation {float} -- [description] (default: {0.8})
        """   
        #PARAMETERS
        self.PARM_attenuation = attenuation
        self.PARM_truncation = truncation
        #check whether searchKernelSize is odd:
        if searchKernelSize % 2 == 0:
            searchKernelSize = searchKernelSize + 1
        self.searchKernelSize = searchKernelSize

        self.canvas, self.filledMap = self.initCanvas(occlussionMap, maskMap)
        #self.examplePatches = self.prepareExamplePatches()
        self.examplePatches = self.prepareExamplePatchesByGauss()
         #init a map of best candidates to be resolved (we want to reuse the information)
        self.bestCandidateMap = np.zeros(np.shape(self.filledMap))
        #self.initCandidateMap()
        self.__DEBUG = False
        self.__DEBUGPATH = 'null'
        
    def textureSynthesis(self):
        resolved_pixels = 0
        pixels_to_resolve = np.sum(np.sum(1 - self.filledMap, axis=1), axis=0)
        while resolved_pixels < pixels_to_resolve:
            self.updateCandidateMap(5)
             #get best candidate coordinates
            candidate_row, candidate_col = self.getBestCandidateCoord()
            #get a candidatePatch to compare to
            candidatePatch = self.getNeighbourhood(self.canvas, self.searchKernelSize, candidate_row, candidate_col)
             #get a maskMap
            candidatePatchMask = self.getNeighbourhood(self.filledMap, self.searchKernelSize, candidate_row, candidate_col)
             #weight it by gaussian
            candidatePatchMask *= self.gkern(np.shape(candidatePatchMask)[0], np.shape(candidatePatchMask)[1])
            #cast to 3d array
            #candidatePatchMask = np.repeat(candidatePatchMask[:, :, np.newaxis], 3, axis=2)

             #now we need to compare it with every examplePatch and construct the distance metric
            #copy everything to match the dimensions of the examplesPatches
            examplePatches_num = np.shape(self.examplePatches)[0]
            candidatePatchMask = np.repeat(candidatePatchMask[np.newaxis, :, :,], examplePatches_num, axis=0)
            candidatePatch = np.repeat(candidatePatch[np.newaxis, :, :, ], examplePatches_num, axis=0)

            distances = candidatePatchMask * pow(self.examplePatches - candidatePatch, 2)
            distances = np.sum(np.sum(distances, axis=2), axis=1) #sum all pixels of a patch into single number

            #convert distances into probabilities 
            probabilities = self.distances2probability(distances, self.PARM_truncation, self.PARM_attenuation)
        
            #sample the constructed PMF and fetch the appropriate pixel value
            sample = np.random.choice(np.arange(examplePatches_num), 1, p=probabilities)
            chosenPatch = self.examplePatches[sample]
            halfKernel = floor(self.searchKernelSize / 2)
            chosenPixel = np.copy(chosenPatch[0,halfKernel, halfKernel])

            #resolvePixel
            if self.filledMap[candidate_row, candidate_col] == 0:
                scrgay = 255 * self.canvas[candidate_row, candidate_col]
                self.canvas[candidate_row, candidate_col] = chosenPixel
                self.filledMap[candidate_row, candidate_col] = 1
                resolved_pixels = resolved_pixels+1
                if self.__DEBUG == True:
                    if resolved_pixels%100 == 0 or resolved_pixels == pixels_to_resolve:
                        cv2.imwrite(self.__DEBUGPATH + str(resolved_pixels) + '.png', np.uint8(self.canvas*(self.max - self.min) + self.min))    
                    print('t:%d / %d, pos row:%d col:%d gay:%d scr: %d' %(resolved_pixels, pixels_to_resolve, candidate_row, candidate_col, 255*chosenPixel, scrgay))
        print('finished')
        return self.canvas*(self.max - self.min) + self.min

    def debugModel(self, path):
        if os.path.exists(path):
            del_file(path)
        else:
            os.makedirs(path)
        self.__DEBUGPATH = path + '/'
        self.__DEBUG = True

    def distances2probability(self, distances, PARM_truncation, PARM_attenuation):
        probabilities = 1 - distances / np.max(distances)  
        probabilities *= (probabilities > self.PARM_truncation)
        probabilities = pow(probabilities, self.PARM_attenuation) #attenuate the values
        #check if we didn't truncate everything!
        if np.sum(probabilities) == 0:
            #then just revert it
            probabilities = 1 - distances / np.max(distances) 
            probabilities *= (probabilities > self.PARM_truncation*np.max(probabilities)) # truncate the values (we want top truncate%)
            probabilities = pow(probabilities, self.PARM_attenuation)
        probabilities /= np.sum(probabilities) #normalize so they add up to one  
        return probabilities
    
    def getBestCandidateCoord(self):
        candidate_row, candidate_col = divmod(np.argmax(self.bestCandidateMap), self.canvas.shape[1])
        return candidate_row, candidate_col

    def updateCandidateMap(self, kernelSize):
        self.bestCandidateMap *= 1 - self.filledMap #remove all resolved from the map
        #check if bestCandidateMap is empty
        if np.argmax(self.bestCandidateMap) == 0:
            #populate from sratch
            integral = self.integral(self.filledMap)
            ridx = int(kernelSize / 2)
            for r in range(np.shape(self.bestCandidateMap)[0]):
                for c in range(np.shape(self.bestCandidateMap)[1]):
                    #self.bestCandidateMap[r, c] = np.sum(self.getNeighbourhood(self.filledMap, kernelSize, self.candidateStartRow + r, self.candidateStartCol + c))
                    # self.bestCandidateMap[r, c] = np.sum(self.getNeighbourhood(self.filledMap, kernelSize, r, c))
                     colL = max(c - ridx, 0)
                     colR = min(c + ridx, np.shape(self.bestCandidateMap)[1] - 1)
                     RowU = max(r - ridx, 0)
                     RowD = min(r + ridx, np.shape(self.bestCandidateMap)[0] - 1)
                     self.bestCandidateMap[r, c] = integral[RowD][colR] - integral[RowU][colR] - integral[RowD][colL]\
                      + integral[RowU][colL]

    def initCandidateMap(self):
        rows, cols = np.shape(self.filledMap)
        minRow = rows - 1
        maxRow = 0
        minCol = cols - 1
        maxCol = 0
        for row in range(rows):
            for col in range(cols):
                if self.filledMap[row][col] == 0:
                    minRow = min(row, minRow)
                    maxRow = max(row, maxRow)
                    minCol = min(col, minCol)
                    maxCol = max(col, maxCol)
        minRow = max(0, minRow - self.searchKernelSize)
        maxRow = min(rows, maxRow + self.searchKernelSize + 1)
        minCol = max(0, minCol - self.searchKernelSize)
        maxCol = min(cols, maxCol + self.searchKernelSize + 1)
        self.bestCandidateMap = np.zeros((maxRow - minRow, maxCol - minCol))
        self.candidateStartRow = minRow
        self.candidateEndRow = maxRow
        self.candidateStartCol = minCol
        self.candidateEndCol = maxCol

    def initCanvas(self, occlussionMap, maskMap):
        #create canvas 
        self.max = np.max(occlussionMap)
        self.min = np.min(occlussionMap)
        canvas = self.Normal(occlussionMap, self.max, self.min)
        mask = self.Normal(maskMap, 255, 0)
        assert canvas.shape == mask.shape, 'occlussionMap and maskMap have different shape'
        filledMap = 1 - mask
        return canvas, filledMap

    def prepareExamplePatches(self):
        #get exampleMap dimensions
        imgRows, imgCols= np.shape(self.canvas)
       
        #find out possible steps for a search window to slide along the image
        num_horiz_patches = imgRows - (self.searchKernelSize-1);
        num_vert_patches = imgCols - (self.searchKernelSize-1);
    
        #init candidates array
        Patches = []#np.zeros((num_horiz_patches*num_vert_patches, searchKernelSize, searchKernelSize, imgChs))

        #populate the array
        for r in range(num_horiz_patches):
            for c in range(num_vert_patches):
                patch = self.canvas[r:r+self.searchKernelSize, c:c+self.searchKernelSize]
                mask = self.filledMap[r:r+self.searchKernelSize, c:c+self.searchKernelSize]
                if mask.min() == 1:
                    Patches.append(patch)   
        examplePatches = np.array(Patches)     
        return examplePatches

    def prepareExamplePatchesByGauss(self):
        #get exampleMap dimensions
        imgRows, imgCols= np.shape(self.canvas)
        centerRow, centerCol = self.getMapCentre(1 - self.filledMap)
        mean = [0, 0]
        cov = [[1, 0], [0, 1]]
        num = int(imgCols * imgRows / 3)
        colf, rowf = np.random.multivariate_normal(mean, cov, num).T
         #init candidates array
        Patches = []#np.zeros((num_horiz_patches*num_vert_patches, searchKernelSize, searchKernelSize, imgChs))
        for i in range(num):
            r = int(rowf[i] * imgRows + centerRow)
            c = int(colf[i] * imgCols + centerCol)
            '''r = min(max(int(rf * imgRows + centerRow), 0), imgRows - self.searchKernelSize)
            c = min(max(int(cf * imgCols + centerCol), 0), imgCols - self.searchKernelSize)'''
            if r < imgRows  - self.searchKernelSize and r >=  0\
            and c < imgCols  - self.searchKernelSize and c >= 0:
                patch = self.canvas[r:r+self.searchKernelSize, c:c+self.searchKernelSize]
                mask = self.filledMap[r:r+self.searchKernelSize, c:c+self.searchKernelSize]
                if mask.min() == 1:
                    Patches.append(patch)   
        examplePatches = np.array(Patches)     
        return examplePatches
                             
    @staticmethod
    def getMapCentre(im):
        rows, cols = np.shape(im)
        rowTotal = 0
        rowCount = 0
        colTotal = 0
        colCount = 0
        for row in range(rows):
            for col in range(cols):
                if im[row][col] != 0:
                    rowTotal = rowTotal + row * im[row][col]
                    rowCount = rowCount + 1
                    colTotal = colTotal + col * im[row][col]
                    colCount = colCount + 1
        if rowCount == 0:
            centerCol = 0
            centerRow = 0
        else:
            centerRow = rowTotal / rowCount
            centerCol = colTotal / colCount
        return centerRow, centerCol

    @staticmethod
    def gkern(kern_x, kern_y, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        """altered copy from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy"""
        # X
        interval = (2*nsig+1.)/(kern_x)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kern_x+1)
        kern1d_x = np.diff(st.norm.cdf(x))
        # Y
        interval = (2*nsig+1.)/(kern_y)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kern_y+1)
        kern1d_y = np.diff(st.norm.cdf(x))
    
        kernel_raw = np.sqrt(np.outer(kern1d_x, kern1d_y))
        kernel = kernel_raw/kernel_raw.sum()
    
        return kernel

    @staticmethod
    def integral(img):
        integ_graph = np.zeros((img.shape[0],img.shape[1]),dtype = np.int32)
        for x in range(img.shape[0]):
            sum_clo = 0
            for y in range(img.shape[1]):
                sum_clo = sum_clo + img[x][y]
                integ_graph[x][y] = integ_graph[x-1][y] + sum_clo;
        return integ_graph
   
    @staticmethod
    def Normal(image, maxV, minV):
        timg = image.astype('float') - minV
        div = float(maxV - minV)
        if div != 0:
            exampleMap = timg / div #normalize
        else:
            exampleMap = np.ones(image.shape) 
        #make sure it is 3channel RGB
        #if (np.shape(exampleMap)[-1] > 3): 
        #    exampleMap = exampleMap[:,:,:3] #remove Alpha Channel
        #if (len(np.shape(exampleMap)) == 2):
        #    exampleMap = np.repeat(exampleMap[np.newaxis, :, :], 3, axis=0) #convert from Grayscale to RGB
        return exampleMap

    @staticmethod
    def getNeighbourhood(mapToGetNeighbourhoodFrom, kernelSize, row, col):
        halfKernel = floor(kernelSize / 2)
    
        if mapToGetNeighbourhoodFrom.ndim == 3:
            npad = ((halfKernel, halfKernel), (halfKernel, halfKernel), (0, 0))
        elif mapToGetNeighbourhoodFrom.ndim == 2:
            npad = ((halfKernel, halfKernel), (halfKernel, halfKernel))
        else:
            print('ERROR: getNeighbourhood function received a map of invalid dimension!')
        
        paddedMap = np.lib.pad(mapToGetNeighbourhoodFrom, npad, 'constant', constant_values=0)
    
        shifted_row = row + halfKernel
        shifted_col = col + halfKernel
    
        row_start = shifted_row - halfKernel
        row_end = shifted_row + halfKernel + 1
        col_start = shifted_col - halfKernel
        col_end = shifted_col + halfKernel + 1
    
        return paddedMap[row_start:row_end, col_start:col_end]


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)
        if os.path.isfile(path_file):
           os.remove(path_file)
        else:
            del_file(path_file)

def main():
    occluImage = cv2.imread('D:/LaserData/plane/estimate/NoSecondySpot/occlusion.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('D:/LaserData/plane/estimate/NoSecondySpot/mask.png', cv2.IMREAD_GRAYSCALE)
    test = OcclusionEstimateImage(occluImage, mask, 15, attenuation = 80, truncation = 0.8)
    test.debugModel('D:/LaserData/plane/estimate/newmodel')
    test.textureSynthesis()
    '''mean = [0, 0]
    cov = [[1, 0], [0, 1]]

    x, y = np.random.multivariate_normal(mean, cov, 1000).T
    plt.plot(x*500 + 1000, y*500 + 1000, 'x')
    plt.axis('equal')
    plt.show()
    plt.waitforbuttonpress()'''

if __name__ == "__main__":
    '''scrImg = cv2.imread('D:/LaserData/plane/P1502__1__924___0.png', cv2.IMREAD_GRAYSCALE)
    grid = GridImage(scrImg, 10)
    grid.show('mean')'''
    main()
