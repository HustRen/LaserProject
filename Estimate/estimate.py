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

class GridImage(object):
    def __init__(self, img, grid, dtype):
        rows, cols = img.shape
        rows = int(rows / grid)
        cols = int(cols / grid)
        self.gridImage = np.zeros((rows, cols))
        self.scrImage = img
        self.computGrid(grid, dtype)
    
    def computGrid(self, grid, dtype):
       for row in range(0, self.gridImage.shape[0]):
           for col in range(0, self.gridImage.shape[1]):
               startRow = grid * row
               endRow   = grid * (row + 1) if row < self.gridImage.shape[0] - 1 else self.scrImage.shape[0]
               startCol = grid * (col - 1)
               endCol   = grid *(col + 1) if col < self.gridImage.shape[1] - 1 else self.scrImage.shape[1]
               if dtype == 'mean':
                   self.gridImage[row][col] = np.mean(self.scrImage[startRow:endRow, startCol:endCol])
               else:
                   self.gridImage[row][col] = np.var(self.scrImage[startRow:endRow, startCol:endCol])

    def girdShow(self):
        plt.imshow(self.gridImage)
        plt.show()
        plt.waitforbuttonpress()
        
class OcclusionEstimateImage(object):
    def __init__(self, occlussionMap, maskMap, searchKernelSize, attenuation = 80, truncation = 0.8):
         #PARAMETERS
        self.PARM_attenuation = attenuation
        self.PARM_truncation = truncation
        #check whether searchKernelSize is odd:
        if searchKernelSize % 2 == 0:
            searchKernelSize = searchKernelSize + 1
        self.searchKernelSize = searchKernelSize

        self.canvas, self.filledMap = self.initCanvas(occlussionMap, maskMap)
        self.examplePatches = self.prepareExamplePatches()

         #init a map of best candidates to be resolved (we want to reuse the information)
        self.bestCandidateMap = np.zeros(np.shape(self.filledMap))
        self.textureSynthesis()
        
    def textureSynthesis(self):
        resolved_pixels = 0
        pixels_to_resolve = 10000
        cv2.namedWindow('image')
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
            self.canvas[candidate_row, candidate_col] = chosenPixel
            self.filledMap[candidate_row, candidate_col] = 1
            if resolved_pixels%99 == 0:
                cv2.imwrite('D:/LaserData/plane/1/' + str(resolved_pixels) + '.png', np.uint8(self.canvas*255))
                
            resolved_pixels = resolved_pixels+1
            print('t:%d, pos row:%d col:%d gay:%d' %(resolved_pixels, candidate_row, candidate_col, 255*chosenPixel))

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

    @staticmethod
    def covertRGB(image):
        exampleMap = image / 255.0 #normalize
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

    def updateCandidateMap(self, kernelSize):
        self.bestCandidateMap *= 1 - self.filledMap #remove all resolved from the map
        #check if bestCandidateMap is empty
        if np.argmax(self.bestCandidateMap) == 0:
            #populate from sratch
            for r in range(np.shape(self.bestCandidateMap)[0]):
                for c in range(np.shape(self.bestCandidateMap)[1]):
                    self.bestCandidateMap[r, c] = np.sum(self.getNeighbourhood(self.filledMap, kernelSize, r, c))

    def initCanvas(self, occlussionMap, maskMap):
        #create canvas 
        canvas = self.covertRGB(occlussionMap)
        mask = maskMap / 255
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

def main():
    occluImage = cv2.imread('D:/LaserData/plane/occlusion.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('D:/LaserData/plane/mask.png', cv2.IMREAD_GRAYSCALE)
    test = OcclusionEstimateImage(occluImage, mask, 15, attenuation = 80, truncation = 0.8)

if __name__ == "__main__":
    '''scrImg = cv2.imread('D:/LaserData/plane/P1502__1__924___0.png', cv2.IMREAD_GRAYSCALE)
    grid = GridImage(scrImg, 10, 'var')
    grid.girdShow()'''
    main()

