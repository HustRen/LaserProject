# coding:utf-8
import os
import re
import sys
import numpy as np
import cv2 
sys.path.append(os.path.dirname(__file__) + os.sep + '../ImageSim')
from sim import GetSatrtXYFromPolar
import matplotlib.pyplot as plt
sys.path.insert(0, 'D:/工作/研究生/激光干扰/LaserInterEval')
from LaserInterSim import traversalDir_FirstDir, mkdir, file_name
from collections import Counter

ImageCol = 320
ImageRow = 256
TargetCol = 99
TargetRow = 107

class Record(object):
    def __init__(self, filename):
        tempfilename = os.path.basename(filename) 
        (shotname,extension) = os.path.splitext(tempfilename)#文件名、后缀名
        splitStr = shotname.split('_')
        self.laser = splitStr[0]
        self.angle = int(splitStr[1])
        self.radius= int(splitStr[2])
        startRow, startCol = GetSatrtXYFromPolar((ImageRow, ImageCol), (TargetRow, TargetCol), self.radius, self.angle)
        recTarget = [startRow, startCol, startRow + TargetRow, startCol + TargetCol]
        self.objects = []
        with open(filename, 'r') as f:
            for line in f:
                if line != 'NULL':
                    obj = {'name':'NULL', 'pro':0.0, 'bbox':[], 'iou': 0.0, 'flage': False}
                    strs = line.split(' ')
                    obj['name'] = strs[0]
                    obj['pro'] = float(strs[1])
                    obj['bbox'] = [float(strs[3]), float(strs[2]), float(strs[5]), float(strs[4])] #(topleft_row, topleft_col, botright_row, botright_col)
                    obj['iou'] = self.ComputeIoU(recTarget, obj['bbox'])
                    if obj['iou'] > 0.3:
                        obj['flage'] = True
                    self.objects.append(obj)

    def GetConfusionMatrix(self, th):
        tp = 0 #真正列
        fn = 0 #假反例
        fp = 0 #假正例
        tn = 0 #真反例

        for obj in self.objects:
            if obj['flage'] == True and obj['pro'] >= th:
                tp = tp + 1
            elif obj['flage'] == True and obj['pro'] < th:
                fn = fn + 1
            elif obj['flage'] == False and obj['pro'] < th:
                tn = tn + 1
            elif obj['flage'] == False and obj['pro'] >= th:
                fp = fp + 1
        return [tp, fn, fp, tn]

    def GetMostCorrectDection(self):
        iou = 0.0
        pro = 0.0
        for obj in self.objects:
            if obj['pro'] > pro and obj['flage'] == True:
                iou = obj['iou']
                pro = obj['pro']
        return [iou, pro]

    @staticmethod
    def ComputeIoU(rec1, rec2):
        """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
        # computing the sum_area
        sum_area = S_rec1 + S_rec2
 
        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
 
         # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return intersect / (sum_area - intersect)

class PRCurve(object):
    def __init__(self, filelist):
        self.P = []
        self.R = []
        self.mAP = 0.0 
        self.__records = []
        for file in filelist:
            record = Record(file)
            self.__records.append(record)
        self.GetPRCurveData()

    def GetPRbyThreshold(self, th):
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for record in self.__records:
            tp, fn, fp, tn = record.GetConfusionMatrix(th)
            TP = TP + tp
            FN = FN + fn
            FP = FP + fp
            TN = TN + tn
        if float(TP + FP) == 0:
            P = 0
        else:
            P = float(TP) / float(TP + FP)
        if float(TP + FN) == 0:
            R = 0
        else:
            R = float(TP) / float(TP + FN)
        return [P, R]

    def GetPRCurveData(self):
        for i in range(0, 1000, 1):
            th = float(i) / 1000.0
            p,r = self.GetPRbyThreshold(th)
            self.mAP = self.mAP + p
            self.P.append(p)
            self.R.append(r)
        self.mAP = self.mAP / 1000.0

    def DrawPRCrve(self):
        plt.xlabel('Recall')
        plt.ylabel('Precison')
        plt.xlim(0.1, 1.03)
        plt.ylim(0.0, 1.03)
        plt.plot(self.R, self.P)
        plt.show()
        #plt.waitforbuttonpress()

class Occlusion(object):
    target = cv2.imread('D:/LaserData/plane/plane1.png', cv2.IMREAD_GRAYSCALE)
    target[target > 4] = 255
    target[target <= 4] = 0
    num = 0
    for rows in target:
        for element in rows:
            if element == 255:
                num = num + 1

    SpotImage = {}
    for i in range(0, 10):
        level = 'level' + str(i)
        SpotImage[level] = cv2.imread('D:/LaserData/LaserSpot/' + level + '.bmp', cv2.IMREAD_GRAYSCALE)
        
    def __init__(self):
        pass

    def GetOcclusionRate(self, level, angle, radius):
        spotImage = Occlusion.SpotImage[level]
        startRow, startCol = GetSatrtXYFromPolar((ImageRow, ImageCol), (TargetRow, TargetCol), radius, angle)
        cooNum = 0
        for row in range(0, TargetRow):
            for col in range(0, TargetCol):
                if spotImage[startRow + row][startCol + col] == 255 and Occlusion.target[row][col] == 255:
                    cooNum = cooNum + 1
        ratio = float(cooNum) / float(Occlusion.num)
        return ratio

class InfluenceOnDetection(object):
    def __init__(self, filelist):
        occlusion = Occlusion()
        dections = []
        for file in filelist:
            dection = {'iou' : 0.0, 'pro' : 0.0, 'ratio' : 0.0, 'name': 'NULL'}
            tempfilename = os.path.basename(file) 
            (shotname,extension) = os.path.splitext(tempfilename)#文件名、后缀名
            splitStr = shotname.split('_')
            record = Record(file)
            dection['iou'], dection['pro'] = record.GetMostCorrectDection()
            dection['ratio'] = occlusion.GetOcclusionRate(splitStr[0], int(splitStr[1]), int(splitStr[2]))
            dection['name'] = shotname
            dections.append(dection)
        self.dections = sorted(dections, key=lambda dections : dections['ratio'])
        LocalType = np.dtype({'names':['totalPro', 'num'], 'formats':['f', 'int']})
        self.dectionRatio = np.zeros(11, dtype=LocalType)
        for dection in self.dections:
            index = round(dection['ratio'] * 10)
            self.dectionRatio[index]['totalPro'] += dection['pro']
            self.dectionRatio[index]['num'] += 1

        self.pro = np.zeros(11, dtype=float)
        for i in range(0, 11):
            self.pro[i] = self.dectionRatio[i]['totalPro'] / float(self.dectionRatio[i]['num'])

        print(self.pro)  

    def save(self):
        name = 'dections.txt'
        root = os.getcwd()
        path = os.path.join(root, name)
        with open(path, 'w') as file:
            for dection in self.dections:
                file.write(str(dection) + '\n')
    
    def DrawDectionLossCurve(self):
        self.save()
        plt.xlabel('Occlusion Ratio')
        plt.ylabel('Dection Pro')
        plt.xlim(-0.01, 1.03)
        plt.ylim(-0.01, 1.03)
        xdata = []
        ydata = []
        for dect in self.dections:
            xdata.append(dect['ratio'])
            ydata.append(dect['pro'])
        z = np.polyfit(xdata, ydata, 3)
        print(np.poly1d(z))
        zdata = np.polyval(z, np.arange(0.0, 1.1, 0.1))

        plt.scatter(xdata, ydata)
        plt.plot(np.arange(0.0, 1.1, 0.1), self.pro)
        plt.plot(np.arange(0.0, 1.1, 0.1), zdata)
        plt.show()
        plt.waitforbuttonpress()


def getOcclusionInfo():
    occlusion = Occlusion()
    ratio = []
    for level in range(0, 10):
        for radius in range(0, 120, 20):
            for angle in range(0, 360, 45):
                ratio.append(occlusion.GetOcclusionRate('level' + str(level), angle, radius))
    cnts, bins = np.histogram(ratio, bins = 20, range = (0.0, 1.0))
    bins = (bins[:-1] + bins[1:])/2
    plt.plot(bins, cnts)
    plt.show()

def myFilesFilter(filelist, id):
    files = []
    template = 'level'+str(id)
    for file in filelist:
        basename = os.path.basename(file)
        if basename.find(template) != -1:
            files.append(file)
    return files

def drawPRCurve():
    templatefiles = file_name('D:/LaserData/alldection/template', '.txt', True)
    allfiles = file_name('D:/LaserData/alldection/sim', '.txt', True)
    
    plt.xlabel('Recall')
    plt.ylabel('Precison')
    plt.xlim(0.1, 1.03)
    plt.ylim(0.0, 1.03)
    #allCurve = PRCurve(allfiles)
    #plt.plot(allCurve.R, allCurve.P, label='All interfered data')
    templateCurve = PRCurve(templatefiles)
    plt.plot(templateCurve.R, templateCurve.P, label='Undisturbed data')
    for i in range(7, 8):
        print('Draw level%d' % i)
        levelFiles = myFilesFilter(allfiles, i)
        levelCurve = PRCurve(levelFiles)
        plt.plot(levelCurve.R, levelCurve.P, label='Level' + str(i) + ' interfered data')
    plt.legend(loc="lower left")
    plt.show()

def mian():
    filelist = file_name('D:/LaserData/alldection/sim', '.txt', True)
    #filelist = ['D:/LaserData/alldection/sim/level5_315_40.txt']
    dections = InfluenceOnDetection(filelist)
    dections.DrawDectionLossCurve()
    #getOcclusionInfo()
    #drawPRCurve()
    #curve.DrawPRCrve()
    '''image = cv2.imread('D:/LaserData/plane/plane1.png', cv2.IMREAD_GRAYSCALE)
    plt.hist(image.ravel(), 256, [0, 256])
    image[image>4] = 255
    cv2.imshow('image', image)
    cv2.waitKey(0)
    plt.show()'''
    #plt.waitforbuttonpress()
  

if __name__ == "__main__":
    mian()