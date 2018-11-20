# coding:utf-8
import os
import re
import sys
import numpy as py
sys.path.append(os.path.dirname(__file__) + os.sep + '../ImageSim')
from sim import GetSatrtXYFromPolar
import matplotlib.pyplot as plt
sys.path.insert(0, 'D:/工作/研究生/激光干扰/LaserInterEval')
from LaserInterSim import traversalDir_FirstDir, mkdir, file_name

ImageCol = 320
ImageRow = 256
TargetCol = 99
TargetRow = 107

class Record(object):
    def __init__(self, filename):
        tempfilename = os.path.basename(filename) 
        #if re.match(r'level\d_(\d)+_(\d)+[.]txt', tempfilename):
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
                    obj['iou'] = ComputeIoU(recTarget, obj['bbox'])
                    if obj['iou'] > 0.5:
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
        for i in range(0, 100, 5):
            th = float(i) / 100.0
            p,r = self.GetPRbyThreshold(th)
            self.P.append(p)
            self.R.append(r)

    def DrawPRCrve(self):
        plt.plot(self.R, self.P)
        plt.show()
        #plt.waitforbuttonpress()


def mian():
    filelist = file_name('D:/LaserData/alldection/sim', '.txt', True)
    curve = PRCurve(filelist)
    curve.DrawPRCrve()
  

if __name__ == "__main__":
    mian()