from RawData import RawData
from Data import *
from FilteredData import *
from Dashboard import *
import seaborn as sns
import pandas as pd


"""
------------------------------------------------------------------------------------------------------------------------
1.DATA
------------------------------------------------------------------------------------------------------------------------
"""

"""Import libraries & Load data"""
rdat = RawData(csvPath, ';', 5, xQualLabels, xQuantLabels, yLabels)

"""Process data"""
dat = Data(rdat, scalers)
df = dat.asDataframe()


"""One hot encoding & Remove outliers"""
noOutlierDf = removeOutliers(df, labels = ['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)'], cutOffThreshhold=3)

# todo : !! my y labels are part of my dataframe

"""Correlation of variables & Feature selection"""
filterDf = filteredData(noOutlierDf, xQuantLabels, yLabels, plot = False)
#

trackDataProcessing(df, noOutlierDf, filterDf)

print(filterDf.keys())

"""
------------------------------------------------------------------------------------------------------------------------
2. MODEL
------------------------------------------------------------------------------------------------------------------------
"""

from PrepData import *

xdf, ydf = XYsplit(filterDf, yLabels)
x, y = normalize(xdf, ydf)
xs, ys = crossvalidationSplit(x, y)
(xTrain, yTrain), (xTests, yTests) = TrainTestSplit(xs, ys,  testSetIndex=1)


print(xTrain.keys())