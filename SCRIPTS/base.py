
from dashBoard import *
from Raw import *
from Features import *
from Data import *
from Format import *
from Filter import *
from visualizeFilter import *
from Models import *
from Wrapper import *
from HyperparamSearch import *

"""
------------------------------------------------------------------------------------------------------------------------
1.RAW DATA
------------------------------------------------------------------------------------------------------------------------
"""
"""Import libraries & Load data"""

rdat = RawData(path = DBpath, dbName = DBname, delimiter = DBdelimiter, firstLine = DBfirstLine, newLabels = None)
# rdat.visualize(displayParams, DBpath)

"""
------------------------------------------------------------------------------------------------------------------------
2.FEATURES
------------------------------------------------------------------------------------------------------------------------
"""

"""Process data & One hot encoding"""
dat = Features(rdat)
df = dat.asDataframe()
print("df", df.shape)
# print(dat)
# print(df.keys())
# print(df.get(['BREEAM Rating_Good']))

"""
------------------------------------------------------------------------------------------------------------------------
3. DATA
------------------------------------------------------------------------------------------------------------------------
"""

""" Remove outliers - only exist/removed on Quantitative features"""
""" 
Dashboard Input : 
    processingParams - cutOffThreshhold
"""
learningDf = removeOutliers(df, labels =xQuantLabels + yLabels, cutOffThreshhold=3)
print("learning", learningDf.shape)

"""
------------------------------------------------------------------------------------------------------------------------
4. FORMAT 
------------------------------------------------------------------------------------------------------------------------
"""

"""Train Test Split - Scale"""

""" 
Dashboard Input : 
    modelingParams - test_size
    modelingParams - random_state
    processingParams - yUnit
"""
import random

trainDf, testDf, MeanStdDf = formatDf(learningDf, xQuantLabels, xQualLabels, yLabels, yUnit = processingParams['yUnit'])

print("train", type(trainDf), trainDf.shape)
print("test", type(testDf), testDf.shape)

yTrain = trainDf[yLabels]
xTrain = trainDf.drop(columns=yLabels)

yTest = testDf[yLabels]
xTest = testDf.drop(columns=yLabels)


"""
------------------------------------------------------------------------------------------------------------------------
4.FILTER FEATURE
------------------------------------------------------------------------------------------------------------------------
"""
"""
SPEARMAN
"""
# uncorrelatedFilterDict, redundantFilterDict = filteringData(trainDf, baseLabels = xQuantLabels, yLabel = yLabels[0])
# filteredTrainDf, filteredTestDf = filterDf(trainDf, testDf, uncorrelatedFilterDict, redundantFilterDict)

# plotCorrelation(computeCorrelation(df), DBpath, displayParams, filteringName="nofilter")
# plotCorrelation(uncorrelatedFilterDict["correlationMatrix"], DBpath, displayParams, filteringName="dropuncorr")
# plotCorrelation(redundantFilterDict["correlationMatrix"], DBpath, displayParams, filteringName="dropcolinear")

"""
RFE
"""

rfecvDict = RFECVGridsearch(RFEEstimators, xTrain, yTrain, step, cv, scoring , display = True, testTuple = (xTest, yTest))

paramDict = RFEHyperparameterSearch(RFEEstimators,featureCount = featureCount, xTrain = xTrain, yTrain = yTrain,
                                    display = True, testTuple = (xTest, yTest))

#todo : check summary equation table
#todo : check formats - numpy vs panda / ravel()/ reshape(-1,1),...
#todo : add linear regression
#todo : how to evaluate RFE - the goal is not to perform the best prediction - what scoring should be inserted?
#todo : understand fit vs fit transform > make sure i am working with updated data

"""
------------------------------------------------------------------------------------------------------------------------
4.HYPERPARAMETER SEARCH
------------------------------------------------------------------------------------------------------------------------
"""

"""
Hyperparam Search
"""

# yTrain = filteredTrainDf[yLabels].to_numpy()
# xTrain = filteredTrainDf.drop(columns=yLabels).to_numpy()
# print('xTrain', xTrain.shape, xTrain)
# print('yTrain', yTrain.shape, yTrain)
# model = predictors[4]
# print(model)
# for k,v in model.items():
#     print (k,v)
# grid, paramDict = paramEval(model, xTrain, yTrain, custom = False, refit ='r2', rounding = 3)
#
# print(grid)
# for k,v in paramDict.items():
#     print (k,v)