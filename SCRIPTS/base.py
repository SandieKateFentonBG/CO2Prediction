
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
print("Full df", df.shape)

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
print("Outliers removed ", learningDf.shape)

"""
------------------------------------------------------------------------------------------------------------------------
4. FORMAT 
------------------------------------------------------------------------------------------------------------------------
"""

"""Train Validate Test Split - Scale"""

""" 
Dashboard Input : 
    modelingParams - test_size
    modelingParams - random_state
    processingParams - yUnit
"""


myFormatedDf = formatedDf(learningDf, xQuantLabels, xQualLabels, yLabels, yUnitFactor = processingParams['yUnit'])


print("train", type(myFormatedDf.trainDf), myFormatedDf.trainDf.shape)
print("validate", type(myFormatedDf.valDf), myFormatedDf.valDf.shape)
print("test", type(myFormatedDf.testDf), myFormatedDf.testDf.shape)

"""
------------------------------------------------------------------------------------------------------------------------
4.FEATURE SELECTION
------------------------------------------------------------------------------------------------------------------------
"""

"""
FILTER - SPEARMAN
"""
spearmanFilter = FilterFeatures(myFormatedDf.trainDf, myFormatedDf.valDf, myFormatedDf.testDf, baseLabels = xQuantLabels, yLabel = yLabels[0])

print('')
print('FILTER - SPEARMAN CORRELATION')
print('LABELS : ', spearmanFilter.trainDf.shape, type(spearmanFilter.trainDf))
print(spearmanFilter.selectedLabels)

# plotCorrelation(myFilteredDf.correlationMatrix_All, DBpath, displayParams, filteringName="nofilter")
# plotCorrelation(myFilteredDf.correlationMatrix_NoUncorrelated, DBpath, displayParams, filteringName="dropuncorr")
# plotCorrelation(myFilteredDf.correlationMatrix_NoRedundant, DBpath, displayParams, filteringName="dropcolinear")

"""
ELIMINATE - RFE
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

LR_RFE = WrapFeatures(method = 'LinearRegression', estimator = LinearRegression(), formatedDf = myFormatedDf,
                         yLabel = yLabels[0], n_features_to_select = n_features_to_select, featureCount = featureCount)
RFR_RFE = WrapFeatures(method = 'RandomForestRegressor', estimator = RandomForestRegressor(random_state = rs),
                        formatedDf = myFormatedDf, yLabel = yLabels[0], n_features_to_select = n_features_to_select, featureCount = featureCount)
DTR_RFE = WrapFeatures(method = 'DecisionTreeRegressor', estimator = DecisionTreeRegressor(random_state = rs),
                        formatedDf = myFormatedDf, yLabel = yLabels[0], n_features_to_select = n_features_to_select, featureCount = featureCount)
GBR_RFE = WrapFeatures(method = 'GradientBoostingRegressor', estimator = GradientBoostingRegressor(random_state = rs),
                        formatedDf = myFormatedDf, yLabel = yLabels[0], n_features_to_select = n_features_to_select, featureCount = featureCount)
DTC_RFE = WrapFeatures(method = 'DecisionTreeClassifier', estimator = DecisionTreeClassifier(random_state = rs),
                        formatedDf = myFormatedDf, yLabel = yLabels[0], n_features_to_select = n_features_to_select, featureCount = featureCount)

print('ELIMINATE - RECURSIVE FEATURE ELIMINATION')

for _RFE in [LR_RFE, RFR_RFE, DTR_RFE, GBR_RFE, DTC_RFE]:
    _RFE.RFEDisplay()
    _RFE.RFECVDisplay()
    _RFE.RFEHypSearchDisplay()

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


# model = predictors[4]
# print(model)
# for k,v in model.items():
#     print (k,v)
# grid, paramDict = paramEval(model, xTrain, yTrain, custom = False, refit ='r2', rounding = 3)
#
# print(grid)
# for k,v in paramDict.items():
#     print (k,v)