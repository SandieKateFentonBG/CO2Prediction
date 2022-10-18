
from dashBoard import *
from Raw import *
from Features import *
from Data import *
from Format import *
from Filter import *
from visualizeFilter import *
from Wrapper import *
from WrapperVisualizer import *
from Models import *
from ModelsGridsearch import *
from ModelPredTruthPt import *
from ModelResidualsPt import *

# from HyperparamSearch import *

"""
------------------------------------------------------------------------------------------------------------------------
1.RAW DATA
------------------------------------------------------------------------------------------------------------------------
"""
"""Import libraries & Load data"""

rdat = RawData(path = DBpath, dbName = DBname, delimiter = DBdelimiter, firstLine = DBfirstLine, newLabels = None)
# rdat.visualize(displayParams, DBpath, DBname, reference, yLabel = yLabels[0], xLabel=xQualLabels[0])
# rdat.visualize(displayParams, DBpath, DBname, reference, yLabel = yLabels[0], xLabel=xQuantLabels[0])

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


myFormatedDf = formatedDf(learningDf, xQuantLabels, xQualLabels, yLabels, yUnitFactor = processingParams['yUnitFactor'],
                          targetLabels= processingParams['targetLabels'])

#todo : check unit > when in tons? when in kgs?
print("train", type(myFormatedDf.trainDf), myFormatedDf.trainDf.shape)
print("validate", type(myFormatedDf.valDf), myFormatedDf.valDf.shape)
print("test", type(myFormatedDf.testDf), myFormatedDf.testDf.shape)

print("train", myFormatedDf.trainDf)

"""
------------------------------------------------------------------------------------------------------------------------
4.FEATURE SELECTION
------------------------------------------------------------------------------------------------------------------------
"""

"""
FILTER - SPEARMAN
"""

"""Remove uncorrelated and redundant features  """

# spearmanFilter = FilterFeatures(myFormatedDf.trainDf, myFormatedDf.valDf, myFormatedDf.testDf, baseLabels = xQuantLabels, yLabel = myFormatedDf.yLabels[0])
# #
# print('')
# print('FILTER - SPEARMAN CORRELATION')
# print('LABELS : ', spearmanFilter.trainDf.shape, type(spearmanFilter.trainDf))
# print(spearmanFilter.selectedLabels)

# plotCorrelation(spearmanFilter.correlationMatrix_All, DBpath, displayParams, filteringName="nofilter", reference = reference)
# plotCorrelation(spearmanFilter.correlationMatrix_NoUncorrelated, DBpath, displayParams, filteringName="dropuncorr", reference = reference)
# plotCorrelation(spearmanFilter.correlationMatrix_NoRedundant, DBpath, displayParams, filteringName="dropcolinear", reference = reference)

#todo - save this

"""
ELIMINATE - RFE
"""

"""select the optimal number of features or combination of features"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

LR_RFE = WrapFeatures(method = 'LinearRegression', estimator = LinearRegression(), formatedDf = myFormatedDf,
                         yLabel = myFormatedDf.yLabels[0], n_features_to_select = n_features_to_select, featureCount = featureCount)
RFR_RFE = WrapFeatures(method = 'RandomForestRegressor', estimator = RandomForestRegressor(random_state = rs),
                        formatedDf = myFormatedDf, yLabel = myFormatedDf.yLabels[0], n_features_to_select = n_features_to_select, featureCount = featureCount)
DTR_RFE = WrapFeatures(method = 'DecisionTreeRegressor', estimator = DecisionTreeRegressor(random_state = rs),
                        formatedDf = myFormatedDf, yLabel = myFormatedDf.yLabels[0], n_features_to_select = n_features_to_select, featureCount = featureCount)
GBR_RFE = WrapFeatures(method = 'GradientBoostingRegressor', estimator = GradientBoostingRegressor(random_state = rs),
                        formatedDf = myFormatedDf, yLabel = myFormatedDf.yLabels[0], n_features_to_select = n_features_to_select, featureCount = featureCount)
DTC_RFE = WrapFeatures(method = 'DecisionTreeClassifier', estimator = DecisionTreeClassifier(random_state = rs),
                        formatedDf = myFormatedDf, yLabel = myFormatedDf.yLabels[0], n_features_to_select = n_features_to_select, featureCount = featureCount)

print('ELIMINATE - RECURSIVE FEATURE ELIMINATION')
RFEs = [LR_RFE, RFR_RFE, DTR_RFE, GBR_RFE, DTC_RFE]
for _RFE in RFEs:

    _RFE.RFECVDisplay()
    _RFE.RFEHypSearchDisplay()
    _RFE.RFEDisplay()

RFEHyperparameterPlot2D(RFEs,  displayParams, DBpath, reference, yLim = None, figFolder = 'RFE', figTitle = 'RFEPlot2d',
                          title ='Influence of Feature Count on Model Performance', xlabel='Feature Count', log = False)

RFEHyperparameterPlot3D(RFEs, displayParams, DBpath, reference, figFolder='RFE', figTitle='RFEPlot3d',
                            colorsPtsLsBest=['b', 'g', 'c', 'y'],
                            title='Influence of Feature Count on Model Performance', ylabel='Feature Count',
                            zlabel='R2 Test score', size=[6, 6],
                            showgrid=False, log=False, max=True, ticks=False, lims=False)

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
#todo : make other GS
# KRR_GS = ModelGridsearch(name = 'KRR', estimator = KernelRidge(), param_dict = KRR_param_grid, df = myFormatedDf)
# SVR_GS = ModelGridsearch(name = 'SVR', estimator = SVR(), param_dict = SVR_param_grid, df = myFormatedDf)
#
# for GS in [KRR_GS, SVR_GS]:
#     plotPredTruth(df = myFormatedDf, displayParams = displayParams, reference = reference, modelGridsearch = GS,
#                   DBpath = DBpath, fontsize = 14)
#     paramResiduals(modelGridsearch = GS, df = myFormatedDf, displayParams = displayParams, reference = reference,
#                           DBpath = DBpath, yLim = displayParams['residualsYLim'] , xLim = displayParams['residualsXLim'])
#     plotResiduals(modelGridsearch = GS, displayParams = displayParams, reference = reference, DBpath = DBpath,
#                   processingParams=processingParams,bins=20, binrange = (-200, 200))

#todo : finish search eval:

# > ModelPredTruthPt : paramResiduals, plotResiduals ;
# > Model Archive : save study, print study




#todo : check summary equation table

