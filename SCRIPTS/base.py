
from dashBoard import *
from Raw import *
from Features import *
from Data import *
from Format import *
from Filter import *
from FilterVisualizer import *
from Wrapper import *
from WrapperVisualizer import *
from Models import *
from ModelsGridsearch import *
from ModelPredTruthPt import *
from ModelResidualsPt import *
from ModelParamPt import *

# from HyperparamSearch import *

"""
------------------------------------------------------------------------------------------------------------------------
1.RAW DATA
------------------------------------------------------------------------------------------------------------------------
"""
"""Import libraries & Load data"""

rdat = RawData(path = DB_Values['DBpath'], dbName = DB_Values['DBname'], delimiter = DB_Values['DBdelimiter'],
               firstLine = DB_Values['DBfirstLine'], updateLabels = None)
# rdat.visualize(displayParams, DBpath = DB_Values['DBpath'], dbName = DB_Values['DBname'],
#               yLabel = yLabels[0], xLabel=xQualLabels[0])
# rdat.visualize(displayParams, DBpath = DB_Values['DBpath'], dbName = DB_Values['DBname'],
#             yLabel = yLabels[0], xLabel=xQuantLabels[0])

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
    PROCESS_VALUES - OutlierCutOffThreshhold
"""
learningDf = removeOutliers(df, labels = xQuantLabels + yLabels, cutOffThreshhold=PROCESS_VALUES['OutlierCutOffThreshhold'])
print("Outliers removed ", learningDf.shape)

"""
------------------------------------------------------------------------------------------------------------------------
4. FORMAT
------------------------------------------------------------------------------------------------------------------------
"""

"""Train Validate Test Split - Scale"""

"""
Dashboard Input :
    PROCESS_VALUES - test_size  # proportion with validation
    PROCESS_VALUES - random_state
    PROCESS_VALUES - yUnit
"""


myFormatedDf = formatedDf(learningDf, xQuantLabels, xQualLabels, yLabels,
                          yUnitFactor = FORMAT_Values['yUnitFactor'],targetLabels= FORMAT_Values['targetLabels'],
                           random_state = PROCESS_VALUES['random_state'], test_size= PROCESS_VALUES['test_size'],
                          train_size= PROCESS_VALUES['train_size'])

# #todo : Should I scale my y values (targets)

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

"""Remove uncorrelated and redundant features  """

"""
Dashboard Input :
    PROCESS_VALUES - corrMethod, corrRounding, corrLowThreshhold, corrHighThreshhold  

"""

spearmanFilter = FilterFeatures(myFormatedDf.trainDf, myFormatedDf.valDf, myFormatedDf.testDf,
                                baseLabels = xQuantLabels, yLabel = myFormatedDf.yLabels[0],
                                method =PROCESS_VALUES['corrMethod'], corrRounding = PROCESS_VALUES['corrRounding'],
                                lowThreshhold = PROCESS_VALUES['corrLowThreshhold'], highThreshhold = PROCESS_VALUES['corrHighThreshhold'])

# > spearmanFilter.trainDf/valDf/testDf

# print('')
# print('FILTER - SPEARMAN CORRELATION')
# print('LABELS : ', spearmanFilter.trainDf.shape, type(spearmanFilter.trainDf))
# print(spearmanFilter.selectedLabels)
#
# plotCorrelation(spearmanFilter, spearmanFilter.correlationMatrix_All, DB_Values['DBpath'], displayParams,
#                 filteringName="nofilter")
# plotCorrelation(spearmanFilter, spearmanFilter.correlationMatrix_NoUncorrelated, DB_Values['DBpath'], displayParams,
#                 filteringName="dropuncorr")
# plotCorrelation(spearmanFilter, spearmanFilter.correlationMatrix_NoRedundant, DB_Values['DBpath'], displayParams,
#                 filteringName="dropcolinear")

#todo - save this

"""
ELIMINATE - RFE
"""

"""select the optimal number of features or combination of features"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

print(myFormatedDf.yLabels[0])

LR_RFE = WrapFeatures(method = 'LinearRegression', estimator = LinearRegression(), formatedDf = myFormatedDf,
                         yLabel = myFormatedDf.yLabels[0], featureCount = HYPERPARAMETERS['RFE_featureCount'])
RFR_RFE = WrapFeatures(method = 'RandomForestRegressor', estimator = RandomForestRegressor(random_state = PROCESS_VALUES['random_state']),
                        formatedDf = myFormatedDf, yLabel = myFormatedDf.yLabels[0],featureCount = HYPERPARAMETERS['RFE_featureCount'])
DTR_RFE = WrapFeatures(method = 'DecisionTreeRegressor', estimator = DecisionTreeRegressor(random_state = PROCESS_VALUES['random_state']),
                        formatedDf = myFormatedDf, yLabel = myFormatedDf.yLabels[0],  featureCount = HYPERPARAMETERS['RFE_featureCount'])
GBR_RFE = WrapFeatures(method = 'GradientBoostingRegressor', estimator = GradientBoostingRegressor(random_state = PROCESS_VALUES['random_state']),
                        formatedDf = myFormatedDf, yLabel = myFormatedDf.yLabels[0],  featureCount = HYPERPARAMETERS['RFE_featureCount'])
DTC_RFE = WrapFeatures(method = 'DecisionTreeClassifier', estimator = DecisionTreeClassifier(random_state = PROCESS_VALUES['random_state']),
                        formatedDf = myFormatedDf, yLabel = myFormatedDf.yLabels[0],  featureCount = HYPERPARAMETERS['RFE_featureCount'])

print('ELIMINATE - RECURSIVE FEATURE ELIMINATION')
RFEs = [LR_RFE, RFR_RFE, DTR_RFE, GBR_RFE, DTC_RFE]
for _RFE in RFEs:
    _RFE.RFECVDisplay()
    _RFE.RFEHypSearchDisplay()
    _RFE.RFEDisplay()

# todo : CONTINUE UPDATE ON DASHBOARD AND FOLDERS FROM HERE
#
# RFEHyperparameterPlot2D(RFEs,  displayParams, DBpath, reference, yLim = None, figFolder = 'RFE', figTitle = 'RFEPlot2d',
#                           title ='Influence of Feature Count on Model Performance', xlabel='Feature Count', log = False)
#
# RFEHyperparameterPlot3D(RFEs, displayParams, DBpath, reference, figFolder='RFE', figTitle='RFEPlot3d',
#                             colorsPtsLsBest=['b', 'g', 'c', 'y'],
#                             title='Influence of Feature Count on Model Performance', ylabel='Feature Count',
#                             zlabel='R2 Test score', size=[6, 6],
#                             showgrid=False, log=False, max=True, ticks=False, lims=False)

#todo : extract from this the training/val/test dfs
#todo : change naming> RFE.trainDf/valDf/testDf   > OK?


#todo : how to evaluate RFE - the goal is not to perform the best prediction - what scoring should be inserted?
#todo : understand fit vs fit transform > make sure i am working with updated data

"""
------------------------------------------------------------------------------------------------------------------------
4.HYPERPARAMETER SEARCH
------------------------------------------------------------------------------------------------------------------------
"""
#todo : finish search eval:
#todo - work on saving
# > Model Archive : save study, print study
# #todo - work on hyperparameter plot - 1 param at time
# #todo clean
#
# """
# Hyperparam Search
# """
# #todo : make other GS
#
# KRR_GS = ModelGridsearch(name = 'KRR', estimator = KernelRidge(), param_dict = KRR_param_grid, df = myFormatedDf, featureSelection = None)
# SVR_GS = ModelGridsearch(name = 'SVR', estimator = SVR(), param_dict = SVR_param_grid, df = myFormatedDf, featureSelection = None)
#
# for GS in [KRR_GS, SVR_GS]:
#     plotPredTruth(df = myFormatedDf, displayParams = displayParams, reference = reference, modelGridsearch = GS,
#                   DBpath = DBpath, fontsize = 14)
#     paramResiduals(modelGridsearch = GS, df = myFormatedDf, displayParams = displayParams, reference = reference,
#                           DBpath = DBpath, yLim = displayParams['residualsYLim'] , xLim = displayParams['residualsXLim'])
#     plotResiduals(modelGridsearch = GS, displayParams = displayParams, reference = reference, DBpath = DBpath,
#                   processingParams=PMV1Params, bins=20, binrange = [-200, 200])
#
# KRR_GS1 = ModelGridsearch(name = 'KRR_lin', estimator = KernelRidge(), param_dict = KRR_param_grid1, df = myFormatedDf, featureSelection = None)
# KRR_GS2 = ModelGridsearch(name = 'KRR_poly', estimator = KernelRidge(), param_dict = KRR_param_grid2, df = myFormatedDf, featureSelection = None)
# KRR_GS3 = ModelGridsearch(name = 'KRR_rbf', estimator = KernelRidge(), param_dict = KRR_param_grid3, df = myFormatedDf, featureSelection = None)
#
# KRRR = [KRR_GS1, KRR_GS2, KRR_GS3]
#
# GSParameterPlot2D(KRRR,  displayParams, DBpath, reference, yLim = None, paramKey ='gamma', score ='mean_test_r2', log = True)
# GSParameterPlot3D(KRRR, displayParams, DBpath, reference,
#                       colorsPtsLsBest=['b', 'g', 'c', 'y'], paramKey='gamma', score='mean_test_r2',
#                       size=[6, 6], showgrid=False, log=True, maxScore=True, absVal = False,  ticks=False, lims=False)
#
#
# #todo : check summary equation table
#
