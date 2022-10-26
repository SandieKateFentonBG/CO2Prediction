#SCRIPT IMPORTS
from dashBoard import *
from HelpersArchiver import *
from Raw import *
from Features import *
from Data import *
from Format import *
from Filter import *
from FilterVisualizer import *
from Wrapper import *
from WrapperReport import *
from WrapperVisualizer import *
from ProcessingReport import *
from Model import *
from ModelPredTruthPt import *
from ModelResidualsPt import *
from ModelParamPt import *
from ModelMetricsPt import *
from ModelWeightsPt import *
from Gridsearch import *


#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge


"""
------------------------------------------------------------------------------------------------------------------------
1.RAW DATA
------------------------------------------------------------------------------------------------------------------------
"""

#ABOUT
"""
GOAL - Import libraries & Load data
"""

#CONSTRUCT
rdat = RawData(path = DB_Values['DBpath'], dbName = DB_Values['DBname'], delimiter = DB_Values['DBdelimiter'],
               firstLine = DB_Values['DBfirstLine'], updateLabels = None)

#VISUALIZE
rdat.visualize(displayParams, DBpath = DB_Values['DBpath'], dbName = DB_Values['DBname'],
              yLabel = yLabels[0], xLabel=xQualLabels[0])
rdat.visualize(displayParams, DBpath = DB_Values['DBpath'], dbName = DB_Values['DBname'],
            yLabel = yLabels[0], xLabel=xQuantLabels[0])

"""
# ------------------------------------------------------------------------------------------------------------------------
# 2.FEATURES
# ------------------------------------------------------------------------------------------------------------------------
# """
#ABOUT
"""
GOAL - Process data & One hot encoding
"""

#CONSTRUCT
dat = Features(rdat)
df = dat.asDataframe()

#REPORT
# print("Full df", df.shape)

#STOCK
pickleDumpMe(DB_Values['DBpath'], displayParams, df, 'DATA', 'df')

"""
------------------------------------------------------------------------------------------------------------------------
3. DATA
------------------------------------------------------------------------------------------------------------------------
"""
#ABOUT
"""
GOAL - Remove outliers - only exist/removed on Quantitative features
Dashboard Input - PROCESS_VALUES : OutlierCutOffThreshhold
"""
#CONSTRUCT
learningDf = removeOutliers(df, labels = xQuantLabels + yLabels, cutOffThreshhold=PROCESS_VALUES['OutlierCutOffThreshhold'])

# #REPORT
# print("Outliers removed ", learningDf.shape)

#STOCK
pickleDumpMe(DB_Values['DBpath'], displayParams, learningDf, 'DATA', 'learningDf')


"""
------------------------------------------------------------------------------------------------------------------------
4. FORMAT
------------------------------------------------------------------------------------------------------------------------
"""
#ABOUT
"""
GOAL - Train Validate Test Split - Scale
Dashboard Input - PROCESS_VALUES : test_size  # proportion with validation, random_state, yUnit
"""

#CONSTRUCT
baseFormatedDf = formatedDf(learningDf, xQuantLabels, xQualLabels, yLabels,
                            yUnitFactor = FORMAT_Values['yUnitFactor'], targetLabels= FORMAT_Values['targetLabels'],
                            random_state = PROCESS_VALUES['random_state'], test_size= PROCESS_VALUES['test_size'],
                            train_size= PROCESS_VALUES['train_size'])

# #REPORT
# print("train", type(baseFormatedDf.trainDf), baseFormatedDf.trainDf.shape)
# print("validate", type(baseFormatedDf.valDf), baseFormatedDf.valDf.shape)
# print("test", type(baseFormatedDf.testDf), baseFormatedDf.testDf.shape)

#STOCK
pickleDumpMe(DB_Values['DBpath'], displayParams, baseFormatedDf, 'DATA', 'baseFormatedDf')

#QUESTIONS
# #todo : Should I scale my y values (targets)

"""
------------------------------------------------------------------------------------------------------------------------
5.FEATURE SELECTION
------------------------------------------------------------------------------------------------------------------------
"""
"""
FILTER - SPEARMAN
"""
#ABOUT
"""
GOAL - Remove uncorrelated and redundant features
Dashboard Input - PROCESS_VALUES : corrMethod, corrRounding, corrLowThreshhold, corrHighThreshhold
"""
#CONSTRUCT
spearmanFilter = FilterFeatures(baseFormatedDf, baseLabels = xQuantLabels, method =PROCESS_VALUES['corrMethod'],
        corrRounding = PROCESS_VALUES['corrRounding'], lowThreshhold = PROCESS_VALUES['corrLowThreshhold'],
                                highThreshhold = PROCESS_VALUES['corrHighThreshhold'])
# REPORT
# print('')
# print('FILTER - SPEARMAN CORRELATION')
# print('LABELS : ', spearmanFilter.trainDf.shape, type(spearmanFilter.trainDf))
# print(spearmanFilter.selectedLabels)

#STOCK
pickleDumpMe(DB_Values['DBpath'], displayParams, spearmanFilter, 'FS', 'spearmanFilter')

#VISUALIZE
plotCorrelation(spearmanFilter, spearmanFilter.correlationMatrix_All, DB_Values['DBpath'], displayParams,
                filteringName="nofilter")
plotCorrelation(spearmanFilter, spearmanFilter.correlationMatrix_NoUncorrelated, DB_Values['DBpath'], displayParams,
                filteringName="dropuncorr")
plotCorrelation(spearmanFilter, spearmanFilter.correlationMatrix_NoRedundant, DB_Values['DBpath'], displayParams,
                filteringName="dropcolinear")

"""
ELIMINATE - RFE
"""
"""
GOAL - select the optimal number of features or combination of features
"""


#CONSTRUCT
LR_RFE = WrapFeatures(method = 'LR', estimator = LinearRegression(), formatedDf = baseFormatedDf,
                      rfe_hyp_feature_count= RFE_VALUES['RFE_featureCount'], output_feature_count='rfeHyp')
RFR_RFE = WrapFeatures(method = 'RFR', estimator = RandomForestRegressor(random_state = PROCESS_VALUES['random_state']),
                       formatedDf = baseFormatedDf, rfe_hyp_feature_count= RFE_VALUES['RFE_featureCount'], output_feature_count='rfeHyp')
DTR_RFE = WrapFeatures(method = 'DTR', estimator = DecisionTreeRegressor(random_state = PROCESS_VALUES['random_state']),
                       formatedDf = baseFormatedDf, rfe_hyp_feature_count= RFE_VALUES['RFE_featureCount'], output_feature_count='rfeHyp')
GBR_RFE = WrapFeatures(method = 'GBR', estimator = GradientBoostingRegressor(random_state = PROCESS_VALUES['random_state']),
                       formatedDf = baseFormatedDf, rfe_hyp_feature_count= RFE_VALUES['RFE_featureCount'], output_feature_count='rfeHyp')
DTC_RFE = WrapFeatures(method = 'DTC', estimator = DecisionTreeClassifier(random_state = PROCESS_VALUES['random_state']),
                       formatedDf = baseFormatedDf, rfe_hyp_feature_count= RFE_VALUES['RFE_featureCount'], output_feature_count='rfeHyp')
RFEs = [LR_RFE, RFR_RFE, DTR_RFE, GBR_RFE, DTC_RFE]

#REPORT
# print('ELIMINATE - RECURSIVE FEATURE ELIMINATION')
# reportRFE(DB_Values['DBpath'], displayParams, RFEs, objFolder ='FS', display = True)
# for _RFE in RFEs:
#     _RFE.RFECVDisplay()
#     _RFE.RFEHypSearchDisplay()
#     _RFE.RFEDisplay()
#
reportProcessing(DB_Values['DBpath'], displayParams, df, learningDf, baseFormatedDf, spearmanFilter, RFEs)
reportRFE(DB_Values['DBpath'], displayParams, RFEs, objFolder ='FS', display = True)
[LR_RFE, RFR_RFE, DTR_RFE, GBR_RFE, DTC_RFE] = RFEs

#STOCK
pickleDumpMe(DB_Values['DBpath'], displayParams, RFEs, 'FS', 'RFEs')

#VISUALIZE
RFEHyperparameterPlot2D(RFEs,  displayParams, DBpath = DB_Values['DBpath'], yLim = None, figTitle = 'RFEPlot2d',
                          title ='Influence of Feature Count on Model Performance', xlabel='Feature Count', log = False)

RFEHyperparameterPlot3D(RFEs, displayParams, DBpath = DB_Values['DBpath'], figTitle='RFEPlot3d',
                            colorsPtsLsBest=['b', 'g', 'c', 'y'],
                            title='Influence of Feature Count on Model Performance', ylabel='Feature Count',
                            zlabel='R2 Test score', size=[6, 6],
                            showgrid=False, log=False, max=True, ticks=False, lims=False)

#QUESTIONS
#todo : how to evaluate RFE - the goal is not to perform the best prediction - what scoring should be inserted?
#todo : understand fit vs fit transform > make sure i am working with updated data

# EXPORT

# df = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/DATA/df.pkl', show = False)
# learningDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/DATA/learningDf.pkl', show = False)
# baseFormatedDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/DATA/baseFormatedDf.pkl', show = False)
# spearmanFilter = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/FS/spearmanFilter.pkl', show = False)
# RFEs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/FS/RFEs.pkl', show = False)



