
from dashBoard import *
from report import *
from Raw import *
from Features import *
from Data import *
from Format import *
from Filter import *
from FilterVisualizer import *
from Wrapper import *
from WrapperVisualizer import *
# from Models import *
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

# #todo : Should I scale my y values (targets)

baseFormatedDf = formatedDf(learningDf, xQuantLabels, xQualLabels, yLabels,
                            yUnitFactor = FORMAT_Values['yUnitFactor'], targetLabels= FORMAT_Values['targetLabels'],
                            random_state = PROCESS_VALUES['random_state'], test_size= PROCESS_VALUES['test_size'],
                            train_size= PROCESS_VALUES['train_size'])

LearningDfs = [baseFormatedDf]

print("train", type(baseFormatedDf.trainDf), baseFormatedDf.trainDf.shape)
print("validate", type(baseFormatedDf.valDf), baseFormatedDf.valDf.shape)
print("test", type(baseFormatedDf.testDf), baseFormatedDf.testDf.shape)

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

spearmanFilter = FilterFeatures(baseFormatedDf, baseLabels = xQuantLabels, method =PROCESS_VALUES['corrMethod'],
        corrRounding = PROCESS_VALUES['corrRounding'], lowThreshhold = PROCESS_VALUES['corrLowThreshhold'],
                                highThreshhold = PROCESS_VALUES['corrHighThreshhold'])


LearningDfs += [spearmanFilter]

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

"""
ELIMINATE - RFE
"""

"""select the optimal number of features or combination of features"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor



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

print('ELIMINATE - RECURSIVE FEATURE ELIMINATION')
RFEs = [LR_RFE, RFR_RFE, DTR_RFE, GBR_RFE, DTC_RFE]
for _RFE in RFEs:
    _RFE.RFECVDisplay()
    _RFE.RFEHypSearchDisplay()
    _RFE.RFEDisplay()

LearningDfs += RFEs

pickleDumpMe(DB_Values['DBpath'], displayParams, LearningDfs, 'LearningDfs')

for df in LearningDfs:
    print('Learning dataframe : ', 'train', df.trainDf.shape, 'validate', df.testDf.shape)
    print('feature selection : ', df.selector)
    print('features kept : ', len(df.selectedLabels))
    print('features dropped : ', len(df.droppedLabels))
    print('features kept labels :', df.selectedLabels)

# RFEHyperparameterPlot2D(RFEs,  displayParams, DBpath = DB_Values['DBpath'], yLim = None, figTitle = 'RFEPlot2d',
#                           title ='Influence of Feature Count on Model Performance', xlabel='Feature Count', log = False)
#
# RFEHyperparameterPlot3D(RFEs, displayParams, DBpath = DB_Values['DBpath'], figTitle='RFEPlot3d',
#                             colorsPtsLsBest=['b', 'g', 'c', 'y'],
#                             title='Influence of Feature Count on Model Performance', ylabel='Feature Count',
#                             zlabel='R2 Test score', size=[6, 6],
#                             showgrid=False, log=False, max=True, ticks=False, lims=False)



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
#todo - work on hyperparameter plot - 1 param at time
#todo clean

"""
Hyperparam Search
"""

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

myFormatedDf = baseFormatedDf

LR_GS = ModelGridsearch('LR', learningDf= myFormatedDf, modelPredictor= LinearRegression(), param_dict=dict())

# todo : fix these
# print(LR_GS)
# LR_LASSO_GS = ModelGridsearch('LR_LASSO', learningDf= myFormatedDf, modelPredictor= Lasso(tol=1e-2), param_dict = LR_param_grid)
# print('LR_LASSO_GS', LR_LASSO_GS)
# LR_RIDGE_GS = ModelGridsearch('LR_RIDGE', learningDf= myFormatedDf, modelPredictor= Ridge(), param_dict = LR_param_grid)
# print('LR_RIDGE_GS', LR_RIDGE_GS)
# LR_ELAST_GS = ModelGridsearch('LR_ELAST', learningDf= myFormatedDf, modelPredictor= ElasticNet(tol=1e-2), param_dict = LR_param_grid)
# print('LR_ELAST_GS', LR_ELAST_GS)

KRR_GS = ModelGridsearch('KRR', learningDf= myFormatedDf, modelPredictor= KernelRidge(), param_dict = KRR_param_grid)
SVR_GS = ModelGridsearch('SVR', learningDf= myFormatedDf, modelPredictor= SVR(), param_dict = SVR_param_grid)

GSs = [LR_GS, KRR_GS, SVR_GS]

pickleDumpMe(DB_Values['DBpath'], displayParams, GSs, 'GSs')

# for GS in GSs:#,LR_LASSO_GS, LR_RIDGE_GS, LR_ELAST_GS
#     plotPredTruth(displayParams = displayParams, modelGridsearch = GS,
#                   DBpath = DB_Values['DBpath'], TargetMinMaxVal = FORMAT_Values['TargetMinMaxVal'], fontsize = 14)
#     paramResiduals(modelGridsearch = GS, displayParams = displayParams,
#                           DBpath = DB_Values['DBpath'], yLim = PROCESS_VALUES['residualsYLim'] , xLim = PROCESS_VALUES['residualsXLim'])
#     plotResiduals(modelGridsearch = GS, displayParams = displayParams,  DBpath = DB_Values['DBpath'],
#                 bins=20, binrange = [-200, 200])

# predTruthCombined(displayParams, GSs, DBpath = DB_Values['DBpath'], scatter=True, fontsize=14) #scatter=False for groundtruth as line

"""
Hyperparam Search
"""

KRR_GS1 = ModelGridsearch(predictorName='KRR_lin', modelPredictor= KernelRidge(), param_dict = KRR_param_grid1, learningDf= myFormatedDf)
KRR_GS2 = ModelGridsearch(predictorName='KRR_poly', modelPredictor= KernelRidge(), param_dict = KRR_param_grid2, learningDf= myFormatedDf)
KRR_GS3 = ModelGridsearch(predictorName='KRR_rbf', modelPredictor= KernelRidge(), param_dict = KRR_param_grid3, learningDf= myFormatedDf)

KRR_GS = [KRR_GS1, KRR_GS2, KRR_GS3]

pickleDumpMe(DB_Values['DBpath'], displayParams, KRR_GS, 'KRR_GS')
#
# GSParameterPlot2D(KRRR,  displayParams, DBpath = DB_Values['DBpath'],  yLim = None, paramKey ='gamma', score ='mean_test_r2', log = True)
# GSParameterPlot3D(KRRR, displayParams, DBpath = DB_Values['DBpath'],
#                       colorsPtsLsBest=['b', 'g', 'c', 'y'], paramKey='gamma', score='mean_test_r2',
#                       size=[6, 6], showgrid=False, log=True, maxScore=True, absVal = False,  ticks=False, lims=False)


#
# #todo : check summary equation table
#

# pickleLoadMe(path, show = False)