#SCRIPT IMPORTS
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
from ModelsGridsearch import *
from ModelPredTruthPt import *
from ModelResidualsPt import *
from ModelParamPt import *
from ModelMetricsPt import *
from ModelWeights import *
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
#
# #ABOUT
# """
# GOAL - Import libraries & Load data
# """
#
# #CONSTRUCT
# rdat = RawData(path = DB_Values['DBpath'], dbName = DB_Values['DBname'], delimiter = DB_Values['DBdelimiter'],
#                firstLine = DB_Values['DBfirstLine'], updateLabels = None)
#
# #VISUALIZE
# # rdat.visualize(displayParams, DBpath = DB_Values['DBpath'], dbName = DB_Values['DBname'],
# #               yLabel = yLabels[0], xLabel=xQualLabels[0])
# # rdat.visualize(displayParams, DBpath = DB_Values['DBpath'], dbName = DB_Values['DBname'],
# #             yLabel = yLabels[0], xLabel=xQuantLabels[0])
#
# DBpath = DB_Values['DBpath']
# """
# ------------------------------------------------------------------------------------------------------------------------
# 2.FEATURES
# ------------------------------------------------------------------------------------------------------------------------
# """
# #ABOUT
# """
# GOAL - Process data & One hot encoding
# """
#
# #CONSTRUCT
# dat = Features(rdat)
# df = dat.asDataframe()
#
# #REPORT
# print("Full df", df.shape)
#
# #STOCK
# pickleDumpMe(DB_Values['DBpath'], displayParams, df, 'DATA', 'df')

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
# #CONSTRUCT
# learningDf = removeOutliers(df, labels = xQuantLabels + yLabels, cutOffThreshhold=PROCESS_VALUES['OutlierCutOffThreshhold'])
#
# #REPORT
# print("Outliers removed ", learningDf.shape)
#
# #STOCK
# pickleDumpMe(DB_Values['DBpath'], displayParams, learningDf, 'DATA', 'learningDf')

#
# """
# ------------------------------------------------------------------------------------------------------------------------
# 4. FORMAT
# ------------------------------------------------------------------------------------------------------------------------
# """
# #ABOUT
# """
# GOAL - Train Validate Test Split - Scale
# Dashboard Input - PROCESS_VALUES : test_size  # proportion with validation, random_state, yUnit
# """
#
#
# #CONSTRUCT
# baseFormatedDf = formatedDf(learningDf, xQuantLabels, xQualLabels, yLabels,
#                             yUnitFactor = FORMAT_Values['yUnitFactor'], targetLabels= FORMAT_Values['targetLabels'],
#                             random_state = PROCESS_VALUES['random_state'], test_size= PROCESS_VALUES['test_size'],
#                             train_size= PROCESS_VALUES['train_size'])
#
# #REPORT
# print("train", type(baseFormatedDf.trainDf), baseFormatedDf.trainDf.shape)
# print("validate", type(baseFormatedDf.valDf), baseFormatedDf.valDf.shape)
# print("test", type(baseFormatedDf.testDf), baseFormatedDf.testDf.shape)
#
# #STOCK
# pickleDumpMe(DB_Values['DBpath'], displayParams, baseFormatedDf, 'DATA', 'baseFormatedDf')
#
# #QUESTIONS
# # #todo : Should I scale my y values (targets)
#
# """
# ------------------------------------------------------------------------------------------------------------------------
# 4.FEATURE SELECTION
# ------------------------------------------------------------------------------------------------------------------------
# """
# """
# FILTER - SPEARMAN
# """
# #ABOUT
# """
# GOAL - Remove uncorrelated and redundant features
# Dashboard Input - PROCESS_VALUES : corrMethod, corrRounding, corrLowThreshhold, corrHighThreshhold
# """
# #CONSTRUCT
# spearmanFilter = FilterFeatures(baseFormatedDf, baseLabels = xQuantLabels, method =PROCESS_VALUES['corrMethod'],
#         corrRounding = PROCESS_VALUES['corrRounding'], lowThreshhold = PROCESS_VALUES['corrLowThreshhold'],
#                                 highThreshhold = PROCESS_VALUES['corrHighThreshhold'])
# #REPORT
# # print('')
# # print('FILTER - SPEARMAN CORRELATION')
# # print('LABELS : ', spearmanFilter.trainDf.shape, type(spearmanFilter.trainDf))
# # print(spearmanFilter.selectedLabels)
#
# #STOCK
# pickleDumpMe(DB_Values['DBpath'], displayParams, spearmanFilter, 'FILTER', 'spearmanFilter')
#
# #VISUALIZE
# # plotCorrelation(spearmanFilter, spearmanFilter.correlationMatrix_All, DB_Values['DBpath'], displayParams,
# #                 filteringName="nofilter")
# # plotCorrelation(spearmanFilter, spearmanFilter.correlationMatrix_NoUncorrelated, DB_Values['DBpath'], displayParams,
# #                 filteringName="dropuncorr")
# # plotCorrelation(spearmanFilter, spearmanFilter.correlationMatrix_NoRedundant, DB_Values['DBpath'], displayParams,
# #                 filteringName="dropcolinear")
#
# """
# ELIMINATE - RFE
# """
# """
# GOAL - select the optimal number of features or combination of features
# """
#
# #CONSTRUCT
# LR_RFE = WrapFeatures(method = 'LR', estimator = LinearRegression(), formatedDf = baseFormatedDf,
#                       rfe_hyp_feature_count= RFE_VALUES['RFE_featureCount'], output_feature_count='rfeHyp')
# RFR_RFE = WrapFeatures(method = 'RFR', estimator = RandomForestRegressor(random_state = PROCESS_VALUES['random_state']),
#                        formatedDf = baseFormatedDf, rfe_hyp_feature_count= RFE_VALUES['RFE_featureCount'], output_feature_count='rfeHyp')
# DTR_RFE = WrapFeatures(method = 'DTR', estimator = DecisionTreeRegressor(random_state = PROCESS_VALUES['random_state']),
#                        formatedDf = baseFormatedDf, rfe_hyp_feature_count= RFE_VALUES['RFE_featureCount'], output_feature_count='rfeHyp')
# GBR_RFE = WrapFeatures(method = 'GBR', estimator = GradientBoostingRegressor(random_state = PROCESS_VALUES['random_state']),
#                        formatedDf = baseFormatedDf, rfe_hyp_feature_count= RFE_VALUES['RFE_featureCount'], output_feature_count='rfeHyp')
# DTC_RFE = WrapFeatures(method = 'DTC', estimator = DecisionTreeClassifier(random_state = PROCESS_VALUES['random_state']),
#                        formatedDf = baseFormatedDf, rfe_hyp_feature_count= RFE_VALUES['RFE_featureCount'], output_feature_count='rfeHyp')
# RFEs = [LR_RFE, RFR_RFE, DTR_RFE, GBR_RFE, DTC_RFE]
#
# #REPORT
# print('ELIMINATE - RECURSIVE FEATURE ELIMINATION')
# for _RFE in RFEs:
#     _RFE.RFECVDisplay()
#     _RFE.RFEHypSearchDisplay()
#     _RFE.RFEDisplay()
#
# #STOCK
# pickleDumpMe(DB_Values['DBpath'], displayParams, RFEs, 'WRAPPER', 'RFEs')

#IMPORT
df = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/DATA/df.pkl', show = False)
learningDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/DATA/learningDf.pkl', show = False)
baseFormatedDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/DATA/baseFormatedDf.pkl', show = True)
spearmanFilter = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/FILTER/spearmanFilter.pkl', show = True)
RFEs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/WRAPPER/RFEs.pkl', show = True)

# REPORT
reportFeatureSelection(DB_Values['DBpath'], displayParams, df, learningDf, baseFormatedDf, spearmanFilter, RFEs)
[LR_RFE, RFR_RFE, DTR_RFE, GBR_RFE, DTC_RFE] = RFEs

#
# #VISUALIZE
# # RFEHyperparameterPlot2D(RFEs,  displayParams, DBpath = DB_Values['DBpath'], yLim = None, figTitle = 'RFEPlot2d',
# #                           title ='Influence of Feature Count on Model Performance', xlabel='Feature Count', log = False)
# #
# # RFEHyperparameterPlot3D(RFEs, displayParams, DBpath = DB_Values['DBpath'], figTitle='RFEPlot3d',
# #                             colorsPtsLsBest=['b', 'g', 'c', 'y'],
# #                             title='Influence of Feature Count on Model Performance', ylabel='Feature Count',
# #                             zlabel='R2 Test score', size=[6, 6],
# #                             showgrid=False, log=False, max=True, ticks=False, lims=False)
#
# #QUESTIONS
# #todo : how to evaluate RFE - the goal is not to perform the best prediction - what scoring should be inserted?
# #todo : understand fit vs fit transform > make sure i am working with updated data
#
# """
# ------------------------------------------------------------------------------------------------------------------------
# 4.HYPERPARAMETER SEARCH
# ------------------------------------------------------------------------------------------------------------------------
# """
#
# """
# MODELS
# """
#
# #ABOUT
# """
# GOAL -  Find best hyperparameters for each models
# Dashboard Input - _VALUES : xx
# """
# #CONSTRUCT
# myFormatedDf = baseFormatedDf
# #
# LR_GS = ModelGridsearch('LR', learningDf= myFormatedDf, modelPredictor= LinearRegression(), param_dict=dict())
# LR_LASSO_GS = ModelGridsearch('LR_LASSO', learningDf= baseFormatedDf, modelPredictor= Lasso(), param_dict = LR_param_grid)
# LR_RIDGE_GS = ModelGridsearch('LR_RIDGE', learningDf= baseFormatedDf, modelPredictor= Ridge(), param_dict = LR_param_grid)
# LR_ELAST_GS = ModelGridsearch('LR_ELAST', learningDf= baseFormatedDf, modelPredictor= ElasticNet(), param_dict = LR_param_grid)
# KRR_GS = ModelGridsearch('KRR', learningDf= myFormatedDf, modelPredictor= KernelRidge(), param_dict = KRR_param_grid)
# SVR_GS = ModelGridsearch('SVR', learningDf= myFormatedDf, modelPredictor= SVR(), param_dict = SVR_param_grid)
#
# GSs = [LR_GS, KRR_GS, SVR_GS]
#
# #STOCK
# pickleDumpMe(DB_Values['DBpath'], displayParams, GSs, 'GS', 'GSs')
# saveStudy(DB_Values['DBpath'], displayParams, obj= baseFormatedDf)

#IMPORT
GSs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/GS/GSs.pkl', show = True)

#REPORT
reportModels(DB_Values['DBpath'], displayParams, GSs, baseFormatedDf, objFolder ='Models', display = True)

# #VISUALIZE
# for GS in GSs:#,LR_LASSO_GS, LR_RIDGE_GS, LR_ELAST_GS
#     plotPredTruth(displayParams = displayParams, modelGridsearch = GS,
#                   DBpath = DB_Values['DBpath'], TargetMinMaxVal = FORMAT_Values['TargetMinMaxVal'], fontsize = 14)
#     paramResiduals(modelGridsearch = GS, displayParams = displayParams,
#                           DBpath = DB_Values['DBpath'], yLim = PROCESS_VALUES['residualsYLim'] , xLim = PROCESS_VALUES['residualsXLim'])
#     plotResiduals(modelGridsearch = GS, displayParams = displayParams,  DBpath = DB_Values['DBpath'],
#                 bins=20, binrange = [-200, 200])
# predTruthCombined(displayParams, GSs, DBpath = DB_Values['DBpath'], scatter=True, fontsize=14) #scatter=False for groundtruth as line
#
# MetricsSummaryPlot(GSs, displayParams, DBpath  = DB_Values['DBpath'], metricLabels=['TrainScore', 'TestScore', 'TestAcc', 'TestMSE'],
#                        title='Model Evaluations', ylabel='Evaluation Metric', fontsize=14, scatter=True)
#
# sortedGSs = sortGridResults(GSs, metric ='TestAcc', highest = True)
#
# WeightsBarplotAll(GSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'],
#                   df=None, yLim = None, sorted = True, key = 'WeightsScaled' )
#
# WeightsSummaryPlot(GSs, displayParams, DB_Values['DBpath'], sorted=True, yLim=None, fontsize=14)


# """
# HYPERPARAMETERS
# """
# #ABOUT
# """
# GOAL -  Find the influence of 1 hyperparameters on models
# Dashboard Input - _VALUES : xx
# """
# #CONSTRUCT
# KRR_GS1 = ModelGridsearch(predictorName='KRR_lin', modelPredictor= KernelRidge(), param_dict = KRR_param_grid1, learningDf= myFormatedDf)
# KRR_GS2 = ModelGridsearch(predictorName='KRR_poly', modelPredictor= KernelRidge(), param_dict = KRR_param_grid2, learningDf= myFormatedDf)
# KRR_GS3 = ModelGridsearch(predictorName='KRR_rbf', modelPredictor= KernelRidge(), param_dict = KRR_param_grid3, learningDf= myFormatedDf)
# KRR_GS = [KRR_GS1, KRR_GS2, KRR_GS3]
#
# #VISUALIZE
# # GSParameterPlot2D(KRRR,  displayParams, DBpath = DB_Values['DBpath'],  yLim = None, paramKey ='gamma', score ='mean_test_r2', log = True)
# # GSParameterPlot3D(KRRR, displayParams, DBpath = DB_Values['DBpath'],
# #                       colorsPtsLsBest=['b', 'g', 'c', 'y'], paramKey='gamma', score='mean_test_r2',
# #                       size=[6, 6], showgrid=False, log=True, maxScore=True, absVal = False,  ticks=False, lims=False)
#
# #REPORT
# # for df in LearningDfs:
# #     print('Learning dataframe : ', 'train', df.trainDf.shape, 'validate', df.testDf.shape)
# #     print('feature selection : ', df.selector)
# #     print('features kept : ', len(df.selectedLabels))
# #     print('features dropped : ', len(df.droppedLabels))
# #     print('features kept labels :', df.selectedLabels)
#
# #STOCK
# pickleDumpMe(DB_Values['DBpath'], displayParams, KRR_GS, 'GS', 'KRR_GS_gamma')



# """
# MODEL x FEATURE SELECTION GRIDSEARCH
# """
# #ABOUT
# """
# GOAL -  Calibrate model hyperparameters for different learning Dfs
# Dashboard Input - _VALUES : xx
# """

# #IMPORT
# learning_dfs = [baseFormatedDf, spearmanFilter] + RFEs

# #CONSTRUCT
LR_ = {'name' : 'LR',  'modelPredictor' : LinearRegression(),'param_dict' : dict()}
LR_RIDGE = {'name' : 'LR_RIDGE',  'modelPredictor' : Lasso(),'param_dict' : LR_param_grid}
LR_LASSO = {'name' : 'LR_LASSO',  'modelPredictor' : Ridge(),'param_dict' : LR_param_grid}
LR_ELAST = {'name' : 'LR_ELAST',  'modelPredictor' : ElasticNet(),'param_dict' : LR_param_grid}
KRR = {'name' : 'KRR',  'modelPredictor' : KernelRidge(),'param_dict' : KRR_param_grid}
SVR = {'name' : 'SVR',  'modelPredictor' : SVR(),'param_dict' : SVR_param_grid}


LR_FS_GS = ModelFeatureSelectionGridsearch(predictorName=LR_['name'], learningDfs=learning_dfs,
                                        modelPredictor=LR_['modelPredictor'], param_dict=LR_['param_dict'])
LR_RIDGE_FS_GS = ModelFeatureSelectionGridsearch(predictorName=LR_RIDGE['name'], learningDfs=learning_dfs,
                                        modelPredictor=LR_RIDGE['modelPredictor'], param_dict=LR_RIDGE['param_dict'])
LR_LASSO_FS_GS = ModelFeatureSelectionGridsearch(predictorName=LR_LASSO['name'], learningDfs=learning_dfs,
                                        modelPredictor=LR_LASSO['modelPredictor'], param_dict=LR_LASSO['param_dict'])
LR_ELAST_FS_GS = ModelFeatureSelectionGridsearch(predictorName=LR_ELAST['name'], learningDfs=learning_dfs,
                                        modelPredictor=LR_ELAST['modelPredictor'], param_dict=LR_ELAST['param_dict'])
KRR_FS_GS = ModelFeatureSelectionGridsearch(predictorName=KRR['name'], learningDfs=learning_dfs,
                                        modelPredictor=KRR['modelPredictor'], param_dict=KRR['param_dict'])
SVR_FS_GS = ModelFeatureSelectionGridsearch(predictorName=SVR['name'], learningDfs=learning_dfs,
                                        modelPredictor=SVR['modelPredictor'], param_dict=SVR['param_dict'])

FS_GS = [LR_FS_GS, LR_RIDGE_FS_GS, LR_LASSO_FS_GS, LR_ELAST_FS_GS, KRR_FS_GS, SVR_FS_GS]


#todo : split this script into smaller ones/ study
#add visuals for FS_GS
#REPORT
for gs in FS_GS:
    #todo : report individually
    reportGridsearch(DB_Values['DBpath'], displayParams, FS_GS, objFolder ='REPORT', display = True)


#todo : make a grid out of results
#todo : check residuals - do i need to replicate/CV results for more data


# todo : display results a a grid
# todo : fix issue results vary strongly depending on if from Model GS or from ModelFeature selection GS
#  - this might be due to the differing cv in both GS > more data  > more reproduciblle?

# #todo : check summary equation table

#todo : fix LASSO, ELASTICNET, RIDGE
#todo ERROR MESSAGEs:
#todo : ak about convergence - what does this mean
"""ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. 
Duality gap: 13643.606562469611, tolerance: 29.734474418604655
  model = cd_fast.enet_coordinate_descent("""
"""LinAlgWarning: Ill-conditioned matrix (rcond=3.113e-18): result may not be accurate."""
"""UserWarning: Singular matrix in solving dual problem. Using least-squares solution instead."""


#IMPORTS
# df = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/DATA/df.pkl', show = False)
# learningDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/DATA/learningDf.pkl', show = False)
# baseFormatedDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/DATA/baseFormatedDf.pkl', show = True)
# spearmanFilter = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/FILTER/spearmanFilter.pkl', show = True)
# RFEs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/WRAPPER/RFEs.pkl', show = True)
# GSs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/GS/GSs.pkl', show = True)
# KRR_GS_gamma = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/GS/KRR_GS_gamma.pkl', show = True)
#
