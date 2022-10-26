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
from GridsearchWeights import *
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

#IMPORT Mainf_FS

df = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/DATA/df.pkl', show = False)
learningDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/DATA/learningDf.pkl', show = False)
baseFormatedDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/DATA/baseFormatedDf.pkl', show = False)
spearmanFilter = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/FS/spearmanFilter.pkl', show = False)
RFEs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/FS/RFEs.pkl', show = False)

#IMPORT Mainf_GS

GSs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS/GSs.pkl', show = False)
KRR_GS_gamma = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS/KRR_GS_gamma.pkl', show = False)


# ------------------------------------------------------------------------------------------------------------------------
# 7.MODEL x FEATURE SELECTION GRIDSEARCH
# ------------------------------------------------------------------------------------------------------------------------
# """

# """
# MODEL x FEATURE SELECTION GRIDSEARCH
# """
# #ABOUT
# """
# GOAL -  Calibrate model hyperparameters for different learning Dfs
# Dashboard Input - GS_VALUES ; _param_grids
# """

#IMPORT
learning_dfs = [ spearmanFilter] + RFEs #baseFormatedDf,

# #CONSTRUCT
LR_ = {'name' : 'LR',  'modelPredictor' : LinearRegression(),'param_dict' : dict()}
LR_RIDGE = {'name' : 'LR_RIDGE',  'modelPredictor' : Lasso(),'param_dict' : LR_param_grid}
LR_LASSO = {'name' : 'LR_LASSO',  'modelPredictor' : Ridge(),'param_dict' : LR_param_grid}
LR_ELAST = {'name' : 'LR_ELAST',  'modelPredictor' : ElasticNet(),'param_dict' : LR_param_grid}
KRR = {'name' : 'KRR',  'modelPredictor' : KernelRidge(),'param_dict' : KRR_param_grid}
SVR = {'name' : 'SVR',  'modelPredictor' : SVR(),'param_dict' : SVR_param_grid}

#CONSTRUCT & REPORT
# LR_FS_GS = ModelFeatureSelectionGridsearch(predictorName=LR_['name'], learningDfs=learning_dfs,
#                                         modelPredictor=LR_['modelPredictor'], param_dict=LR_['param_dict'])
# reportGridsearch(DB_Values['DBpath'], displayParams, LR_FS_GS, objFolder ='GS_FS', display = True)
# pickleDumpMe(DB_Values['DBpath'], displayParams, LR_FS_GS, 'GS_FS', 'LR_FS_GS')
#
# LR_RIDGE_FS_GS = ModelFeatureSelectionGridsearch(predictorName=LR_RIDGE['name'], learningDfs=learning_dfs,
#                                         modelPredictor=LR_RIDGE['modelPredictor'], param_dict=LR_RIDGE['param_dict'])
# reportGridsearch(DB_Values['DBpath'], displayParams, LR_RIDGE_FS_GS, objFolder ='GS_FS', display = True)
# pickleDumpMe(DB_Values['DBpath'], displayParams, LR_RIDGE_FS_GS, 'GS_FS', 'LR_RIDGE_FS_GS')
#
# LR_LASSO_FS_GS = ModelFeatureSelectionGridsearch(predictorName=LR_LASSO['name'], learningDfs=learning_dfs,
#                                         modelPredictor=LR_LASSO['modelPredictor'], param_dict=LR_LASSO['param_dict'])
# reportGridsearch(DB_Values['DBpath'], displayParams, LR_LASSO_FS_GS, objFolder ='GS_FS', display = True)
# pickleDumpMe(DB_Values['DBpath'], displayParams, LR_LASSO_FS_GS, 'GS_FS', 'LR_LASSO_FS_GS')
#
# LR_ELAST_FS_GS = ModelFeatureSelectionGridsearch(predictorName=LR_ELAST['name'], learningDfs=learning_dfs,
#                                         modelPredictor=LR_ELAST['modelPredictor'], param_dict=LR_ELAST['param_dict'])
# reportGridsearch(DB_Values['DBpath'], displayParams, LR_ELAST_FS_GS, objFolder ='GS_FS', display = True)
# pickleDumpMe(DB_Values['DBpath'], displayParams, LR_ELAST_FS_GS, 'GS_FS', 'LR_ELAST_FS_GS')
#
# KRR_FS_GS = ModelFeatureSelectionGridsearch(predictorName=KRR['name'], learningDfs=learning_dfs,
#                                         modelPredictor=KRR['modelPredictor'], param_dict=KRR['param_dict'])
# reportGridsearch(DB_Values['DBpath'], displayParams, KRR_FS_GS, objFolder ='GS_FS', display = True)
# pickleDumpMe(DB_Values['DBpath'], displayParams, KRR_FS_GS, 'GS_FS', 'KRR_FS_GS')


#todo : change the learningdf for svr below

# SVR_FS_GS = ModelFeatureSelectionGridsearch(predictorName=SVR['name'], learningDfs=[spearmanFilter],
#                                         modelPredictor=SVR['modelPredictor'], param_dict=SVR['param_dict'])
# reportGridsearch(DB_Values['DBpath'], displayParams, SVR_FS_GS, objFolder ='GS_FS', display = True)
# pickleDumpMe(DB_Values['DBpath'], displayParams, SVR_FS_GS, 'GS_FS', 'SVR_FS_GS')


#IMPORT
LR_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/LR_FS_GS.pkl', show = False)
LR_RIDGE_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/LR_RIDGE_FS_GS.pkl', show = False)
LR_LASSO_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/LR_LASSO_FS_GS.pkl', show = True)
LR_ELAST_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/LR_ELAST_FS_GS.pkl', show = True)
KRR_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/KRR_FS_GS.pkl', show = True)
SVR_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/SVR_FS_GS.pkl', show = True)

# LR_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/GS/LR_FS_GS.pkl', show = False)
# LR_RIDGE_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/GS/LR_RIDGE_FS_GS.pkl', show = False)
# LR_LASSO_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/GS/LR_LASSO_FS_GS.pkl', show = True)
# LR_ELAST_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/GS/LR_ELAST_FS_GS.pkl', show = True)
# KRR_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/GS/KRR_FS_GS.pkl', show = True)
# SVR_FS_GS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/GS/SVR_FS_GS.pkl', show = True)

GS_FSs = [LR_FS_GS, LR_RIDGE_FS_GS, LR_LASSO_FS_GS, LR_ELAST_FS_GS, KRR_FS_GS, SVR_FS_GS] #

#VISUALIZE

# #INDIVIDUAL


# for GS_FS in GS_FSs:#,LR_LASSO_FS_GS, LR_RIDGE_FS_GS, LR_ELAST_FS_GS
#     for learningDflabel in GS_FS.learningDfsList:
#         GS = GS_FS.__getattribute__(learningDflabel)
#         print(GS.predictorName)
#         print(GS.selectorName)
#         plotPredTruth(displayParams = displayParams, modelGridsearch = GS,
#                       DBpath = DB_Values['DBpath'], TargetMinMaxVal = FORMAT_Values['TargetMinMaxVal'], fontsize = 14,
#                       studyFolder = 'GS_FS/')
#
#         plotResiduals(modelGridsearch = GS, displayParams = displayParams,  DBpath = DB_Values['DBpath'],
#                     bins=20, binrange = [-200, 200], studyFolder = 'GS_FS/')
#
#         paramResiduals(modelGridsearch = GS, displayParams = displayParams,
#                               DBpath = DB_Values['DBpath'], yLim = PROCESS_VALUES['residualsYLim'] ,
#                        xLim = PROCESS_VALUES['residualsXLim'], studyFolder = 'GS_FS/')
#
# #GROUPED
# GS_predTruthCombined(displayParams, GS_FSs, DBpath = DB_Values['DBpath'], content = 'GS_FSs', scatter=True, fontsize=14, studyFolder = 'GS_FS/') #scatter=False for groundtruth as line
#
# GS_predTruthCombined(displayParams, [LR_FS_GS], DBpath = DB_Values['DBpath'], content = 'LR_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/') #scatter=False for groundtruth as line
# GS_predTruthCombined(displayParams, [LR_RIDGE_FS_GS], DBpath = DB_Values['DBpath'], content = 'LR_RIDGE_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/') #scatter=False for groundtruth as line
# GS_predTruthCombined(displayParams, [LR_LASSO_FS_GS], DBpath = DB_Values['DBpath'], content = 'LR_LASSO_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/') #scatter=False for groundtruth as line
# GS_predTruthCombined(displayParams, [LR_ELAST_FS_GS], DBpath = DB_Values['DBpath'], content = 'LR_ELAST_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/') #scatter=False for groundtruth as line
# GS_predTruthCombined(displayParams, [KRR_FS_GS], DBpath = DB_Values['DBpath'], content = 'KRR_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/') #scatter=False for groundtruth as line
# GS_predTruthCombined(displayParams, [SVR_FS_GS], DBpath = DB_Values['DBpath'], content = 'SVR_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/') #scatter=False for groundtruth as line
#
# #
#

# GS_MetricsSummaryPlot(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FSs', scatter=True, studyFolder = 'GS_FS/')
#
# GS_MetricsSummaryPlot([LR_FS_GS], displayParams, DB_Values['DBpath'], content = 'LR_FS_GS', scatter=True, studyFolder = 'GS_FS/')
# GS_MetricsSummaryPlot([LR_RIDGE_FS_GS], displayParams, DB_Values['DBpath'], content = 'LR_RIDGE_FS_GS', scatter=True, studyFolder = 'GS_FS/')
# GS_MetricsSummaryPlot([LR_LASSO_FS_GS], displayParams, DB_Values['DBpath'], content = 'LR_LASSO_FS_GS', scatter=True, studyFolder = 'GS_FS/')
# GS_MetricsSummaryPlot([LR_ELAST_FS_GS], displayParams, DB_Values['DBpath'], content = 'LR_ELAST_FS_GS', scatter=True, studyFolder = 'GS_FS/')
# GS_MetricsSummaryPlot([KRR_FS_GS], displayParams, DB_Values['DBpath'], content = 'KRR_FS_GS', scatter=True, studyFolder = 'GS_FS/')
# GS_MetricsSummaryPlot([SVR_FS_GS], displayParams, DB_Values['DBpath'], content = 'SVR_FS_GS', scatter=True, studyFolder = 'GS_FS/')


#ONLY FOR identical weights
# GS_WeightsBarplotAll([LR_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'LR_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
# GS_WeightsBarplotAll([LR_RIDGE_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'LR_RIDGE_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
# GS_WeightsBarplotAll([LR_LASSO_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'LR_LASSO_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
# GS_WeightsBarplotAll([LR_ELAST_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'LR_ELAST_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
# GS_WeightsBarplotAll([KRR_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'KRR_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
# GS_WeightsBarplotAll([SVR_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'SVR_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')

# GS_WeightsSummaryPlot([LR_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='LR_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')
#
# GS_WeightsSummaryPlot([LR_RIDGE_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='LR_RIDGE_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')
#
# GS_WeightsSummaryPlot([LR_LASSO_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='LR_LASSO_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')
#
# GS_WeightsSummaryPlot([LR_ELAST_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='LR_ELAST_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')
#
# GS_WeightsSummaryPlot([KRR_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='KRR_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')
#
# GS_WeightsSummaryPlot([SVR_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='SVR_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')

GS_WeightsSummaryPlot(GS_FSs, GS_FSs, target = FORMAT_Values['targetLabels'],
                      df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
                      DBpath = DB_Values['DBpath'], content='GS_FSs', sorted=True, yLim=4,
                          fontsize=14,  studyFolder='GS_FS/')



#todo : convert to functions > for a, b in zip >
#todo : make a grid out of results
#todo : check residuals - do i need to replicate/CV results for more data
# todod : scaled but reaches 4??

# todo : display results a a grid
# todo : fix issue results vary strongly depending on if from Model GS or from ModelFeature selection GS
#  - this might be due to the differing cv in both GS > more data  > more reproduciblle?

# #todo : check summary equation table

#todo :find a way of estimating time for run - is this long?

#todo : fix LASSO, ELASTICNET, RIDGE, SVR
#todo ERROR MESSAGEs:
#todo : ak about convergence - what does this mean
#LR_LASSO,LR_ELAST
"""ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. 
Duality gap: 13643.606562469611, tolerance: 29.734474418604655
  model = cd_fast.enet_coordinate_descent("""
#KRR with Base formated df
"""LinAlgWarning: Ill-conditioned matrix (rcond=3.113e-18): result may not be accurate."""
"""UserWarning: Singular matrix in solving dual problem. Using least-squares solution instead."""

# usually caused by duplicated or highly correlated variables > exclude them
# adding a small random noise to the dataset
# dummy variable which had random numbers between (0, 1) and append it to the dataset,
