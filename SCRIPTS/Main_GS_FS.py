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
from WrapperVisualizer import *
from Model import *
from ModelPredTruthPt import *
from ModelResidualsPt import *
from ModelParamPt import *
from ModelMetricsPt import *
from ModelWeightsPt import *
from Gridsearch import *
from GridsearchPredTruthPt import *
from GridsearchWeightsPt import *
from GridsearchParamPt import *
from GridsearchReport import *


#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

#IMPORT Main_FS
df = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/DATA/df.pkl', show = False)
learningDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/DATA/learningDf.pkl', show = False)
baseFormatedDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/DATA/baseFormatedDf.pkl', show = False)
spearmanFilter = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/FS/spearmanFilter.pkl', show = False)
RFEs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/FS/RFEs.pkl', show = False)

#>>>
learning_dfs = [spearmanFilter] + RFEs #baseFormatedDf,

#IMPORT Main_GS
GSs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/GS/GSs.pkl', show = False)
KRR_GS_gamma = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/GS/KRR_GS_gamma.pkl', show = False)

#IMPORT Main_GS_FS
# GS_FSs = []
# for FS_GS_lab in ['LR_GS_FS', 'LR_RIDGE_GS_FS', 'LR_LASSO_GS_FS', 'LR_ELAST_GS_FS', 'KRR_GS_FS', 'SVR_GS_FS']:
#     path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/GS_FS/' + FS_GS_lab + '.pkl'
#     GS_FS = pickleLoadMe(path = path, show = False)
#     GS_FSs.append(GS_FS)
# #>>>
# [LR_GS_FS, LR_RIDGE_GS_FS, LR_LASSO_GS_FS, LR_ELAST_GS_FS, KRR_GS_FS, SVR_GS_FS] = GS_FSs
"""-------------------------------------------------------------------------------------------------------------------
7.MODEL x FEATURE SELECTION GRIDSEARCH
------------------------------------------------------------------------------------------------------------------------
"""

"""
MODEL x FEATURE SELECTION GRIDSEARCH
"""
#ABOUT
"""
GOAL -  Calibrate model hyperparameters for different learning Dfs
Dashboard Input - GS_VALUES ; _param_grids
"""

# #CONSTRUCT
LR_ = {'name' : 'LR',  'modelPredictor' : LinearRegression(),'param_dict' : dict()}
LR_RIDGE = {'name' : 'LR_RIDGE',  'modelPredictor' : Lasso(),'param_dict' : LR_param_grid}
LR_LASSO = {'name' : 'LR_LASSO',  'modelPredictor' : Ridge(),'param_dict' : LR_param_grid}
LR_ELAST = {'name' : 'LR_ELAST',  'modelPredictor' : ElasticNet(),'param_dict' : LR_param_grid}
KRR = {'name' : 'KRR',  'modelPredictor' : KernelRidge(),'param_dict' : KRR_param_grid}
SVR = {'name' : 'SVR',  'modelPredictor' : SVR(),'param_dict' : SVR_param_grid}
#
# # CONSTRUCT & REPORT
# LR_GS_FS = ModelFeatureSelectionGridsearch(predictorName=LR_['name'], learningDfs=learning_dfs,
#                                         modelPredictor=LR_['modelPredictor'], param_dict=LR_['param_dict'])
# reportGridsearch(DB_Values['DBpath'], displayParams, LR_GS_FS, objFolder ='GS_FS', display = True)
# pickleDumpMe(DB_Values['DBpath'], displayParams, LR_GS_FS, 'GS_FS', 'LR_GS_FS')
#
# LR_RIDGE_GS_FS = ModelFeatureSelectionGridsearch(predictorName=LR_RIDGE['name'], learningDfs=learning_dfs,
#                                         modelPredictor=LR_RIDGE['modelPredictor'], param_dict=LR_RIDGE['param_dict'])
# reportGridsearch(DB_Values['DBpath'], displayParams, LR_RIDGE_GS_FS, objFolder ='GS_FS', display = True)
# pickleDumpMe(DB_Values['DBpath'], displayParams, LR_RIDGE_GS_FS, 'GS_FS', 'LR_RIDGE_GS_FS')
#
# LR_LASSO_GS_FS = ModelFeatureSelectionGridsearch(predictorName=LR_LASSO['name'], learningDfs=learning_dfs,
#                                         modelPredictor=LR_LASSO['modelPredictor'], param_dict=LR_LASSO['param_dict'])
# reportGridsearch(DB_Values['DBpath'], displayParams, LR_LASSO_GS_FS, objFolder ='GS_FS', display = True)
# pickleDumpMe(DB_Values['DBpath'], displayParams, LR_LASSO_GS_FS, 'GS_FS', 'LR_LASSO_GS_FS')
#
# LR_ELAST_GS_FS = ModelFeatureSelectionGridsearch(predictorName=LR_ELAST['name'], learningDfs=learning_dfs,
#                                         modelPredictor=LR_ELAST['modelPredictor'], param_dict=LR_ELAST['param_dict'])
# reportGridsearch(DB_Values['DBpath'], displayParams, LR_ELAST_GS_FS, objFolder ='GS_FS', display = True)
# pickleDumpMe(DB_Values['DBpath'], displayParams, LR_ELAST_GS_FS, 'GS_FS', 'LR_ELAST_GS_FS')
#
# KRR_GS_FS = ModelFeatureSelectionGridsearch(predictorName=KRR['name'], learningDfs=learning_dfs,
#                                         modelPredictor=KRR['modelPredictor'], param_dict=KRR['param_dict'])
# reportGridsearch(DB_Values['DBpath'], displayParams, KRR_GS_FS, objFolder ='GS_FS', display = True)
# pickleDumpMe(DB_Values['DBpath'], displayParams, KRR_GS_FS, 'GS_FS', 'KRR_GS_FS')

SVR_GS_FS = ModelFeatureSelectionGridsearch(predictorName=SVR['name'], learningDfs=learning_dfs,
                                        modelPredictor=SVR['modelPredictor'], param_dict=SVR['param_dict']) #todo : check
reportGridsearch(DB_Values['DBpath'], displayParams, SVR_GS_FS, objFolder ='GS_FS', display = True)
pickleDumpMe(DB_Values['DBpath'], displayParams, SVR_GS_FS, 'GS_FS', 'SVR_GS_FS')

#>>>
GS_FSs = [LR_GS_FS, LR_RIDGE_GS_FS, LR_LASSO_GS_FS, LR_ELAST_GS_FS, KRR_GS_FS, SVR_GS_FS]

# REPORT
scoreList = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore']
reportGridsearchAsTable (DB_Values['DBpath'], displayParams, GS_FSs, scoreList = scoreList, objFolder ='GS_FS', display = True)

#VISUALIZE

# SCORES
for scoreLabel in scoreList:
    GS_ParameterPlot2D(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', yLim=None, score=scoreLabel,
                       studyFolder='GS_FS/')

    GS_ParameterPlot3D(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', yLim=None,
                       score=scoreLabel, colorsPtsLsBest=['b', 'g', 'c', 'y', 'r'], size=[6, 6], showgrid=True,
                       maxScore=True, absVal=False, ticks=False, lims=False, studyFolder='GS_FS/')

    heatmap(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', score=scoreLabel, studyFolder='GS_FS/')

# PREDICTION VS GROUNDTRUTH
for GS_FS in GS_FSs:
    for learningDflabel in GS_FS.learningDfsList:
        GS = GS_FS.__getattribute__(learningDflabel)
        plotPredTruth(displayParams = displayParams, modelGridsearch = GS,DBpath = DB_Values['DBpath'],
                       TargetMinMaxVal = FORMAT_Values['TargetMinMaxVal'], fontsize = 14,studyFolder = 'GS_FS/')
        plotResiduals(modelGridsearch = GS, displayParams = displayParams,  DBpath = DB_Values['DBpath'],
                    bins=20, binrange = [-200, 200], studyFolder = 'GS_FS/')
        paramResiduals(modelGridsearch = GS, displayParams = displayParams, DBpath = DB_Values['DBpath'],
                   yLim = PROCESS_VALUES['residualsYLim'], xLim = PROCESS_VALUES['residualsXLim'], studyFolder = 'GS_FS/')

GS_predTruthCombined(displayParams, GS_FSs, DB_Values['DBpath'], content = 'GS_FSs', scatter=True, fontsize=14, studyFolder = 'GS_FS/') #scatter=False for groundtruth as line
for GS_FS in GS_FSs:
    GS_predTruthCombined(displayParams, [GS_FS], DB_Values['DBpath'], content = str(GS_FS), scatter=True, fontsize=14, studyFolder = 'GS_FS/')

# METRICS
GS_MetricsSummaryPlot(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FSs', scatter=True, studyFolder = 'GS_FS/')
for GS_FS in GS_FSs:
    GS_MetricsSummaryPlot([GS_FS], displayParams, DB_Values['DBpath'], content = str(GS_FS), scatter=True, studyFolder = 'GS_FS/')

# WEIGHTS                   #ONLY FOR GS with identical weights
for GS_FS in GS_FSs:
    GS_WeightsBarplotAll([GS_FS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'],
                         content=str(GS_FS), df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
GS_WeightsSummaryPlot(GS_FSs, GS_FSs, target = FORMAT_Values['targetLabels'], displayParams =displayParams,
                       DBpath = DB_Values['DBpath'], content='GS_FSs', sorted=True, yLim=4,
                         df_for_empty_labels=baseFormatedDf.trainDf, fontsize=14,  studyFolder='GS_FS/')
for GS_FS in GS_FSs:
    GS_WeightsSummaryPlot([GS_FS], GS_FSs, target=FORMAT_Values['targetLabels'], displayParams=displayParams,
                          DBpath=DB_Values['DBpath'], content=str(GS_FS), sorted=True, yLim=4,
                          df_for_empty_labels=baseFormatedDf.trainDf, fontsize=14, studyFolder='GS_FS/')



# run all > check baseformatted for SVR
# change name ModelsGS for Models

#todo : check residuals - do i need to replicate/CV results for more data > see plotresiduals
# todo : scaled but reaches 4??


# todo : fix issue results vary strongly depending on if from Model GS or from ModelFeature selection GS
#  - this might be due to the differing cv in both GS > more data  > more reproduciblle?
#todo :test on other data !!


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

# #todo : check summary equation table