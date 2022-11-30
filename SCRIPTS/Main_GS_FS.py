# todo : choose database

#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
from Dashboard_EUCB_FR import *

#SCRIPT IMPORTS
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
from ExportStudy import *
from GridsearchSHAPPt import *

#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import shap

import_reference = 'CSTB_large/' #displayParams["reference"]

# #IMPORT Main_FS
rdat = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/DATA/rdat.pkl', show = False)
df = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/DATA/df.pkl', show = False)
learningDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/DATA/learningDf.pkl', show = False)
baseFormatedDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/DATA/baseFormatedDf.pkl', show = False)
spearmanFilter = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/FS/spearmanFilter.pkl', show = False)
pearsonFilter = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/FS/pearsonFilter.pkl', show = False)
RFEs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/FS/RFEs.pkl', show = False)

# #>>>
# learning_dfs = [spearmanFilter, pearsonFilter] + RFEs + [baseFormatedDf]
# print("Learning dataframes (%s) :" % len(learning_dfs), learning_dfs )

# #IMPORT Main_GS_FS
# todo : update this list if you are importing GS_FS from pickles #'LR',
GS_FS_List_Labels = ['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST',  'KRR_LIN', 'KRR_RBF', 'KRR_POL', 'SVR_LIN', 'SVR_RBF']#, 'SVR_POL'

GS_FSs = []
for FS_GS_lab in GS_FS_List_Labels:
    path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/GS_FS/' + FS_GS_lab + '.pkl'
    GS_FS = pickleLoadMe(path = path, show = False)
    for DfLabel in GS_FS.learningDfsList:
        GS = GS_FS.__getattribute__(DfLabel)

    GS_FSs.append(GS_FS)
#
# # unpack
# todo : update this list if you are importing GS_FS from pickles - should match GS_FS_List_Labels #
[LR, LR_RIDGE, LR_LASSO, LR_ELAST,  KRR_LIN, KRR_RBF,KRR_POL, SVR_LIN, SVR_RBF] = GS_FSs #, SVR_POL

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

# todo : (i) kernel can be a hyperparameter
#
# #CONSTRUCT
# LR_CONSTRUCTOR = {'name' : 'LR',  'modelPredictor' : LinearRegression(),'param_dict' : dict()}
# LR_RIDGE_CONSTRUCTOR = {'name' : 'LR_RIDGE',  'modelPredictor' : Lasso(),'param_dict' : LR_param_grid}
# LR_LASSO_CONSTRUCTOR = {'name' : 'LR_LASSO',  'modelPredictor' : Ridge(),'param_dict' : LR_param_grid}
# LR_ELAST_CONSTRUCTOR = {'name' : 'LR_ELAST',  'modelPredictor' : ElasticNet(),'param_dict' : LR_param_grid}
# KRR_LIN_CONSTRUCTOR = {'name': 'KRR_LIN', 'modelPredictor': KernelRidge(kernel='linear'), 'param_dict': KRR_param_grid}
# KRR_RBF_CONSTRUCTOR = {'name': 'KRR_RBF', 'modelPredictor': KernelRidge(kernel='rbf'), 'param_dict': KRR_param_grid}
# KRR_POL_CONSTRUCTOR = {'name' : 'KRR_POL',  'modelPredictor' : KernelRidge(kernel = 'poly'),'param_dict' : KRR_param_grid}
# SVR_LIN_CONSTRUCTOR = {'name' : 'SVR_LIN',  'modelPredictor' : SVR(kernel ='linear'),'param_dict' : SVR_param_grid}
# SVR_RBF_CONSTRUCTOR = {'name' : 'SVR_RBF',  'modelPredictor' : SVR(kernel ='rbf'),'param_dict' : SVR_param_grid}
# SVR_POL_CONSTRUCTOR = {'name' : 'SVR_POL',  'modelPredictor' : SVR(kernel ='poly'),'param_dict' : SVR_param_grid}
#
# GS_CONSTRUCTOR = [LR_CONSTRUCTOR, LR_RIDGE_CONSTRUCTOR, LR_LASSO_CONSTRUCTOR, LR_ELAST_CONSTRUCTOR, KRR_LIN_CONSTRUCTOR,
#                   KRR_RBF_CONSTRUCTOR,KRR_POL_CONSTRUCTOR, SVR_LIN_CONSTRUCTOR, SVR_RBF_CONSTRUCTOR] #, SVR_POL_CONSTRUCTOR
#
# # CONSTRUCT & REPORT
#
# GS_FSs = []
# for constructor in GS_CONSTRUCTOR :
#     GS_FS = ModelFeatureSelectionGridsearch(predictorName=constructor['name'], learningDfs=learning_dfs,
#                                         modelPredictor=constructor['modelPredictor'], param_dict=constructor['param_dict'])
#     GS_FSs.append(GS_FS)
#     reportGridsearch(DB_Values['DBpath'], displayParams, GS_FS, objFolder='GS_FS', display=True)
#     pickleDumpMe(DB_Values['DBpath'], displayParams, GS_FS, 'GS_FS', constructor['name'])
#
# unpack
# [LR, LR_RIDGE, LR_LASSO, LR_ELAST, KRR_LIN, KRR_RBF,KRR_POL, SVR_LIN, SVR_RBF] = GS_FSs #, SVR_POL
#
# """
# ------------------------------------------------------------------------------------------------------------------------
# VISUALIZE AND REPORT
# ------------------------------------------------------------------------------------------------------------------------
# """


exportStudy(displayParams, DB_Values, FORMAT_Values, PROCESS_VALUES, RFE_VALUES, GS_VALUES, rdat, df, learningDf,
                baseFormatedDf, FiltersLs = [spearmanFilter, pearsonFilter], RFEs = RFEs, GSlist = GS_FSs, GSwithFS = True)
#
# # REPORT
# scoreList = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore']
# scoreListMax = [True, False, True, True, True]
# reportGridsearchAsTable(DB_Values['DBpath'], displayParams, GS_FSs, scoreList = scoreList, objFolder ='GS_FS', display = True)
#
# # VISUALIZE
#
# # SCORES
# for scoreLabel in scoreList:
#     heatmap(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', score=scoreLabel, studyFolder='GS_FS/')
#
# for scoreLabel, scoreMax in zip(scoreList, scoreListMax):
#     GS_ParameterPlot2D(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', yLim=None, score=scoreLabel,
#                        studyFolder='GS_FS/')
#     GS_ParameterPlot3D(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', yLim=None,
#                        score=scoreLabel, colorsPtsLsBest=['b', 'g', 'c', 'y', 'r'], size=[6, 6], showgrid=True,
#                        maxScore=scoreMax, absVal=False, ticks=False, lims=False, studyFolder='GS_FS/')


# # WEIGHTS                   #ONLY FOR GS with identical weights
# for GS_FS in GS_FSs:
#     name = GS_FS.predictorName + '_GS_FS'
#     print(name)
#     GS_WeightsBarplotAll([GS_FS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'],
#                          content=name, df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
# GS_WeightsSummaryPlot(GS_FSs, GS_FSs, target = FORMAT_Values['targetLabels'], displayParams =displayParams,
#                        DBpath = DB_Values['DBpath'], content='GS_FSs', sorted=True, yLim=4,
#                          df_for_empty_labels=baseFormatedDf.trainDf, fontsize=14,  studyFolder='GS_FS/')
# for GS_FS in GS_FSs:
#     GS_WeightsSummaryPlot([GS_FS], GS_FSs, target=FORMAT_Values['targetLabels'], displayParams=displayParams,
#                           DBpath=DB_Values['DBpath'], content=GS_FS.predictorName + '_GS_FS', sorted=True, yLim=4,
#                           df_for_empty_labels=baseFormatedDf.trainDf, fontsize=14, studyFolder='GS_FS/')

# METRICS
GS_MetricsSummaryPlot(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FSs', scatter=True, studyFolder = 'GS_FS/')
for GS_FS in GS_FSs:
    GS_MetricsSummaryPlot([GS_FS], displayParams, DB_Values['DBpath'], content = GS_FS.predictorName + '_GS_FS', scatter=True, studyFolder = 'GS_FS/')


# PREDICTION VS GROUNDTRUTH
GS_predTruthCombined(displayParams, GS_FSs, DB_Values['DBpath'], content = 'GS_FSs', scatter=True, fontsize=14, studyFolder = 'GS_FS/') #scatter=False for groundtruth as line
for GS_FS in GS_FSs:
    GS_predTruthCombined(displayParams, [GS_FS], DB_Values['DBpath'], content = GS_FS.predictorName + '_GS_FS', scatter=True, fontsize=14, studyFolder = 'GS_FS/')

for GS_FS in GS_FSs:
    for learningDflabel in GS_FS.learningDfsList:
        GS = GS_FS.__getattribute__(learningDflabel)
        plotPredTruth(displayParams = displayParams, modelGridsearch = GS,DBpath = DB_Values['DBpath'],
                       TargetMinMaxVal = FORMAT_Values['TargetMinMaxVal'], fontsize = 14,studyFolder = 'GS_FS/')
        plotResiduals(modelGridsearch = GS, displayParams = displayParams,  DBpath = DB_Values['DBpath'],
                    bins=20, binrange = [-200, 200], studyFolder = 'GS_FS/')
        paramResiduals(modelGridsearch = GS, displayParams = displayParams, DBpath = DB_Values['DBpath'],
                   yLim = PROCESS_VALUES['residualsYLim'], xLim = PROCESS_VALUES['residualsXLim'], studyFolder = 'GS_FS/')


# SHAP

for GS_FS in GS_FSs:
    for learningDflabel in GS_FS.learningDfsList:
        GS = GS_FS.__getattribute__(learningDflabel)
        shap_values, explainer = computeSHAP(GS)
        plot_shap_group_cat(GS, xQuantLabels, xQualLabels, displayParams=displayParams, DBpath=DB_Values['DBpath'])
        plot_shap(GS, displayParams, DBpath=DB_Values['DBpath'], content='', studyFolder='GS_FS/')



#todo : check correlation matrix with empty lines - why? how to deal with this?
#todo : check the code is working > GS > am I using the class properly? Am I working with calibrated fit models?
#todo : compare results for different targets but same inputs
#todo : check residuals - do i need to replicate/CV results for more data > see plotresiduals


# #todo : check summary equation table

#todo : check variability of coefficients over the 10 runs :
# - If coefficients vary significantly when changing the input dataset their robustness is not guaranteed,
# and they should probably be interpreted with caution.
# todo : ok interpretation of linear - what about non linear???


#todo how would SHAP handle this? since i have scaled my data ...


"""


#coefficient values - why are my values so high??? they should rage between 0 and 0.5?
#https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html

Multiplying the coefficients by the standard deviation of the related feature would reduce all the coefficients to the same unit of measure. As
we will see after this is equivalent to normalize numerical variables to their standard deviation, as 
y = ∑ coefi × Xi = ∑ (coefi × stdi) × (Xi/stdi)
In that way, we emphasize that the greater the variance of a feature, the larger the weight of the corresponding coefficient on the output, all
else being equal.

... further

For the coefficient analysis, scaling is not needed this time because it was performed during the preprocessing step.



weights obtained with regularization are more stable (see the Ridge regression and classification User Guide section).
This increased stability is visible from the plot, > so can we keep both features or should we still remove one?

unstable resulkts  - does this mean every time we run the script coefficients can vary greatly > convergence is not always reached?
"""
