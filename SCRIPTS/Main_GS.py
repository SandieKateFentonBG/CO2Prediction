# todo : choose database

#DASHBOARD IMPORT
from dashBoard import *
# from Dashboard_PM_v2 import *
# from Dashboard_EUCB_FR import *

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
from ModelReport import *
from Gridsearch import *
from ExportStudy import *

#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

import_reference = displayParams["reference"]

# #IMPORT Main_FS
rdat = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/DATA/rdat.pkl', show = False)
df = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/DATA/df.pkl', show = False)
learningDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/DATA/learningDf.pkl', show = False)
baseFormatedDf = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/DATA/baseFormatedDf.pkl', show = False)
spearmanFilter = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/FS/spearmanFilter.pkl', show = False)
pearsonFilter = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/FS/pearsonFilter.pkl', show = False)
RFEs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ import_reference +'RECORDS/FS/RFEs.pkl', show = False)


"""
------------------------------------------------------------------------------------------------------------------------
6.HYPERPARAMETER SEARCH
------------------------------------------------------------------------------------------------------------------------
"""

"""
MODELS
"""

#ABOUT
"""
GOAL -  Find best hyperparameters for each models
Dashboard Input - _VALUES : xx
"""
#CONSTRUCT
myFormatedDf = spearmanFilter
# #
LR_GS = ModelGridsearch('LR', learningDf= myFormatedDf, modelPredictor= LinearRegression(), param_dict=dict())
LR_LASSO_GS = ModelGridsearch('LR_LASSO', learningDf= myFormatedDf, modelPredictor= Lasso(), param_dict = LR_param_grid)
LR_RIDGE_GS = ModelGridsearch('LR_RIDGE', learningDf= myFormatedDf, modelPredictor= Ridge(), param_dict = LR_param_grid)
LR_ELAST_GS = ModelGridsearch('LR_ELAST', learningDf= myFormatedDf, modelPredictor= ElasticNet(), param_dict = LR_param_grid)
KRR_GS = ModelGridsearch('KRR', learningDf= myFormatedDf, modelPredictor= KernelRidge(), param_dict = KRR_param_grid)
SVR_GS = ModelGridsearch('SVR', learningDf= myFormatedDf, modelPredictor= SVR(), param_dict = SVR_param_grid)

GSs = [LR_GS, LR_LASSO_GS, LR_RIDGE_GS, LR_ELAST_GS, KRR_GS, SVR_GS]


#STOCK
pickleDumpMe(DB_Values['DBpath'], displayParams, GSs, 'GS', 'GSs')
# saveStudy(DB_Values['DBpath'], displayParams, obj= myFormatedDf, objFolder = 'GS')

#REPORT
reportModels(DB_Values['DBpath'], displayParams, GSs, myFormatedDf, display = True)

#VISUALIZE

#INDIVIDUAL
for GS in GSs:#,LR_LASSO_GS, LR_RIDGE_GS, LR_ELAST_GS
    plotPredTruth(displayParams = displayParams, modelGridsearch = GS,
                  DBpath = DB_Values['DBpath'], TargetMinMaxVal = FORMAT_Values['TargetMinMaxVal'], fontsize = 14)
    paramResiduals(modelGridsearch = GS, displayParams = displayParams,
                          DBpath = DB_Values['DBpath'], yLim = PROCESS_VALUES['residualsYLim'] , xLim = PROCESS_VALUES['residualsXLim'])
    plotResiduals(modelGridsearch = GS, displayParams = displayParams,  DBpath = DB_Values['DBpath'],
                bins=20, binrange = [-200, 200])

#GROUPED
predTruthCombined(displayParams, GSs, DBpath = DB_Values['DBpath'], scatter=True, fontsize=14) #scatter=False for groundtruth as line


MetricsSummaryPlot(GSs, displayParams, DBpath  = DB_Values['DBpath'], metricLabels=['TrainScore', 'TestScore', 'TestAcc', 'TestMSE'],
                       title='Model Evaluations', ylabel='Evaluation Metric', fontsize=14, scatter=True)

WeightsBarplotAll(GSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'],
                  df=None, yLim = None, sorted = True, key = 'WeightsScaled' )

WeightsSummaryPlot(GSs, displayParams, DB_Values['DBpath'], sorted=True, yLim=None, fontsize=14)


"""
HYPERPARAMETERS
"""
#ABOUT
"""
GOAL -  Find the influence of 1 hyperparameters on models
Dashboard Input - _VALUES : xx
"""

myFormatedDf = pearsonFilter

#CONSTRUCT
KRR_GS1 = ModelGridsearch(predictorName='KRR_lin', modelPredictor= KernelRidge(), param_dict = KRR_param_grid1, learningDf= myFormatedDf)
KRR_GS2 = ModelGridsearch(predictorName='KRR_poly', modelPredictor= KernelRidge(), param_dict = KRR_param_grid2, learningDf= myFormatedDf)
KRR_GS3 = ModelGridsearch(predictorName='KRR_rbf', modelPredictor= KernelRidge(), param_dict = KRR_param_grid3, learningDf= myFormatedDf)
KRR_GS = [KRR_GS1, KRR_GS2, KRR_GS3]

#VISUALIZE
ParameterPlot2D(KRR_GS,  displayParams, DBpath = DB_Values['DBpath'],  yLim = None, paramKey ='gamma', score ='mean_test_r2', log = True)
ParameterPlot3D(KRR_GS, displayParams, DBpath = DB_Values['DBpath'],
                      colorsPtsLsBest=['b', 'g', 'c', 'y'], paramKey='gamma', score='mean_test_r2',
                      size=[6, 6], showgrid=False, log=True, maxScore=True, absVal = False,  ticks=False, lims=False)

#STOCK
pickleDumpMe(DB_Values['DBpath'], displayParams, KRR_GS, 'GS', 'KRR_GS_gamma')

#EXPORT

# GSs = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/GS/GSs.pkl', show = False)
# KRR_GS_gamma = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'+ displayParams["reference"] +'RECORDS/GS/KRR_GS_gamma.pkl', show = False)

FiltersLs = [spearmanFilter, pearsonFilter]
GSlist = GSs
exportStudy(displayParams, DB_Values, FORMAT_Values, PROCESS_VALUES, RFE_VALUES, GS_VALUES, rdat, df, learningDf,
                baseFormatedDf, FiltersLs, RFEs, GSlist, GSwithFS = False)

#QUESTIONS
#how do I interpret dual coefs in weights plot? does this bias the results?





