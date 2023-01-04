# todo : choose database

#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
# from Dashboard_EUCB_FR import *
from Dashboard_EUCB_FR_v2 import *

#SCRIPT IMPORTS
from Main_FS_Steps import *
from Main_GS_FS_Steps import *
from FeatureReport import *
from StudyResiduals import *
from ModelBlending import *
from CombineSHAP import *

DBname = DB_Values['acronym'] + '_rd'


CV_AllModels = []
CV_BlenderNBest = []

randomvalues = list(range(40, 51))

# for value in randomvalues:
#     PROCESS_VALUES['random_state'] = value
#     displayParams["reference"] = DBname + str(PROCESS_VALUES['random_state']) + '/'
#     print('Run Study for random_state:', value)
# #
# #     # RUN
#     rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = Run_FS_Study()
#     GS_FSs, blendModel = Run_GS_FS_Study(DBname + str(PROCESS_VALUES['random_state']) + '/', importMainGSFS = False)

#IMPORT
for value in randomvalues:
    PROCESS_VALUES['random_state'] = value
    displayParams["reference"] = DBname + str(PROCESS_VALUES['random_state']) + '/'
    print('Import Study for random_state:', value)

    #IMPORT
    import_reference = displayParams["reference"]
    rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(import_reference, show = False)
    GS_FSs = import_Main_GS_FS(import_reference, GS_FS_List_Labels = ['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST',  'KRR_RBF', 'KRR_LIN','KRR_POL','SVR_LIN', 'SVR_RBF'])#

    Blender = import_Main_Blender(import_reference)
    #
    CV_AllModels.append(GS_FSs)
    CV_BlenderNBest.append(Blender)

# PREDICT
# PredictionDict = computePrediction(GS)

#COMBINE
reportCV_Scores_NBest(CV_BlenderNBest, displayParams, DB_Values['DBpath'], random_seeds = randomvalues)
ResultsDf = reportCV_ScoresAvg_All(CV_AllModels, displayParams, DB_Values['DBpath'])

reportCV_CV_ModelRanking_NBest(CV_AllModels, CV_BlenderNBest, seeds = randomvalues, displayParams = displayParams, DBpath =DB_Values['DBpath'])
#

RUN_SHAP_Combined(displayParams, DB_Values["DBpath"], CV_BlenderNBest, CV_AllModels, xQuantLabels, xQualLabels, randomValues = randomvalues)
RUN_CombinedResiduals(CV_AllModels, CV_BlenderNBest, displayParams, FORMAT_Values, DBpath = DB_Values['DBpath'])



#why trains score = 0 / test score = negatif but accuracy super high? > this occurs for LASSO/ELAST/RIDGE
#my models with highest accuracy and positive R2 are the ones with lowest MSE!
#SHAP graphs with vertical bars > means all sample points have same value ! > this is good?

#what about my way of combining SHAPs?
#CCAI
#contribution to article

#todo : GRID doesn't display well > change










