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



DBname = DB_Values['acronym'] + '_rd' #'CSTB_v2_rd' + yLabels[0]  #'CSTB_v2_rd'


studies_GS_FS = []
studies_Blender = []
# randomvalues = list(range(42, 53))
randomvalues = list(range(38, 40))#44

for value in randomvalues:
    PROCESS_VALUES['random_state'] = value
    displayParams["reference"] = DBname + str(PROCESS_VALUES['random_state']) + '/'
    print('Run Study for random_state:', value)
#
#     # RUN
    rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = Run_FS_Study()
    GS_FSs, blendModel = Run_GS_FS_Study(DBname + str(PROCESS_VALUES['random_state']) + '/')

#IMPORT
for value in randomvalues:
    PROCESS_VALUES['random_state'] = value
    displayParams["reference"] = DBname + str(PROCESS_VALUES['random_state']) + '/'
    print('Import Study for random_state:', value)

    #IMPORT
    import_reference = displayParams["reference"]
    rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(import_reference, show = False)
    GS_FSs = import_Main_GS_FS(import_reference, GS_FS_List_Labels = ['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST',  'KRR_RBF', 'KRR_LIN','KRR_POL','SVR_LIN', 'SVR_RBF'])#

    # blendModel1 = Run_Blending(GS_FSs, displayParams, DB_Values["DBpath"], 10, checkR2 = True)
    # blendModel2 = Run_Blending(GS_FSs, displayParams, DB_Values["DBpath"], 10, checkR2 = False)
    #

    # scoreList = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore']
    # scoreListMax = [True, False, True, True, True]
    # Plot_GS_FS_Scores(GS_FSs, scoreList, scoreListMax)

    Blender = import_Main_Blender(import_reference)
    #
    studies_GS_FS.append(GS_FSs)
    studies_Blender.append(Blender)

# PREDICT
# PredictionDict = computePrediction(GS)

#COMBINE
reportCombinedStudies(studies_Blender, displayParams, DB_Values['DBpath'], random_seeds = randomvalues)
ResultsDf = ReportStudyResults(studies_GS_FS, displayParams, DB_Values['DBpath'])

RUN_CombinedResiduals(studies_GS_FS, studies_Blender, displayParams, FORMAT_Values, DBpath = DB_Values['DBpath'])
RUN_SHAP_Combined(displayParams, DB_Values["DBpath"], studies_Blender, studies_GS_FS, xQuantLabels, xQualLabels)














