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
from AccuracyCheck import *

# DBname = DB_Values['acronym'] + '_rd'


Studies_CV_BlenderNBest = []

sets = [
    ['Embodied_Carbon[kgCO2e_m2]','EC','TestR2'],
    ['Embodied_Carbon[kgCO2e_m2]','EC','TestAcc'],
    ['Embodied_Carbon_Structure[kgCO2e_m2]','ECS', 'TestR2'],
    ['Embodied_Carbon_Structure[kgCO2e_m2]','ECS','TestAcc']]

for set in sets:
    yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set

    print("Study for :", set)

    DBname = DB_Values['acronym'] + '_' + yLabelsAc + '_rd'

    CV_AllModels = []
    CV_BlenderNBest = []

    # randomvalues = list(range(40, 51))
    randomvalues = list(range(40, 41))

    # for value in randomvalues:
    #     PROCESS_VALUES['random_state'] = value
    #     displayParams["reference"] = DBname + str(PROCESS_VALUES['random_state']) + '/'
    #     print('Run Study for random_state:', value)
    # #
    # #     # RUN
    # #     rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = Run_FS_Study()
    #     GS_FSs, blendModel = Run_GS_FS_Study(DBname + str(PROCESS_VALUES['random_state']) + '/', importMainGSFS = False)

    #IMPORT
    for value in randomvalues:
        PROCESS_VALUES['random_state'] = value
        displayParams["reference"] = DBname + str(PROCESS_VALUES['random_state']) + '/'
        print('Import Study for random_state:', value)

        #IMPORT
        import_reference = displayParams["reference"]
        rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(import_reference, show = False)
        # GS_FSs, blendModel = Run_GS_FS_Study(DBname + str(PROCESS_VALUES['random_state']) + '/', importMainGSFS = True)
        GS_FSs = import_Main_GS_FS(import_reference, GS_FS_List_Labels = ['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST',  'KRR_RBF', 'KRR_LIN','KRR_POL','SVR_LIN', 'SVR_RBF'])#
        Blender = import_Main_Blender(import_reference, NBestScore = BLE_VALUES['NBestScore'])
        #
        CV_AllModels.append(GS_FSs)
        CV_BlenderNBest.append(Blender)

    Studies_CV_BlenderNBest.append(CV_BlenderNBest)



    # PREDICT
    computePrediction_NBest(CV_BlenderNBest)
    # PredictionDict = computePrediction(GS)

    #COMBINE

    # reportCV_Scores_NBest(CV_BlenderNBest, displayParams, DB_Values['DBpath'], NBestScore=BLE_VALUES['NBestScore'], random_seeds = randomvalues)
    # ResultsDf = reportCV_ScoresAvg_All(CV_AllModels, displayParams, DB_Values['DBpath'])
    # reportCV_ModelRanking_NBest(CV_AllModels, CV_BlenderNBest, seeds = randomvalues, displayParams = displayParams, DBpath =DB_Values['DBpath'], NBestScore=BLE_VALUES['NBestScore'])
    #
    # RUN_SHAP_Combined(displayParams, DB_Values["DBpath"], CV_BlenderNBest, CV_AllModels, xQuantLabels, xQualLabels, NBestScore=BLE_VALUES['NBestScore'], randomValues = randomvalues)
    # RUN_CombinedResiduals(CV_AllModels, CV_BlenderNBest, displayParams, FORMAT_Values, DBpath = DB_Values['DBpath'], NBestScore=BLE_VALUES['NBestScore'])
print(DB_Values['acronym'])
AccuracyCheck(Studies_CV_BlenderNBest, sets, DB_Values['acronym'], displayParams, DB_Values['DBpath'], tolerance=0.15)


# look at my schema
#why trains score = 0 / test score = negatif but accuracy super high? > this occurs for LASSO/ELAST/RIDGE
#my models with highest accuracy and positive R2 are the ones with lowest MSE!
# how use R2 at best?
#what conclusions to take from my CVModel ranking? how to take best average? Should I have done this with all models?
# why does Structural EC work so bad? but R2 are good
# Could I integrate WEC to features and predict SEC with it?
# could I do a 2 step ML 1. predict WEC 2. Predict SEC with WEC

#what about my way of combining SHAPs?
#SHAP graphs with vertical bars > means all sample points have same value ! > this is good?

#todo : GRID doesn't display well > change










