# todo : choose database

#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
# from Dashboard_EUCB_FR import *
from Dashboard_EUCB_FR_v2 import *

#SCRIPT IMPORTS
from Main_FS_Steps import *
from Main_GS_FS_Steps import *
from Main_BL_Steps import *
from Main_Combine_Steps import *
from CVBlending import *




Studies_CV_BlenderNBest = []

for set in studyParams['sets']:
    yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set


    print("Study for :", set)

    DBname = DB_Values['acronym'] + '_' + yLabelsAc + '_rd'

    CV_AllModels, BL_NBest_All, CV_Filters_Pearson, CV_Filters_Spearman = [], [], [], []
    randomvalues = studyParams['randomvalues']

    for value in randomvalues:

        PROCESS_VALUES['random_state'] = value
        displayParams["reference"] = DBname + str(PROCESS_VALUES['random_state']) + '/'

        # IMPORTS
        import_reference = displayParams["reference"]
        print('Import Study for random_state:', value)

        # FEATURE PROCESSING
        # rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(import_reference, show = False)


        # MODEL PROCESSING
        # GS_FSs = import_Main_GS_FS(import_reference, GS_FS_List_Labels = studyParams['Regressors'])
        # GS_FSs = import_Main_GS_FS(import_reference, GS_FS_List_Labels=['KRR_LIN'])

        # NBEST PROCESSING
        # NBestModels = import_NBest(import_reference, value)

        # BLENDER PROCESSING
        # Blender_NBest = import_Blender_NBest(import_reference, label = BLE_VALUES['Regressor'] + '_Blender_NBest')


        # RUNS

        # FEATURE PROCESSING
        # rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = Run_FS_Study()

        # MODEL PROCESSING & NBEST PROCESSING
        GS_FSs, NBestModels = Run_GS_FS_Study(DBname + str(PROCESS_VALUES['random_state']) + '/', importMainGSFS=False)

        # BLENDER PROCESSING
        blendModel_NBest = Run_Blending_NBest(NBestModels.modelList, displayParams, DB_Values['DBpath'], import_reference, ConstructorKey='LR_RIDGE')


#         # STORE
#         CV_AllModels.append(GS_FSs)
#
#         #
#         BL_NBest_All.append(Blender_NBest)

#todo : check from here
#todo : launch from 45-50

#         # CV_Filters_Pearson.append(pearsonFilter)
#         # CV_Filters_Spearman.append(spearmanFilter)
#
#     # PREDICT
#     # computePrediction_NBest(CV_BlenderNBest)
#     # PredictionDict = computePrediction(GS)
#
#     # COMBINE
#     RUN_CV_Report(CV_AllModels, CV_BlenderNBest, CV_Filters_Spearman, CV_Filters_Pearson, randomvalues,
#                   displayParams, GSName = "LR")
#
#     # RUN_SHAP_Combined_NBest(displayParams, DB_Values["DBpath"], CV_BlenderNBest, CV_AllModels, xQuantLabels, xQualLabels, n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'], randomValues = randomvalues)
#     # RUN_SHAP_Combined_All(displayParams, DB_Values["DBpath"], CV_AllModels, GSName = 'KRR_LIN', xQuantLabels = xQuantLabels, xQualLabels = xQualLabels, randomValues = randomvalues)
#
#     # # META STORE
#     # Studies_CV_BlenderNBest.append(CV_BlenderNBest)
#
# # AccuracyCheck(Studies_CV_BlenderNBest, sets, DB_Values['acronym'], displayParams, DB_Values['DBpath'], tolerance=0.15)
#
# CV_Blender = CVBlend(CV_AllModels)
# # print(CV_Blender)
#
#
# # Could I integrate WEC to features and predict SEC with it?
# # could I do a 2 step ML 1. predict WEC 2. Predict SEC with WEC
#
# #todo : there was a name change from Weights to Model Weights > changes might have been done wrong > could generate errors

report_BL_NBest_All(BL_NBest_All, displayParams, DB_Values["DBpath"], studyParams['randomvalues'])
#
#


