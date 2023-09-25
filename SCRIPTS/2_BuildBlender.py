from Dashboard_Current import *
from Main_GS_FS_Steps import *
from Main_BL_Steps import *
from Main_Combine_Steps import *
from AccuracyCheck import *

# 4 BLEND MODELS
#
# for set in studyParams['sets']:
#
#     yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set
#     displayParams["ref_prefix"] = acronym + '_' + studyParams['sets'][0][1]
#
#     for i in range(1, cv + 1):
#
#         displayParams["reference"] = displayParams["ref_prefix"] + '_rd' + str(i) + '/'
#
#         # NBEST PROCESSING
#
#         # RUN
#         NBestModels = Run_NBest_Study(import_FS_ref=displayParams["reference"], import_GS_FS_ref=displayParams["reference"],
#                                       importNBest=False, OverallBest=BLE_VALUES['OverallBest'])
#         # # IMPORT
#         # NBestModels = import_NBest(displayParams["reference"] , OverallBest=BLE_VALUES['OverallBest'], number = i)
#
#         # BLENDER PROCESSING
#         # RUN
#         Blender_NBest = Run_Blending_NBest(NBestModels.modelList, displayParams, DB_Values['DBpath'], ref_single=displayParams["reference"],
#                                            ConstructorKey=BLE_VALUES['Regressor'][0])
#         # # IMPORT
#         # Blender_NBest = import_Blender_NBest(ref_single=displayParams["reference"], label=BLE_VALUES['Regressor'][0] + '_Blender_NBest')
#
#
# for set in studyParams['sets']:
#
#     yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set
#     displayParams["ref_prefix"] = acronym + '_' + studyParams['sets'][0][1]
#
#     for i in range(1, cv + 1):
#
#         displayParams["reference"] = displayParams["ref_prefix"] + '_rd' + str(i) + '/'
#
#         # NBEST PROCESSING
#
#         # RUN
#         # NBestModels = Run_NBest_Study(import_FS_ref=displayParams["reference"], import_GS_FS_ref=displayParams["reference"],
#         #                               importNBest=False, OverallBest=BLE_VALUES['OverallBest'])
#         # # IMPORT
#         NBestModels = import_NBest(displayParams["reference"] , OverallBest=BLE_VALUES['OverallBest'], number = i)
#
#         # BLENDER PROCESSING
#         # RUN
#         Blender_NBest = Run_Blending_NBest(NBestModels.modelList, displayParams, DB_Values['DBpath'], ref_single=displayParams["reference"],
#                                            ConstructorKey=BLE_VALUES['Regressor'][1])
#         # # IMPORT
#         # Blender_NBest = import_Blender_NBest(ref_single=displayParams["reference"], label=BLE_VALUES['Regressor'][1] + '_Blender_NBest')

# 0 REPORT

Studies_CV_BlenderNBest = []

for set in studyParams['sets']:

    yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set
    displayParams["ref_prefix"] = acronym + '_' + studyParams['sets'][0][1]
    All_CV, NBest_CV, regressors_CV,Blender_NBests, models_CV = [],[],[],[],[]
    LR_RIDGE_BLs, SVR_RBF_BLs = [], []

    for i in range(1, cv + 1):

        displayParams["reference"] = displayParams["ref_prefix"] + '_rd' + str(i) + '/'

        regressor = import_Main_GS_FS(displayParams["reference"], GS_FS_List_Labels=[studyParams['Regressors'][0]])  # SVR_RBF
        model = regressor[0].NoSelector # SVR_RBF.NoSelector
        NBestModels = import_NBest(displayParams["reference"], OverallBest=BLE_VALUES['OverallBest'], number = i)  # 10 Best Models
        GS_FSs = import_Main_GS_FS(displayParams["reference"], GS_FS_List_Labels=studyParams['Regressors']) #All models (54)

        # Blender_NBest = import_Blender_NBest(ref_single=displayParams["reference"],label=BLE_VALUES['Regressor'] + '_Blender_NBest') # Blender
        LR_RIDGE_BL = import_Blender_NBest(displayParams["reference"], label = 'LR_RIDGE' + '_Blender_NBest') # LR_RIDGE Blender
        SVR_RBF_BL = import_Blender_NBest(displayParams["reference"], label = 'SVR_RBF' + '_Blender_NBest') # SVR_RBF Blender

        regressors_CV.append(regressor)
        models_CV.append(model)
        All_CV.append(GS_FSs)
        NBest_CV.append(NBestModels)
        # Blender_NBests.append(Blender_NBest)
        LR_RIDGE_BLs.append(LR_RIDGE_BL)
        SVR_RBF_BLs.append(SVR_RBF_BL)

    Blenders_NBest_CV = [LR_RIDGE_BLs,SVR_RBF_BLs] #[Blender_NBests]

    # COMBINE
    RUN_Combine_Report(All_CV, NBest_CV, Blenders_NBest_CV, regressors_CV, models_CV, randomvalues=list(range(1, cv+1)), displayParams=displayParams)

    Studies_CV_BlenderNBest.append(Blenders_NBest_CV)

AccuracyCheck(Blenders_NBest_CV, studyParams['sets'], displayParams, DB_Values['DBpath'], tolerance=PROCESS_VALUES['accuracyTol'])
