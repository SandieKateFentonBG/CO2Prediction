""""
# GOAL 
# RUN NBest
# RUN BLender
"""""

#DASHBOARD IMPORT
from Dashboard_EUCB_FR_v2 import *

#SCRIPT IMPORTS
from Main_GS_FS_Steps import *
from Main_BL_Steps import *
from Main_Combine_Steps import *

#LIBRARY IMPORTS



Studies_CV_BlenderNBest = []

for set in studyParams['sets']:
    yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set

    print("Study for :", set)
    All_CV, NBest_CV, Filters_Pearson_CV, Filters_Spearman_CV, Blenders_NBest_CV, Blenders_Single_CV = [],[],[],[],[],[]
    randomvalues = studyParams['randomvalues']

    for value in randomvalues:

        PROCESS_VALUES['random_state'] = value
        print('Import Study for random_state:', value)

        # ">>IMPORT"

        ref_prefix = DB_Values['acronym'] + '_' + studyParams['sets'][0][1]
        ref_suffix_single, ref_suffix_combined = '_rd'+ str(PROCESS_VALUES['random_state']) + '/', '_Combined/'
        ref_single, ref_combined = ref_prefix + ref_suffix_single, ref_prefix + ref_suffix_combined
        displayParams["ref_prefix"], displayParams["reference"] = ref_prefix, ref_single

        # MODEL PROCESSING
        GS_FSs = import_Main_GS_FS(ref_single, GS_FS_List_Labels = studyParams['Regressors'])
        # GS_FSs = import_Main_GS_FS(ref_single, GS_FS_List_Labels=['KRR_LIN'])
        #
        # # NBEST PROCESSING
        NBestModels = import_NBest(ref_single, OverallBest = BLE_VALUES['OverallBest'])

        # # BLENDER PROCESSING
        Blender_NBest = import_Blender_NBest(ref_single, label = BLE_VALUES['Regressor'] + '_Blender_NBest')

        # ">>RUN"

        # # NBEST PROCESSING
        # NBestModels = Run_NBest_Study(ref_single, importNBest=False, OverallBest = BLE_VALUES['OverallBest'])
        #
        # # BLENDER PROCESSING
        # Blender_NBest = Run_Blending_NBest(NBestModels.modelList, displayParams, DB_Values['DBpath'], ref_single,
        #                                    ConstructorKey=BLE_VALUES['Regressor'])

        # ">>STORE"

        All_CV.append(GS_FSs)
        NBest_CV.append(NBestModels)
        Blenders_NBest_CV.append(Blender_NBest)

    # COMBINE

    RUN_Combine_Report(All_CV, NBest_CV, Blenders_NBest_CV, randomvalues, displayParams)

    # META STORE

    # Studies_CV_BlenderNBest.append(CV_BlenderNBest)

# AccuracyCheck(Studies_CV_BlenderNBest, sets, DB_Values['acronym'], displayParams, DB_Values['DBpath'], tolerance=0.15)
