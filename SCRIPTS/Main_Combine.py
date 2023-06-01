""""
# GOAL 
# RUN NBest
# RUN BLender
"""""

#DASHBOARD IMPORT
# from Dashboard_EUCB_FR_v2 import *
# from Dashboard_EUCB_Structures import *
from Dashboard_Current import *

#SCRIPT IMPORTS
from Main_GS_FS_Steps import *
from Main_BL_Steps import *
from Main_Combine_Steps import *
from AccuracyCheck import *

#LIBRARY IMPORTS

Studies_CV_BlenderNBest = []

for set in studyParams['sets']:
    yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set

    print("Study for :", set)
    All_CV, NBest_CV, regressors_CV,LR_RIDGE_BLs, SVR_RBF_BLs, models_CV = [],[],[],[],[],[]
    # KRR_LINs, SVR_RBFs, SVR_LINs, KRR_RBFs, KRR_POLs, LRs, LR_RIDGEs, LR_ELASTs, LR_LASSOs = [], [], [], [], [], [], [], [], []
    # Blenders_NBest_CV,  SVR_RBFs = [], []
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

        regressor = import_Main_GS_FS(ref_single, GS_FS_List_Labels=['LR_ELAST']) #SVR_RBF
        model = regressor[0].RFE_RFR
        #
        # # #
        # # # NBEST PROCESSING
        NBestModels = import_NBest(ref_single, OverallBest = BLE_VALUES['OverallBest'])
        # # #
        # # # # # BLENDER PROCESSING
        # # # Blender_NBest = import_Blender_NBest(ref_single, label = BLE_VALUES['Regressor'] + '_Blender_NBest')
        LR_RIDGE_BL = import_Blender_NBest(ref_single, label = 'LR_RIDGE' + '_Blender_NBest')
        # SVR_RBF_BL = import_Blender_NBest(ref_single, label = 'SVR_RBF' + '_Blender_NBest')

        # ">>RUN"

        # # # NBEST PROCESSING
        # NBestModels = Run_NBest_Study(ref_single, importNBest=False, OverallBest = BLE_VALUES['OverallBest'])
        #
        # # BLENDER PROCESSING
        # Blender_NBest = Run_Blending_NBest(NBestModels.modelList, displayParams, DB_Values['DBpath'], ref_single,
        #                                    ConstructorKey=BLE_VALUES['Regressor'])

        # ">>STORE"

        regressors_CV.append(regressor)
        models_CV.append(model)
        All_CV.append(GS_FSs)
        NBest_CV.append(NBestModels)
        LR_RIDGE_BLs.append(LR_RIDGE_BL)
        # SVR_RBF_BLs.append(SVR_RBF_BL)

    Blenders_NBest_CV = [LR_RIDGE_BLs] #,SVR_RBF_BLs

    # COMBINE

    RUN_Combine_Report(All_CV, NBest_CV, Blenders_NBest_CV, regressors_CV, models_CV, randomvalues, displayParams)
    # #
    # RUN_CombinedResiduals(All_CV, NBest_CV, Blenders_NBest_CV, regressors_CV, models_CV, displayParams, FORMAT_Values,
    #                       DB_Values['DBpath'], randomvalues)
    #
    #
    # ResidualPlot_Scatter_Combined(LR_RIDGE_BLs, displayParams, FORMAT_Values, DB_Values['DBpath'], Blender=True, setyLim=None, setxLim=None,
    #                               y_axis = 'yPred', x_axis = 'yTest', yLabel = 'Predicted value', xLabel = 'Groundtruth', labels = ["R² = 0.03; PA = 100 %"])
    # ResidualPlot_Scatter_Combined(SVR_RBF_BLs, displayParams, FORMAT_Values, DB_Values['DBpath'], Blender=True, setyLim=None, setxLim=None,
    #                               y_axis = 'yPred', x_axis = 'yTest', yLabel = 'Predicted value', xLabel = 'Groundtruth', labels = ["R² = 0.10; PA = 100 %"])
    # ResidualPlot_Scatter_Combined(studies_NBest, displayParams, FORMAT_Values, DB_Values['DBpath'], NBest=True, setyLim=None, setxLim=None,
    #                               y_axis = 'yPred', x_axis = 'yTest', yLabel = 'Predicted value', xLabel = 'Groundtruth', labels = ["R² = 0.14; PA = 83 %"])
    #
    # # ResidualPlot_Scatter_Combined(All_CV, displayParams, FORMAT_Values, DB_Values['DBpath'], setyLim=None, setxLim=None,
    # #                               y_axis = 'yPred', x_axis = 'yTest', yLabel = 'Predicted value', xLabel = 'Groundtruth')
    # ResidualPlot_Scatter_Combined(SVR_RBFs, displayParams, FORMAT_Values, DB_Values['DBpath'], setyLim=None, setxLim=None,
    #                               y_axis = 'yPred', x_axis = 'yTest', yLabel = 'Predicted value', xLabel = 'Groundtruth', labels = ["R² = 0.16; PA = 84 %"])
    #

    # META STORE

AccuracyCheck(Blenders_NBest_CV, studyParams['sets'], displayParams, DB_Values['DBpath'], tolerance=PROCESS_VALUES['accuracyTol'])

    # Studies_CV_BlenderNBest.append(Blenders_NBest_CV)

# AccuracyCheck(Studies_CV_BlenderNBest, sets, DB_Values['acronym'], displayParams, DB_Values['DBpath'], tolerance=0.15)




