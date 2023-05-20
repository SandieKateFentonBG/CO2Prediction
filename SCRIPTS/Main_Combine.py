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

#LIBRARY IMPORTS



Studies_CV_BlenderNBest = []

for set in studyParams['sets']:
    yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set

    print("Study for :", set)
    All_CV, NBest_CV, Filters_Pearson_CV, Filters_Spearman_CV, Blenders_NBest_CV, Blenders_Single_CV = [],[],[],[],[],[]
    # KRR_LINs, SVR_RBFs, SVR_LINs, KRR_RBFs, KRR_POLs, LRs, LR_RIDGEs, LR_ELASTs, LR_LASSOs = [], [], [], [], [], [], [], [], []
    LR_RIDGE_BLs, SVR_RBF_BLs, SVR_RBFs = [], [], []
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
        # GS_FSs = import_Main_GS_FS(ref_single, GS_FS_List_Labels = studyParams['Regressors'])

        SVR_RBF = import_Main_GS_FS(ref_single, GS_FS_List_Labels=['SVR_RBF'])


        #
        # # NBEST PROCESSING
        NBestModels = import_NBest(ref_single, OverallBest = BLE_VALUES['OverallBest'])
        #
        # # # BLENDER PROCESSING
        # Blender_NBest = import_Blender_NBest(ref_single, label = BLE_VALUES['Regressor'] + '_Blender_NBest')
        LR_RIDGE_BL = import_Blender_NBest(ref_single, label = 'LR_RIDGE' + '_Blender_NBest')
        SVR_RBF_BL = import_Blender_NBest(ref_single, label = 'SVR_RBF' + '_Blender_NBest')

        # ">>RUN"

        # # NBEST PROCESSING
        # NBestModels = Run_NBest_Study(ref_single, importNBest=False, OverallBest = BLE_VALUES['OverallBest'])
        #
        # # BLENDER PROCESSING
        # Blender_NBest = Run_Blending_NBest(NBestModels.modelList, displayParams, DB_Values['DBpath'], ref_single,
        #                                    ConstructorKey=BLE_VALUES['Regressor'])

        # ">>STORE"
        # All_CV.append(GS_FSs)
        NBest_CV.append(NBestModels)
        # Blenders_NBest_CV.append(Blender_NBest)
        LR_RIDGE_BLs.append(LR_RIDGE_BL)
        SVR_RBF_BLs.append(SVR_RBF_BL)
        SVR_RBFs.append(SVR_RBF)

    # COMBINE

    # RUN_Combine_Report(All_CV, NBest_CV, Blenders_NBest_CV, randomvalues, displayParams)

    studies_GS_FS, studies_NBest, studies_Blender = All_CV, NBest_CV, Blenders_NBest_CV

    #
    # RUN_CombinedResiduals(studies_GS_FS, studies_NBest, studies_Blender, displayParams, FORMAT_Values, DB_Values['DBpath'],
    #                       randomvalues)


    ResidualPlot_Scatter_Combined(LR_RIDGE_BLs, displayParams, FORMAT_Values, DB_Values['DBpath'], Blender=True, setyLim=None, setxLim=None,
                                  y_axis = 'yPred', x_axis = 'yTest', yLabel = 'Predicted value', xLabel = 'Groundtruth', labels = ["R² = 0.03; PA = 100 %"])
    ResidualPlot_Scatter_Combined(SVR_RBF_BLs, displayParams, FORMAT_Values, DB_Values['DBpath'], Blender=True, setyLim=None, setxLim=None,
                                  y_axis = 'yPred', x_axis = 'yTest', yLabel = 'Predicted value', xLabel = 'Groundtruth', labels = ["R² = 0.10; PA = 100 %"])
    ResidualPlot_Scatter_Combined(studies_NBest, displayParams, FORMAT_Values, DB_Values['DBpath'], NBest=True, setyLim=None, setxLim=None,
                                  y_axis = 'yPred', x_axis = 'yTest', yLabel = 'Predicted value', xLabel = 'Groundtruth', labels = ["R² = 0.14; PA = 83 %"])

    # ResidualPlot_Scatter_Combined(All_CV, displayParams, FORMAT_Values, DB_Values['DBpath'], setyLim=None, setxLim=None,
    #                               y_axis = 'yPred', x_axis = 'yTest', yLabel = 'Predicted value', xLabel = 'Groundtruth')
    ResidualPlot_Scatter_Combined(SVR_RBFs, displayParams, FORMAT_Values, DB_Values['DBpath'], setyLim=None, setxLim=None,
                                  y_axis = 'yPred', x_axis = 'yTest', yLabel = 'Predicted value', xLabel = 'Groundtruth', labels = ["R² = 0.16; PA = 84 %"])


    # META STORE

    # Studies_CV_BlenderNBest.append(Blenders_NBest_CV)

# AccuracyCheck(Studies_CV_BlenderNBest, sets, DB_Values['acronym'], displayParams, DB_Values['DBpath'], tolerance=0.15)


#todo : the problem is the learningdf No Selector - yTest has a type pandas.core.frame.DataFrame rather than 'pandas.core.series.Series
# > it has a length of 1 and not of 32 as the other lerningdfs
#todo : this is because in format :
# self.yTest = pd.DataFrame(data=yTest, columns=yLabels)
# vs in filter and wrapper:
# self.yTest = self.testDf[self.yLabel]
# > change the type so that that it has good shapoe > so that ypred and ytest are the same

# make a function trhat replaces in all models > in the format file :
# self.yTrain = self.trainDf[self.yLabel]
# self.yVal = self.valDf[self.yLabel]
# self.yTest = self.testDf[self.yLabel]
# self.yCheck = self.checkDf[self.yLabel]


