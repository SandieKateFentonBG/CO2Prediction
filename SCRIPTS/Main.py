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
from StudyResiduals import *

# DBname = DB_Values['acronym'] + '_rd'

Studies_CV_BlenderNBest = []

# sets = [
#     ['Embodied_Carbon[kgCO2e_m2]','EC','TestR2'],
#     ['Embodied_Carbon[kgCO2e_m2]','EC','TestAcc'],
#     ['Embodied_Carbon_Structure[kgCO2e_m2]','ECS', 'TestR2'],
#     ['Embodied_Carbon_Structure[kgCO2e_m2]','ECS','TestAcc']]

sets = [['Embodied_Carbon[kgCO2e_m2]','EC','TestR2']]

for set in sets:
    yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set

    print("Study for :", set)

    DBname = DB_Values['acronym'] + '_' + yLabelsAc + '_rd'

    CV_AllModels = []
    CV_BlenderNBest = []

    randomvalues = list(range(48, 49))

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
        # GS_FSs, blendModel = Run_GS_FS_Study(DBname + str(PROCESS_VALUES['random_state']) + '/', ConstructorKey = 'LR_RIDGE', importMainGSFS = True, BlendingOnVal = False)
        GS_FSs = import_Main_GS_FS(import_reference, GS_FS_List_Labels = ['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST',  'KRR_RBF', 'KRR_LIN','KRR_POL','SVR_LIN', 'SVR_RBF'])#

        # Model_List = unpackGS_FSs(GS_FSs, remove='')

        Blender = import_Main_Blender(import_reference, n = BLE_VALUES['NCount'], NBestScore = BLE_VALUES['NBestScore'], label = 'LR_RIDGE_Blender') #blendModel.GSName

        GS_WeightsSummaryPlot_NBest(Blender, target=FORMAT_Values['targetLabels'], displayParams=displayParams,
                              DBpath=DB_Values['DBpath'], content='GS_FSs', sorted=True, yLim=4,
                              df_for_empty_labels=baseFormatedDf.trainDf, fontsize=14, studyFolder='GS_FS/')

        print(Blender)
        print(Blender.GSName)



        # reportGS_FeatureWeights(DB_Values['DBpath'], displayParams, GS_FSs, blender=Blender)


        CV_AllModels.append(GS_FSs)
        CV_BlenderNBest.append(Blender)

    Studies_CV_BlenderNBest.append(CV_BlenderNBest)

    # PREDICT
    # computePrediction_NBest(CV_BlenderNBest)
    # PredictionDict = computePrediction(GS)

    # COMBINE
    #
    reportCV_Scores_NBest(CV_BlenderNBest, displayParams, DB_Values['DBpath'], n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'], random_seeds = randomvalues)
    # ResultsDf = reportCV_ScoresAvg_All(CV_AllModels, displayParams, DB_Values['DBpath'])
    # reportCV_ModelRanking_NBest(CV_AllModels, CV_BlenderNBest, seeds = randomvalues, displayParams = displayParams, DBpath =DB_Values['DBpath'], n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'])
    #
    # RUN_SHAP_Combined(displayParams, DB_Values["DBpath"], CV_BlenderNBest, CV_AllModels, xQuantLabels, xQualLabels, n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'], randomValues = randomvalues)
    RUN_CombinedResiduals(CV_AllModels, CV_BlenderNBest, displayParams, FORMAT_Values, DBpath = DB_Values['DBpath'], n= BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'])
    # #

# AccuracyCheck(Studies_CV_BlenderNBest, sets, DB_Values['acronym'], displayParams, DB_Values['DBpath'], tolerance=0.15)


    # plotCVResidualsGaussian(studies_Blender, displayParams, FORMAT_Values, DBpath,
    #                         studyFolder='GaussianPlot_indivModels', NBest=True)




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
#todo : there was a name change from Weights to Model Weights > changes might have been done wrong > could generate errors



# GS_FSs_alpha = import_Main_GS_FS(import_reference, GS_FS_List_Labels = ['LR_RIDGE', 'LR_LASSO', 'LR_ELAST',  'KRR_RBF', 'KRR_LIN','KRR_POL'])#
# GS_FSs_gamma = import_Main_GS_FS(import_reference, GS_FS_List_Labels = ['KRR_RBF', 'KRR_LIN','KRR_POL','SVR_LIN', 'SVR_RBF'])#
# GS_FSs_degree = import_Main_GS_FS(import_reference, GS_FS_List_Labels = ['KRR_POL'])#
# GS_FSs_epsilon = import_Main_GS_FS(import_reference, GS_FS_List_Labels = ['SVR_LIN', 'SVR_RBF'])#
# GS_FSs_C = import_Main_GS_FS(import_reference, GS_FS_List_Labels = ['SVR_LIN', 'SVR_RBF'])#
# GS_FSs_coef0 = import_Main_GS_FS(import_reference, GS_FS_List_Labels=['KRR_POL'])  #
#
# for ML, key, log, maxVal, absVal in zip([GS_FSs_alpha, GS_FSs_gamma, GS_FSs_degree, GS_FSs_epsilon, GS_FSs_C, GS_FSs_coef0],
#                                 ['alpha', 'gamma', 'degree', 'epsilon', 'C', 'coef0'],
#                                 [True, True, False, True, True, True],
#                               [True, True, True,True, True, True], [True, True, True,True, True, True]) :
#     Model_List = unpackGS_FSs(ML, remove='')
#
#     ParameterPlot2D(Model_List, displayParams, DB_Values['DBpath'], yLim=None,
#                     paramKey=key, score='mean_test_r2', log=log, studyFolder='GS/')
#     ParameterPlot3D(Model_List, displayParams, DB_Values['DBpath'],
#                     colorsPtsLsBest=['b', 'g', 'c', 'y'], paramKey=key, score='mean_test_r2',
#                     size=[6, 6], showgrid=False, log=log, maxScore=maxVal, absVal=absVal, ticks=False, lims=False,
#                     studyFolder='GS/')










