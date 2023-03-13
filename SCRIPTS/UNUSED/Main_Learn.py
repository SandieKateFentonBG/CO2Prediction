# # todo : choose database
#
# #DASHBOARD IMPORT
# # from dashBoard import *
# # from Dashboard_PM_v2 import *
# # from Dashboard_EUCB_FR import *
#
# #SCRIPT IMPORTS
# from Main_GS_FS_Steps import *
# from Main_BL_Steps import *
# from Main_Combine_Steps import *
#
# # Studies_CV_BlenderNBest = []
# #
# # for set in studyParams['sets']:
# #     yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set
# #
# #     print("Study for :", set)
# #
# #     DBname = DB_Values['acronym'] + '_' + yLabelsAc + '_rd'
# #     All_CV, NBest_CV, Filters_Pearson_CV, Filters_Spearman_CV, Blenders_NBest_CV, Blenders_Single_CV = [],[],[],[],[],[]
# #     randomvalues = studyParams['randomvalues']
# #
# #     for value in randomvalues:
# #
# #         PROCESS_VALUES['random_state'] = value
# #         print('Import Study for random_state:', value)
# #
# #         # "IMPORT"
# #
# #         ref_prefix = DB_Values['acronym'] + '_' + studyParams['sets'][0][1]
# #         ref_suffix_single = '_rd'
# #         ref_suffix_combined = '_Combined/'
# #         ref_single = ref_prefix + ref_suffix_single + str(PROCESS_VALUES['random_state']) + '/'
# #         ref_combined = ref_prefix + ref_suffix_combined
# #         displayParams["ref_prefix"] = ref_prefix
# #         displayParams["reference"] = ref_single
# #
# #         # FEATURE PROCESSING
# #         # rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(ref_single, show = False)
# #
# #         # MODEL PROCESSING
# #         GS_FSs = import_Main_GS_FS(ref_single, GS_FS_List_Labels = studyParams['Regressors'])
#         # # GS_FSs = import_Main_GS_FS(import_reference, GS_FS_List_Labels=['KRR_LIN'])
#         #
#         # # NBEST PROCESSING
#         # NBestModels = import_NBest(ref_single, OverallBest = BLE_VALUES['OverallBest'])
# #
# #         # # BLENDER PROCESSING
# #         Blender_NBest = import_Blender_NBest(ref_single, label = BLE_VALUES['Regressor'] + '_Blender_NBest')
#
# #
# #         # "RUN"
# #         #
# #         # FEATURE PROCESSING
# #         # rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = Run_FS_Study()
# #         #
# #         # MODEL PROCESSING
# #         GS_FSs = Run_GS_FS_Study(DBname + str(PROCESS_VALUES['random_state']) + '/', importMainGSFS=False)
# #         #
#         # NBEST PROCESSING
#
#         # NBestModels = Run_NBest_Study(DBname + str(PROCESS_VALUES['random_state']) + '/', importNBest=False, OverallBest = BLE_VALUES['OverallBest'])
#
#         # BLENDER PROCESSING
#         # Blender_NBest = Run_Blending_NBest(NBestModels.modelList, displayParams, DB_Values['DBpath'], ref_single,
#         #                                    ConstructorKey=BLE_VALUES['Regressor'])
#
#         # "STORE"
#
#         # All_CV.append(GS_FSs)
#         # NBest_CV.append(NBestModels)
#         # Filters_Pearson_CV.append(pearsonFilter)
#         # Filters_Spearman_CV.append(spearmanFilter)
#         # Blenders_NBest_CV.append(Blender_NBest)
#
# #
#     # COMBINE
#     RUN_Combine_Report(All_CV, NBest_CV, Blenders_NBest_CV, Filters_Spearman_CV, Filters_Pearson_CV, randomvalues, displayParams, GSName ="LR")
#     report_BL_NBest_CV(Blenders_NBest_CV, displayParams,  DB_Values['DBpath'], randomvalues)
#     plotCVResidualsGaussian_indiv(All_CV, displayParams, FORMAT_Values, DB_Values['DBpath'],
#                                   studyFolder='GaussianPlot_indivModels', studies_Blender = Blenders_NBest_CV)
#     plotCVResidualsGaussian_Combined(Blenders_NBest_CV, displayParams, FORMAT_Values, DB_Values['DBpath'],
#                                      studyFolder='GaussianPlot_Blender_NBest_' + str(BLE_VALUES['NCount']) + '_' + BLE_VALUES['NBestScore'], Blender=True)
#
# # META STORE
#     # Studies_CV_BlenderNBest.append(CV_BlenderNBest)
#
# # AccuracyCheck(Studies_CV_BlenderNBest, sets, DB_Values['acronym'], displayParams, DB_Values['DBpath'], tolerance=0.15)
#
#
# # BEST MODELS
#
# ref_prefix = DB_Values['acronym'] + '_' + studyParams['sets'][0][1]
# ref_suffix_single = 'rd'
# ref_suffix_combined = '_Combined/'
# ref_single = ref_prefix + ref_suffix_single + str(PROCESS_VALUES['random_state']) + '/'
# ref_combined = ref_prefix + ref_suffix_combined
# displayParams["ref_prefix"] = ref_prefix
# displayParams["reference"] = ref_single
#
#
# ResultsDf = computeCV_Scores_Avg_All(All_CV)
# BestModelNames = find_Overall_Best_Models(ResultsDf, n=BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'])
# reportCV_ScoresAvg_All(ResultsDf, displayParams, DB_Values['DBpath'], NBestScore=BLE_VALUES['NBestScore'])
#
# # GS_FSs = avgModel(All_CV, DB_Values['DBpath'], displayParams)
#
# GS_FSs = import_Main_GS_FS(ref_combined, GS_FS_List_Labels = ['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST', 'KRR_LIN', 'KRR_RBF', 'KRR_POL', 'SVR_LIN','SVR_RBF'])
# # GS_ParameterPlot2D(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', yLim=None, score='TestAcc', studyFolder='GS_FS/', combined = True)
#
# scoreList = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore']
# scoreListMax = [True, False, True, True, True]
# Plot_GS_FS_Scores(GS_FSs, scoreList, scoreListMax, combined = True, plot_all = True)
#
#
# #
# #
# # ref_prefix = DB_Values['acronym'] + '_' + studyParams['sets'][0][1]
# # ref_suffix_single = 'rd'
# # ref_suffix_combined = '_Combined/'
# # # ref_single = ref_prefix + ref_suffix_single + str(PROCESS_VALUES['random_state']) + '/'
# # ref_combined = ref_prefix + ref_suffix_combined
# # displayParams["ref_prefix"] = ref_prefix
# # # displayParams["reference"] = ref_single
# #
# # #
# #
