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

#
# Studies_CV_BlenderNBest = []
#
# for set in studyParams['sets']:
#     yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set
#
#     print("Study for :", set)
#
#     DBname = DB_Values['acronym'] + '_' + yLabelsAc + '_rd'
#     All_CV, NBest_CV, Filters_Pearson_CV, Filters_Spearman_CV, Blenders_NBest_CV, Blenders_Single_CV = [],[],[],[],[],[]
#     randomvalues = studyParams['randomvalues']
#
#     for value in randomvalues:
#
#         PROCESS_VALUES['random_state'] = value
#         print('Import Study for random_state:', value)
#
#         # "IMPORT"
#
#         ref_prefix = DB_Values['acronym'] + '_' + studyParams['sets'][0][1]
#         ref_suffix_single = '_rd'
#         ref_suffix_combined = '_Combined/'
#         ref_single = ref_prefix + ref_suffix_single + str(PROCESS_VALUES['random_state']) + '/'
#         ref_combined = ref_prefix + ref_suffix_combined
#         displayParams["ref_prefix"] = ref_prefix
#         displayParams["reference"] = ref_single
#
#         # FEATURE PROCESSING
#         # rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(ref_single, show = False)
#
#         # MODEL PROCESSING
#         # GS_FSs = import_Main_GS_FS(ref_single, GS_FS_List_Labels = studyParams['Regressors'])
#         # # GS_FSs = import_Main_GS_FS(import_reference, GS_FS_List_Labels=['KRR_LIN'])
#         #
#         # # NBEST PROCESSING
#         # NBestModels = import_NBest(ref_single)
# #
# #         # # BLENDER PROCESSING
#         Blender_NBest = import_Blender_NBest(ref_single, label = BLE_VALUES['Regressor'] + '_Blender_NBest')
#
# #
# #         # "RUN"
# #         #
# #         # FEATURE PROCESSING
# #         # rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = Run_FS_Study()
# #         #
# #         # MODEL PROCESSING & NBEST PROCESSING
# #         # GS_FSs, NBestModels = Run_GS_FS_Study(DBname + str(PROCESS_VALUES['random_state']) + '/', importMainGSFS=False)
# #         #
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
#         Blenders_NBest_CV.append(Blender_NBest)
#
# #
# #     # COMBINE
# #     RUN_CV_Report(All_CV, NBest_CV, Blenders_NBest_CV, Filters_Spearman_CV, Filters_Pearson_CV, randomvalues,displayParams, GSName = "LR")
#     report_BL_NBest_CV(Blenders_NBest_CV, displayParams,  DB_Values['DBpath'], randomvalues)
    # plotCVResidualsGaussian_indiv(All_CV, displayParams, FORMAT_Values, DB_Values['DBpath'],
    #                               studyFolder='GaussianPlot_indivModels', studies_Blender = Blenders_NBest_CV)
    # plotCVResidualsGaussian_Combined(Blenders_NBest_CV, displayParams, FORMAT_Values, DB_Values['DBpath'],
    #                                  studyFolder='GaussianPlot_Blender_NBest_' + str(BLE_VALUES['NCount']) + '_' + BLE_VALUES['NBestScore'], Blender=True)

# META STORE
    # Studies_CV_BlenderNBest.append(CV_BlenderNBest)

# AccuracyCheck(Studies_CV_BlenderNBest, sets, DB_Values['acronym'], displayParams, DB_Values['DBpath'], tolerance=0.15)





ref_prefix = DB_Values['acronym'] + '_' + studyParams['sets'][0][1]
ref_suffix_single = 'rd'
ref_suffix_combined = '_Combined/'
# ref_single = ref_prefix + ref_suffix_single + str(PROCESS_VALUES['random_state']) + '/'
ref_combined = ref_prefix + ref_suffix_combined
displayParams["ref_prefix"] = ref_prefix
# displayParams["reference"] = ref_single

#

Run_Blending_CV(displayParams, DB_Values['DBpath'], ref_prefix, ConstructorKey = BLE_VALUES['Regressor'],
                GS_FS_List_Labels=['LR', 'LR_RIDGE', 'KRR_RBF', 'KRR_LIN', 'KRR_POL',
                                   'SVR_RBF', 'SVR_LIN'],
                GS_name_list=['LR_fl_spearman', 'LR_fl_pearson', 'LR_RFE_RFR', 'LR_RFE_DTR', 'LR_RFE_GBR',
                              'LR_NoSelector',
                              'LR_RIDGE_fl_spearman', 'LR_RIDGE_fl_pearson', 'LR_RIDGE_RFE_RFR', 'LR_RIDGE_RFE_DTR',
                              'LR_RIDGE_RFE_GBR', 'LR_RIDGE_NoSelector', 'KRR_LIN_fl_spearman', 'KRR_LIN_fl_pearson',
                              'KRR_LIN_RFE_RFR',
                              'KRR_LIN_RFE_DTR', 'KRR_LIN_RFE_GBR', 'KRR_LIN_NoSelector', 'KRR_RBF_fl_spearman',
                              'KRR_RBF_fl_pearson',
                              'KRR_RBF_RFE_RFR', 'KRR_RBF_RFE_DTR', 'KRR_RBF_RFE_GBR', 'KRR_RBF_NoSelector',
                              'KRR_POL_fl_spearman',
                              'KRR_POL_fl_pearson', 'KRR_POL_RFE_RFR', 'KRR_POL_RFE_DTR', 'KRR_POL_RFE_GBR',
                              'KRR_POL_NoSelector',
                              'SVR_LIN_fl_spearman', 'SVR_LIN_fl_pearson', 'SVR_LIN_RFE_RFR', 'SVR_LIN_RFE_DTR',
                              'SVR_LIN_RFE_GBR',
                              'SVR_LIN_NoSelector', 'SVR_RBF_fl_spearman', 'SVR_RBF_fl_pearson', 'SVR_RBF_RFE_RFR',
                              'SVR_RBF_RFE_DTR',
                              'SVR_RBF_RFE_GBR', 'SVR_RBF_NoSelector'],
                      single=False, predictor='SVR_RBF', ft_selector='RFE_GBR', runBlending = True)
#
# # # Could I integrate WEC to features and predict SEC with it?
# # # could I do a 2 step ML 1. predict WEC 2. Predict SEC with WEC
#
# # my train score is super low > why??? change this ! maybe LOO?
#
#
#
