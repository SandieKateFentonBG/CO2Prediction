# todo : choose database

#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
from Dashboard_EUCB_FR import *

#SCRIPT IMPORTS
from Main_FS_Steps import *
from Main_GS_FS_Steps import *
from FeatureReport import *
from StudyResiduals import *
from ModelBlending import *
from CombineSHAP import *


# 'CSTB_rd' 'PMV2_rd''CSTB_A123_rd''CSTB_rd'
DBname = DB_Values['acronym'] + yLabels[0] + '_rd' #'CSTB_rd'

#todo : change database link !

studies_GS_FS = []
studies_Blender = []
# randomvalues = list(range(42, 53))
randomvalues = list(range(33, 44))#44

# for value in randomvalues:
#     PROCESS_VALUES['random_state'] = value
#     displayParams["reference"] = DBname + str(PROCESS_VALUES['random_state']) + '/'
#     print('Run Study for random_state:', value)
# #
# #     # RUN
#     rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = Run_FS_Study()
#     GS_FSs, blendModel = Run_GS_FS_Study(DBname + str(PROCESS_VALUES['random_state']) + '/')

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
    # print('blendModel1', type(blendModel1), blendModel1.modelList)
    # print('blendModel2', type(blendModel2), blendModel2.modelList)

    Blender = import_Main_Blender(import_reference)

    # print(Blender)
    # print(Blender.ModelWeights)

    studies_GS_FS.append(GS_FSs)
    studies_Blender.append(Blender)

# PREDICT
# PredictionDict = computePrediction(GS)

#COMBINE
# reportCombinedStudies(studies_Blender, displayParams, DB_Values['DBpath'], random_seeds = randomvalues)
# ResultsDf = ReportStudyResults(studies_GS_FS, displayParams, DB_Values['DBpath'])
#
# RUN_CombinedResiduals(studies_GS_FS, displayParams, FORMAT_Values, DBpath = DB_Values['DBpath'])
RUN_SHAP_Combined(displayParams, DB_Values["DBpath"], studies_Blender, studies_GS_FS, xQuantLabels, xQualLabels)





# combine all residuals
# interpret shap > does this mean i can predict from the top 5?

#blending report for study


# UNDERSTAND RESIDUALS

# Model_List = unpackGS_FSs(GS_FSs) #, remove='KRR_LIN'
# for m in Model_List:
#     print(m.GSName)
#     # print(len(m.Resid), m.Resid)
#     for i in range(len(m.Resid)):
#         if abs(m.Resid[i]) > 300:
#             print(i, 'resid :', m.Resid[i], 'Ypred :', m.yPred[i])
#             print('Y', m.learningDf.yTest.iloc[[i]])
#             # print('X', m.learningDf.XTest.iloc[[i]])
# print('Ypred', m.yPred[i])
# print('resid', m.Resid[i])
# print([elem for elem in m.Resid if abs(elem) > 200])

# 4 runs
# 1. all embodied - all data > CSTB_rd
# 2. only structural - all data > CSTB_fr_GHG_P2_sum_m2_rd35
# 3. all embodied - resid data > CSTB_residential_rd
# 2. only structural - resid data


# nBestModels = selectnBestModels(GS_FSs, sortedModelsData, n=10, checkR2 = True)

# Questions:

# Models selected differ strongly - why ?
# results differ between runs
# blender scores worse sometimes
# wat des CV do? > since i do a manual CV > how should I extract 1 value?
# Gridsearch CV > extract best estimator > do the same for my CV?
# What does negative R2 on test mean - if accuracy still fine?
#
# > SHAP > only use best model and concatenate ? Concatenate all? use blender?
# SHAP with CV
# https://lucasramos-34338.medium.com/visualizing-variable-importance-using-shap-and-cross-validation-bd5075e9063a
# https://stats.stackexchange.com/questions/435944/best-way-to-assess-shap-values-variability
# calculate the variance of the rank of importance for each variable, then take the mean of the rank variance across all variables. So if the rank of variables change a lot I can trust them less.
# consider SHAP waterfall for single example








