from Dashboard_EUCB_FR_v2 import *

from Main_FS_Steps import *
from Main_GS_FS_Steps import *
from Main_Combine_Steps import *
from CV_Blending import *
#
#
#
# yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = studyParams['sets'][0]
# displayParams["reference"] = DB_Values['acronym'] + '_' + yLabelsAc + '_rd' + str(PROCESS_VALUES['random_state']) + '/'
# print("Creating learningdf for :", set)
# PROCESS_VALUES['random_state'] = 32
# DBname = DB_Values['acronym'] + '_' + yLabelsAc + '_rd'
#
# #CONSTRUCT BLENDER TRAINING TESTING SET
# # CVrdat, CVdf, CVlearningDf, CVbaseFormatedDf = Run_Data_Processing()
# CVref = DBname + str(PROCESS_VALUES['random_state']) + '/'
#
# #IMPORT BLENDER TRAINING TESTING SET
# CVrdat, CVdf, CVlearningDf, CVbaseFormatedDf = import_Processed_Data(CVref, show=False)
#
# #QUERRY MODELS TO BLEND
# All_List = []
# SVR_List = []
# randomvalues = studyParams['randomvalues']
#
# for value in randomvalues:
#     PROCESS_VALUES['random_state'] = value
#     displayParams["reference"] = DBname + str(PROCESS_VALUES['random_state']) + '/'
#
#     print('Import Study for random_state:', value)
#     import_reference = displayParams["reference"]
#
# #     #SINGLE
# #     GS_FSs = import_Main_GS_FS(import_reference, GS_FS_List_Labels=['SVR_RBF'])
# #     # STORE
# #     SVR_Mod = GS_FSs[0].RFE_GBR
# #     SVR_List.append(SVR_Mod)
# #
# # # BLEND
# # CV_Blender = Run_CV_Blending(SVR_List, CVbaseFormatedDf, displayParams, DB_Values["DBpath"],  NBestScore ='TestR2' , ConstructorKey = 'LR_RIDGE', Gridsearch = True)
#
#     # MULTIPLE
#     GS_FSs = import_Main_GS_FS(import_reference, GS_FS_List_Labels=['LR', 'LR_RIDGE', 'KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN', 'SVR_RBF'])
#
#     # STORE
#     Model_List = unpackGS_FSs(GS_FSs)
#     All_List.append(Model_List)
#
# All = repackGS_FSs(All_List)
#
#
# # BLEND
# CV_Blender = Run_CV_Blending(All, CVbaseFormatedDf, displayParams, DB_Values["DBpath"],  NBestScore ='TestR2' , ConstructorKey = 'LR_RIDGE', Gridsearch = True)


GS_FS_List_Labels = ['LR', 'LR_RIDGE', 'KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN', 'SVR_RBF']

blendModel_ls = []
for GSName in GS_FS_List_Labels:
    n = GSName
    for FS in ['_RFE_GBR', '_NoSelector', '_RFE_DTR', '_RFE_RFR', '_fl_pearson', '_fl_spearman']:
        path = 'K:/Temp/Sandie/Pycharm/RESULTS/CSTB_res_nf_EC_Combined/RECORDS/CV_BLENDER/LR_RIDGE_Blender_' \
               + n + FS +'_TestR2' + '.pkl'
        print(path)

        Blender = pickleLoadMe(path=path, show=False)

        print('model_loaded')
        blendModel_ls.append(Blender)

reportCV_Scores_NBest(blendModel_ls, displayParams, DB_Values['DBpath'], n=None,
                      NBestScore=BLE_VALUES['NBestScore'], random_seeds=None)