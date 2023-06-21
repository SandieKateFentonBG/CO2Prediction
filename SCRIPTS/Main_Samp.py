from Main_Samp_Steps import *
from Main_BL_Steps import *

#REFERENCE

# for set in studyParams['sets']:
#     yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set
#     DBname = DB_Values['acronym'] + '_' + yLabelsAc + '_rd'
#
#     for value in studyParams['randomvalues']:
#         PROCESS_VALUES['random_state'] = value

yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = studyParams['sets'][0]



ref_prefix = DB_Values['acronym'] + '_' + studyParams['sets'][0][1]
ref_suffix_single = '_rd'
ref_suffix_combined = '_Combined/'
ref_combined = ref_prefix + ref_suffix_combined
displayParams["ref_prefix"] = ref_prefix
ref_single = ref_prefix + ref_suffix_single + str(PROCESS_VALUES['random_state']) + '/'
displayParams["reference"] = ref_single


print(displayParams["reference"])

# ">>IMPORT"

# # BLENDER PROCESSING
# Blender_NBest = import_Blender_NBest(ref_single, label=BLE_VALUES['Regressor'] + '_Blender_NBest')
Blender_SVR = import_Blender_NBest(ref_single, label='SVR_RBF' + '_Blender_NBest')
Blender_LR = import_Blender_NBest(ref_single, label='LR_RIDGE' + '_Blender_NBest')
#
# # # MODEL
# # GS_FSs = import_Main_GS_FS(displayParams["reference"], GS_FS_List_Labels = studyParams['Regressors'])
# # # Model_List_All = unpackGS_FSs(GS_FSs, remove='')
# # LRidge = GS_FSs[1].RFE_RFR
# #
# # Blender_NBest = import_Blender_NBest(ref_single, label = BLE_VALUES['Regressor'] + '_Blender_NBest')
#
# # Blender_NBest = import_Main_Blender(displayParams["reference"], n = BLE_VALUES['NCount'], NBestScore = BLE_VALUES['NBestScore'], label =BLE_VALUES['Regressor'] + '_Blender')
# # B_M = Blender_NBest.modelList
B_M = Blender_LR.modelList


for s in [MyPred_Sample_CONCRETEselection, MyPred_Sample_CONCRETE, MyPred_Sample_TIMBER, MyPred_Sample_GLT]: #
    sample = RUN_Samp_Steps(s, DBpath=DB_Values["DBpath"], ref_single = ref_single, Model_List=B_M, Blender_List=[Blender_SVR]+[Blender_LR], precomputed = False)
    print(sample.SHAPGroupKeys)
    print(sample.SHAPGroupvalues)

# regressor = import_Main_GS_FS(ref_single, GS_FS_List_Labels=['LR_ELAST'])  # SVR_RBF
# print(regressor)
# for s in regressor[0].learningDfsList:
#     print(s)
#     model = regressor[0].__getattribute__(s)
#     for k, v in model.SHAPGroup_RemapDict.items():
#         print(k, v)
#     print('')



