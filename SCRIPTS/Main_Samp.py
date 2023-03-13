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
Blender_SVR = import_Blender_NBest(ref_single, label='LR_RIDGE' + '_Blender_NBest')
Blender_LR = import_Blender_NBest(ref_single, label='SVR_RBF' + '_Blender_NBest')

# # MODEL
# GS_FSs = import_Main_GS_FS(displayParams["reference"], GS_FS_List_Labels = studyParams['Regressors'])
# # Model_List_All = unpackGS_FSs(GS_FSs, remove='')
# LRidge = GS_FSs[1].RFE_RFR
#
# Blender_NBest = import_Blender_NBest(ref_single, label = BLE_VALUES['Regressor'] + '_Blender_NBest')

# Blender_NBest = import_Main_Blender(displayParams["reference"], n = BLE_VALUES['NCount'], NBestScore = BLE_VALUES['NBestScore'], label =BLE_VALUES['Regressor'] + '_Blender')
# B_M = Blender_NBest.modelList
B_M = Blender_SVR.modelList

Run_Model_Predictions_Explainer(MyPred_Sample, DB_Values["DBpath"], Model_List=B_M, Blender_List=[Blender_SVR]+[Blender_LR], precomputed = False)
Run_Feature_Predictions_2D(MyPred_Sample, feature1='Structure', feature2='Main_Material', Model_List=B_M, Blender_List=[Blender_SVR]+[Blender_LR])


# pickleDumpMe(DB_Values['DBpath'], displayParams, predDf, 'PREDICTIONS', MyPred_Sample["DBname"])
# predDf = pickleLoadMe(path=path, show=False)