from Main_Samp_Steps import *

#REFERENCE

# for set in studyParams['sets']:
#     yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set
#     DBname = DB_Values['acronym'] + '_' + yLabelsAc + '_rd'
#
#     for value in studyParams['randomvalues']:
#         PROCESS_VALUES['random_state'] = value

yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = studyParams['sets'][0]
displayParams["reference"] = DB_Values['acronym'] + '_' + yLabelsAc + '_rd' + str(PROCESS_VALUES['random_state']) + '/'

print(displayParams["reference"])

# MODEL
GS_FSs = import_Main_GS_FS(displayParams["reference"], GS_FS_List_Labels = studyParams['Regressors'])
# Model_List_All = unpackGS_FSs(GS_FSs, remove='')
LRidge = GS_FSs[1].RFE_RFR

Blender = import_Main_Blender(displayParams["reference"], n = BLE_VALUES['NCount'], NBestScore = BLE_VALUES['NBestScore'], label = BLE_VALUES['Regressor'] + '_Blender')
B_M = Blender.modelList


Run_Model_Predictions_Explainer(MyPred_Sample, DB_Values["DBpath"], Model_List=B_M + [LRidge], Blender_List=[Blender], precomputed = False)
Run_Feature_Predictions_2D(MyPred_Sample, feature1='Structure', feature2='Main_Material', Model_List=B_M + [LRidge], Blender_List=[Blender])


# pickleDumpMe(DB_Values['DBpath'], displayParams, predDf, 'PREDICTIONS', MyPred_Sample["DBname"])
# predDf = pickleLoadMe(path=path, show=False)