from Dashboard_EUCB_FR_v2 import *

from Main_FS_Steps import *
from Main_GS_FS_Steps import *
from Main_Combine_Steps import *
from CV_Blending import *



yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = studyParams['sets'][0]
displayParams["reference"] = DB_Values['acronym'] + '_' + yLabelsAc + '_rd' + str(PROCESS_VALUES['random_state']) + '/'
print("Creating learningdf for :", set)
PROCESS_VALUES['random_state'] = 32
DBname = DB_Values['acronym'] + '_' + yLabelsAc + '_rd'

#CONSTRUCT BLENDER TRAINING TESTING SET
# CVrdat, CVdf, CVlearningDf, CVbaseFormatedDf = Run_Data_Processing()
CVref = DBname + str(PROCESS_VALUES['random_state']) + '/'

#IMPORT BLENDER TRAINING TESTING SET
CVrdat, CVdf, CVlearningDf, CVbaseFormatedDf = import_Processed_Data(CVref, show=False)

#QUERRY MODELS TO BLEND
SVR_List = []
randomvalues = studyParams['randomvalues']

for value in randomvalues:
    PROCESS_VALUES['random_state'] = value
    displayParams["reference"] = DBname + str(PROCESS_VALUES['random_state']) + '/'

    print('Import Study for random_state:', value)
    import_reference = displayParams["reference"]

    GS_FSs = import_Main_GS_FS(import_reference, GS_FS_List_Labels=['SVR_RBF'])

    # STORE
    SVR_Mod = GS_FSs[0].RFE_RFR
    SVR_List.append(SVR_Mod)

# BLEND
CV_Blender = Run_CV_Blending(SVR_List, CVbaseFormatedDf, displayParams, DB_Values["DBpath"],  NBestScore ='TestR2' , ConstructorKey = 'LR_RIDGE', Gridsearch = True)

