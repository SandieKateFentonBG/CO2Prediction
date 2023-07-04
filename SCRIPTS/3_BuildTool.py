from Main_Samp_Steps import *
from Main_BL_Steps import *

yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = studyParams['sets'][0]

displayParams["ref_prefix"] = acronym + '_' + studyParams['sets'][0][1]
displayParams["reference"] = displayParams["ref_prefix"] + '_rd' + str(PROCESS_VALUES['random_state']) + '/'

Blender_SVR = import_Blender_NBest(displayParams["reference"], label='SVR_RBF' + '_Blender_NBest')
Blender_LR = import_Blender_NBest(displayParams["reference"], label='LR_RIDGE' + '_Blender_NBest')

B_M = Blender_LR.modelList

for s in [MyPred_Sample_SELECTION, MyPred_Sample_CONCRETE, MyPred_Sample_TIMBER, MyPred_Sample_GLT]: #

    print(">>>", s)

    sample = RUN_Samp_Steps(s, DBpath=DB_Values["DBpath"], ref_single = displayParams["reference"], Model_List=B_M, Blender_List=[Blender_SVR]+[Blender_LR], precomputed = False)


# RE-RUN WHOLE SCRIPT WITH IASS DATA/UPDATE DATA
# SAME FOR PMV3
# SAME FOR CSTB
# SAME FOR ALL 3 WITH MLP REG + DECISION TREES