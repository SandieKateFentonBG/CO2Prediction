from report import *
from dashBoard import *


# pickleDumpMe( DBpath= DB_Values['DBpath'], displayParams = displayParams, obj = RFE_VALUES, name = 'RFE_VALUES')


# pickleLoadMe(path = DB_Values['DBpath'], show = True)


obj = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/TEST_RUN/RECORDS/GSss', show = False)
# print(obj)



# Studies = [LR_, LR_RIDGE, LR_LASSO, LR_ELAST, KRR, SVR]
# for study in Studies:
#     print('')
#     print('')
#     print('MODEL PREDICTOR', study['name'])
#     for learning_df in learning_dfs:
#         name = study['name'] + '_' + learning_df.selector
#         print('')
#         print('FEATURE SELECTION', learning_df.selector)
#
#         MY_GS = ModelGridsearch(name, learningDf=learning_df, modelPredictor=study['modelPredictor'], param_dict=study['param_dict'])
#         print('Study name :', name)
#         print('size :', learning_df.trainDf.shape)
#         print('GS params :', MY_GS.Param)
#         print('GS TrainScore :', MY_GS.TrainScore)
#         print('GS TestScore :', MY_GS.TestScore)
#         print('GS TestAcc :', MY_GS.TestAcc)
#         print('GS TestMSE :', MY_GS.TestMSE)
#         print('GS TestR2 :', MY_GS.TestR2)
#         print('GS Resid - Mean/std :', np.mean(MY_GS.Resid), np.std(MY_GS.Resid))
#         print('GS Resid - Min/Max :', min(MY_GS.Resid), max(MY_GS.Resid))
#         print('GS Resid :', MY_GS.Resid)

LASSO_GS = ModelGridsearch('LR_LASSO', learningDf= RFEs[1], modelPredictor= Lasso(), param_dict = LR_param_grid)
print('1')
print(RFEs[1].selector)
reportModels(DB_Values['DBpath'], displayParams, [LASSO_GS], RFEs[1], objFolder ='Models', display = True)

print('2')
print('LASSO_FS_GS.RFE_RFR', type(LASSO_GS), len(dir(LASSO_GS)), dir(LASSO_GS))
print('LASSO_FS_GS.RFE_DTR.Param', LASSO_GS.Param)
print('LASSO_FS_GS.RFE_DTR.param_dict', LASSO_GS.param_dict)
print('LASSO_FS_GS.RFE_DTR.TestAcc', LASSO_GS.TestAcc)
print('LASSO_FS_GS.RFE_DTR.TrainScore', LASSO_GS.TrainScore)
print('LASSO_FS_GS.RFE_DTR.TestScore', LASSO_GS.TestScore)
print('LASSO_FS_GS.RFE_DTR.TestMSE', LASSO_GS.TestMSE)
print('.selectedLabels', len(LASSO_GS.selectedLabels))
print('.GridMSE', len(LASSO_GS.GridMSE), LASSO_GS.GridMSE)
print('.GridR2', len(LASSO_GS.GridR2), LASSO_GS.GridR2)

print('LASSO_GS', len(dir(LASSO_GS)), dir(LASSO_GS))
LASSO_FS_GS = ModelFeatureSelectionGridsearch(predictorName=LR_LASSO['name'], learningDfs=RFEs,
                                        modelPredictor=LR_LASSO['modelPredictor'], param_dict=LR_LASSO['param_dict'])


print('3')
print('LASSO_FS_GS', len(dir(LASSO_FS_GS)), dir(LASSO_FS_GS))

print('LASSO_FS_GS.RFE_RFR', type(LASSO_FS_GS.RFE_RFR), len(dir(LASSO_FS_GS.RFE_RFR)), dir(LASSO_FS_GS.RFE_RFR))
print('LASSO_FS_GS.RFE_DTR.Param', LASSO_FS_GS.RFE_RFR.Param)
print('LASSO_FS_GS.RFE_DTR.param_dict', LASSO_FS_GS.RFE_RFR.param_dict)
print('LASSO_FS_GS.RFE_DTR.TestAcc', LASSO_FS_GS.RFE_RFR.TestAcc)
print('LASSO_FS_GS.RFE_DTR.TrainScore', LASSO_FS_GS.RFE_RFR.TrainScore)
print('LASSO_FS_GS.RFE_DTR.TestScore', LASSO_FS_GS.RFE_RFR.TestScore)
print('LASSO_FS_GS.RFE_DTR.TestMSE', LASSO_FS_GS.RFE_RFR.TestMSE)
print('.selectedLabels', len(LASSO_FS_GS.RFE_RFR.selectedLabels))
print('.GridMSE', len(LASSO_FS_GS.RFE_RFR.GridMSE), LASSO_FS_GS.RFE_RFR.GridMSE)
print('.GridR2', len(LASSO_FS_GS.RFE_RFR.GridR2), LASSO_FS_GS.RFE_RFR.GridR2)
