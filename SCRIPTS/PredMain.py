from Predict import *


def Run_Prediction(Model_List, MyPred_Sample, ArchPath):

    sample = Sample(displayParams["reference"], MyPred_Sample)

    pickleDumpMe(ArchPath, displayParams, sample, 'PREDICTIONS', MyPred_Sample["DBname"])
    predDict = dict()
    for model in Model_List:
        print(model.GSName)
        print(model.SHAPexplainer)
        print("model.SHAPexplainer")
        # showAttributes(model.SHAPexplainer)

        pred = sample.SamplePrediction(model)
        predDict[model.GSName] = pred

        sample.SHAP_WaterfallPlot(model, DB_Values['DBpath'])
        sample.SHAP_ScatterPlot(model, DB_Values['DBpath'])
        sample.SHAP_ForcePlot(model, DB_Values['DBpath'])

    predDf = pd.DataFrame.from_dict(predDict).T
    SheetNames = ['Input', 'Predictions']
    AllDfs = [sample.input.T, predDf]

    if displayParams['archive']:

        import os
        reference = displayParams['reference']
        outputPathStudy = DB_Values['DBpath'] + "RESULTS/" + reference + 'RECORDS/' + 'PREDICTIONS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)


        with pd.ExcelWriter(outputPathStudy + sample.name + '_Pred_Records_All' + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, SheetNames):
                df.to_excel(writer, sheet_name=name)

        # with pd.ExcelWriter(outputPathStudy + sample.name + '_Pred_Records_All' + ".xlsx", mode='w', if_sheet_exists="overlay") as writer:
        #     sample.input.T.to_excel(writer, sheet_name="Sheet1")
        #     predDf.T.to_excel(writer, sheet_name="Sheet1", startrow=len(sample.input.T) + 5)

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
Model_List_All = unpackGS_FSs(GS_FSs, remove='')
Model_List = [Model_List_All[-1]]
Model_List = Model_List_All[:]
LRidge = [GS_FSs[1].RFE_RFR]

Blender = import_Main_Blender(displayParams["reference"], n = BLE_VALUES['NCount'], NBestScore = BLE_VALUES['NBestScore'], label = BLE_VALUES['Regressor'] + '_Blender')
B_M = Blender.modelList

Run_Prediction(LRidge, MyPred_Sample, DB_Values['DBpath'])
