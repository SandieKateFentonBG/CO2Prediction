from Model import *

def AccuracyCheck(Studies_CV_BlenderNBest, sets, name, displayParams, DBpath, tolerance=0.15):

    import pandas as pd
    AllDfs = []
    for CV_BlenderNBest in Studies_CV_BlenderNBest:
        for BlenderNBest in CV_BlenderNBest:

            ModelDfs = []
            for Model in BlenderNBest.modelList:
                yPred, PredictionDict = computePrediction(Model)
                yTrue, yPred = PredictionDict['yTest'], PredictionDict['yPred']
                SampleAcc = abs((yPred - yTrue) / yTrue)
                SampleAccBool = [1 if SampleAcc[i] < tolerance else 0 for i in range(len(yTrue))]

                AccuracyDf_Model = pd.DataFrame(columns=[i for i in range(len(PredictionDict['yTest']))],
                                           index=[Model.GSName + '_' + l for l in ['yTest', 'yPred', 'Resid', 'SampleAcc','SampleAccBool']])

                AccuracyDf_Model.loc[Model.GSName + '_' +'yTest', :] = PredictionDict['yTest']
                AccuracyDf_Model.loc[Model.GSName + '_' +'yPred', :] = PredictionDict['yPred']
                AccuracyDf_Model.loc[Model.GSName + '_' +'Resid', :] = PredictionDict['Resid']
                AccuracyDf_Model.loc[Model.GSName + '_' +'SampleAcc', :] = SampleAcc
                AccuracyDf_Model.loc[Model.GSName + '_' +'SampleAccBool', :] = SampleAccBool
                AccuracyDf_Model.loc[:, 'Mean'] = AccuracyDf_Model.abs().mean(axis=1) #todo : check if abs makes sense
                AccuracyDf_Model.loc[:, 'Stdv'] = AccuracyDf_Model.abs().std(axis=1)

                ModelDfs.append(AccuracyDf_Model)

            CombinedDf = pd.concat(ModelDfs, axis=0)

        AllDfs.append(CombinedDf)

        if displayParams['archive']:
            import os

            path, folder, subFolder = DBpath, "RESULTS/", name + '/'
            outputPathStudy = path + folder + subFolder

            if not os.path.isdir(outputPathStudy):
                os.makedirs(outputPathStudy)

            sheetNames = [sets[i][1] + '_' + sets[i][2] for i in range(len(sets))]  # 4

            print('outputPathStudy', outputPathStudy)
            with pd.ExcelWriter(outputPathStudy + "AccuracyCheck" + name + ".xlsx", mode='w') as writer:
                for df, sh in zip(AllDfs, sheetNames):
                    df.to_excel(writer, sheet_name=sh, freeze_panes=(0, 1))





