import pandas as pd

def sortedModels(GS_FSs, NBestScore='TestR2'):  # 'TestAcc' #todo : the score was changed from TestAcc to TestR2
    # sorting key = 'TestAcc' , last in list
    keys = ['predictorName', 'selectorName', NBestScore]

    allModels = []

    for i in range(len(GS_FSs)):
        indexPredictor = i
        GS_FS = GS_FSs[i]
        for j in range(len(GS_FS.learningDfsList)):
            indexLearningDf = j
            DfLabel = GS_FS.learningDfsList[j]
            GS = GS_FS.__getattribute__(DfLabel)
            v = [GS.__getattribute__(keys[k]) for k in range(len(keys))]
            v.append(indexPredictor)
            v.append(indexLearningDf)
            # v.append(GS)

            allModels.append(v)

    sortedModelsData = sorted(allModels, key=lambda x: x[-3], reverse=True)

    return sortedModelsData


def selectnBestModels(GS_FSs, sortedModelsData, n=10, checkR2=True):
    nBestModels = []

    if checkR2:  # ony take models with positive R2

        count = 0

        # while len(nBestModels) < n:
        for data in sortedModelsData:  # data =['predictorName', 'selectorName', score, indexPredictor, indexLearningDf]
            predictor = GS_FSs[data[3]]
            DfLabel = predictor.learningDfsList[data[4]]
            selector = predictor.__getattribute__(DfLabel)
            if selector.TestScore > 0 and selector.TrainScore > 0:
                nBestModels.append(selector)
        nBestModels = nBestModels[0:n]

        if len(nBestModels) == 0:  # keep n best models if all R2 are negative
            print('nbestmodels selected with negative R2')

            for data in sortedModelsData[
                        0:n]:  # data =['predictorName', 'selectorName', score, indexPredictor, indexLearningDf]
                predictor = GS_FSs[data[3]]
                DfLabel = predictor.learningDfsList[data[4]]
                selector = predictor.__getattribute__(DfLabel)

                nBestModels.append(selector)

    else:

        for data in sortedModelsData[
                    0:n]:  # data =['predictorName', 'selectorName', score, indexPredictor, indexLearningDf]
            predictor = GS_FSs[data[3]]
            DfLabel = predictor.learningDfsList[data[4]]
            selector = predictor.__getattribute__(DfLabel)

            nBestModels.append(selector)

    return nBestModels


def reportGS_Scores_Blending(blendModel, displayParams, DBpath, NBestScore, NCount):
    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        index = [model.GSName for model in blendModel.modelList] + [blendModel.GSName]
        columns = ['TrainScore', 'TestScore', 'TestMSE', 'TestR2', 'TestAcc', 'ResidMean', 'ResidVariance',
                   'ModelWeights']  #
        BlendingDf = pd.DataFrame(columns=columns, index=index)
        for col in columns[:-1]:
            BlendingDf[col] = [model.__getattribute__(col) for model in blendModel.modelList] + [
                blendModel.__getattribute__(col)]
        BlendingDf['ModelWeights'] = [round(elem, 3) for elem in list(blendModel.ModelWeights)] + [
            0]  # todo : this naming was changed from ModelWeights
        sortedDf = BlendingDf.sort_values('ModelWeights', ascending=False)

        AllDfs = [BlendingDf, sortedDf]
        sheetNames = ['Residuals_MeanVar', 'Sorted_Residuals_MeanVar']

        if NCount:  # this is a BestmodelBlender
            with pd.ExcelWriter(outputPathStudy + reference[:-1] + "_GS_Scores_NBest" + '_' + str(
                    NCount) + '_' + NBestScore + '_' + blendModel.GSName + ".xlsx", mode='w') as writer:
                for df, name in zip(AllDfs, sheetNames):
                    df.to_excel(writer, sheet_name=name)
        else:  # this is a CV Blender
            with pd.ExcelWriter(outputPathStudy + reference[
                                                  :-1] + "_CV_Scores_"  + NBestScore + '_' + blendModel.GSName + ".xlsx",
                                mode='w') as writer:
                for df, name in zip(AllDfs, sheetNames):
                    df.to_excel(writer, sheet_name=name)