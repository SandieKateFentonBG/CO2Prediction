def reportStudy_GS_FS(displayParams, DB_Values, FORMAT_Values, PROCESS_VALUES, RFE_VALUES, GS_VALUES, rdat, df, learningDf,
                      baseFormatedDf, FiltersLs, RFEs, GSlist, blendModel, GSwithFS = True):

    if displayParams['archive']:

        import os

        reference = displayParams['reference']
        outputPathStudy = DB_Values['DBpath'] + "RESULTS/" + reference + 'RECORDS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        type = 'GS_'
        if GSwithFS:
            type = 'GS_FS_'

        import csv

        name = reference.rstrip(reference[-1])

        with open(outputPathStudy + name + '_Records_' + type + ".csv", 'w', encoding='UTF8', newline='') as e:
            writer = csv.writer(e, delimiter = ";")

            writer.writerow(['INPUT DATA'])
            writer.writerow(['DBname', DB_Values['DBname']])
            writer.writerow(['DBpath', DB_Values['DBpath']])
            writer.writerow(['DB_Values', DB_Values])
            writer.writerow('')
            writer.writerow(['displayParams', displayParams])
            writer.writerow(['xQuali', rdat.xQuali.keys()])
            writer.writerow(['xQuanti', rdat.xQuanti.keys()])
            writer.writerow(['yLabels', rdat.y.keys()])
            writer.writerow(['FORMAT_Values', FORMAT_Values])
            writer.writerow(['PROCESS_VALUES', PROCESS_VALUES])
            writer.writerow(['RFE_VALUES', RFE_VALUES])
            writer.writerow(['GS_VALUES', GS_VALUES])
            writer.writerow('')

            writer.writerow(['PREPROCESSED DATA'])


            writer.writerow(["Full df ", df.shape])
            writer.writerow(["Outliers removed ", learningDf.shape])
            writer.writerow('')
            writer.writerow(['FORMAT'])
            writer.writerow(["train", baseFormatedDf.trainDf.shape])
            writer.writerow(["validate", baseFormatedDf.valDf.shape])
            writer.writerow(["test", baseFormatedDf.testDf.shape])
            writer.writerow([''])
            writer.writerow(['FILTER'])
            for filter in FiltersLs :
                writer.writerow(['FILTER ', filter.method])
                writer.writerow(['LABELS ', filter.trainDf.shape])
                writer.writerow([filter.selectedLabels])
                writer.writerow('')
            writer.writerow(['RFE'])
            for RFE in RFEs:
                writer.writerow(["RFE with  ", RFE.method])
                writer.writerow(["Number of features fixed ", RFE.n_features_to_select])
                writer.writerow(['Selected feature labels ', list(RFE.selectedLabels)])
                writer.writerow(["Score on training ", RFE.rfe_trainScore])
                writer.writerow(["Score on validation ", RFE.rfe_valScore])
                writer.writerow('')

            writer.writerow('')

            writer.writerow(['GRIDSEARCH DATA'])

            keys = ['predictorName', 'selectorName',  'selectedLabels',
                 'param_dict', 'GridR2', 'GridR2Rank',  'GridMSERank',
                 'scoring', 'Index', 'Estimator','Param', 'Weights', 'WeightsScaled', 'SHAPScoreDict', 'SHAPGroupScoreDict',
                 'ResidMean', 'ResidVariance', 'TrainScore', 'TestScore', 'TestMSE', 'TestR2', 'TestAcc'] #'GridMSE',


            writer.writerow(keys)

            if GSwithFS: # then GSlist should be GS_FSs
                allModels = []
                for GS_FS in GSlist:

                    for DfLabel in GS_FS.learningDfsList:
                        GS = GS_FS.__getattribute__(DfLabel)

                        v = [GS.__getattribute__(keys[i]) for i in range(len(keys))]
                        writer.writerow(v)
                        allModels.append(v)

                sortedModels_Acc = sorted(allModels, key=lambda x: x[-1], reverse=True)
                sortedModels_MSE = sorted(allModels, key=lambda x: x[-3], reverse=True)


            else : # then GSlist should be GSs
                allModels = []
                for Model in GSlist:
                    v = [Model.__getattribute__(keys[i]) for i in range(len(keys))]
                    writer.writerow(v)
                    allModels.append(v)
                # sortedModels = sorted(allModels, key=lambda x: x[-1], reverse=True)
                sortedModels_Acc = sorted(allModels, key=lambda x: x[-1], reverse=True)
                sortedModels_MSE = sorted(allModels, key=lambda x: x[-3], reverse=True)

            writer.writerow('')

            writer.writerow(['SORTED GRIDSEARCH DATA - Acc'])
            writer.writerow(keys)
            for elem in sortedModels_Acc:
                writer.writerow(elem)

            writer.writerow(['SORTED GRIDSEARCH DATA - MSE'])
            writer.writerow(keys)
            for elem in sortedModels_MSE:
                writer.writerow(elem)

            writer.writerow('')

        e.close()


def reportCV_Scores_NBest(studies_Blender, displayParams, DBpath, random_seeds = None):

    import pandas as pd
    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'RECORDS/'
        outputPathStudy = path + folder + subFolder

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        AllDfs = []

        for blendModel in studies_Blender:
            if random_seeds:
                sheetNames = [str(elem) for elem in random_seeds]
            else :
                sheetNames = [str(elem) for elem in list(range(len(studies_Blender)))]

            index = [model.GSName for model in blendModel.modelList] + [blendModel.GSName]
            columns = ['TrainScore', 'TestScore', 'TestMSE', 'TestR2', 'TestAcc', 'ResidMean', 'ResidVariance','ModelWeights'] #
            BlendingDf = pd.DataFrame(columns=columns, index=index)
            for col in columns[:-1]:
                BlendingDf[col] = [model.__getattribute__(col) for model in blendModel.modelList] + [blendModel.__getattribute__(col)]
            BlendingDf['ModelWeights'] = [round(elem,3) for elem in list(blendModel.ModelWeights)] + [0]

            AllDfs.append(BlendingDf)
            with pd.ExcelWriter(outputPathStudy + reference[:-6] + "_CV_Scores_NBest" + ".xlsx", mode='w') as writer:
                for df, name in zip(AllDfs, sheetNames):
                    df.to_excel(writer, sheet_name=name)

def reportCV_CV_ModelRanking_NBest(CV_AllModels, CV_BlenderNBest, seeds, displayParams, DBpath):

    import pandas as pd

    horizTitles = []
    horizLabels = []
    horizValues = []

    vertiLabels = []

    #query 54 model names for columns
    for predictor in CV_AllModels[0]:
        for learningDflabel in predictor.learningDfsList:
            Model = predictor.__getattribute__(learningDflabel)
            name = Model.GSName  # LR_RFR
            vertiLabels.append(name) #verti = models

    for BlenderNBest, seed in zip(CV_BlenderNBest, seeds): #10studies
        horizTitle = seed #ex : 38
        horizLabel = []
        horizValue = []
        for Model in BlenderNBest.modelList: #10best
            name = Model.GSName
            acc = Model.TestAcc
            horizLabel.append(name) #ex : [LR_RFR, LR_DTR, LR_GBR]
            horizValue.append(acc) #ex : [0.8, 0.8, 0.8]
        horizTitles.append(horizTitle)#ex : [38 38 38]
        horizLabels.append(horizLabel) #ex : [[LR_RFR, LR_DTR, LR_GBR][LR_RFR, LR_DTR, LR_GBR][LR_RFR, LR_DTR, LR_GBR]]
        horizValues.append(horizValue) #ex : [[0.8, 0.8, 0.8][0.8, 0.8, 0.8][0.8, 0.8, 0.8]]

    # create empty dfs
    ScoresDf = pd.DataFrame(columns=horizTitles, index=vertiLabels)

    for i in range(len(horizLabels)): #col par col #ex i = 4 : [[Gifa, floors_bg, structure][Gifa, floors_bg, structure][Gifa, floors_bg, structure][Gifa, floors_bg, structure]]
        for j in range(len(horizLabels[i])): #ex : j = 3
            ScoresDf.loc[[horizLabels[i][j]], [horizTitles[i]]] = horizValues[i][j] #ex : HAPDf.loc[[structure], [LR_RFR_test1_seed38]] = 3
    print("vertiLabels", vertiLabels)
    print("horizTitles", horizTitles)
    print("horizLabels", len(horizLabels))
    print("horizValues", len(horizValues))

    print("ScoresDf", ScoresDf)
    AllDfs = [ScoresDf]
    sheetNames = ['ScoresDf', 'ScoresDfsorted']

    sortedDfs = []
    for df in AllDfs:
        df.loc[:, 'Total'] = df.abs().sum(axis=1)
        df.loc[:, 'Occurences'] = df.notnull().sum(axis=1) - 1
        df.loc[:, 'Total/Occurences'] = df['Total'] / df['Occurences']  # check this

        sortedDf = df.sort_values('Total', ascending=False)
        sortedDfs.append(sortedDf)
    AllDfs += sortedDfs

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference[:-6] + '_Combined/' + 'RECORDS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)
        with pd.ExcelWriter(outputPathStudy + reference[:-6] + "_CV_ModelRanking_NBest" + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)

