def ReportStudy(displayParams, DB_Values, FORMAT_Values, PROCESS_VALUES, RFE_VALUES, GS_VALUES, rdat, df, learningDf,
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
            # writer.writerow(['BLENDING DATA'])
            #
            # index = [model.GSName for model in blendModel.modelList] + [blendModel.GSName]
            # columns = ['TrainScore', 'TestScore', 'TestMSE', 'TestR2', 'TestAcc', 'ResidMean', 'ResidVariance',
            #            'ModelWeights']  #
            #
            # import pandas as pd
            #
            # BlendingDf = pd.DataFrame(columns=columns, index=index)
            # for col in columns[:-1]:
            #     BlendingDf[col] = [model.__getattribute__(col) for model in blendModel.modelList] + [
            #         blendModel.__getattribute__(col)]
            # BlendingDf['ModelWeights'] = [round(elem, 3) for elem in list(blendModel.ModelWeights)] + [0]
            #
            # writer.writerow(BlendingDf)


        e.close()


def reportCombinedStudies(studies_Blender, displayParams, DBpath, random_seeds = None):

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

            with pd.ExcelWriter(outputPathStudy + reference[:-1] + "_CombinedReport" + ".xlsx", mode='w') as writer:
                for df, name in zip(AllDfs, sheetNames):
                    df.to_excel(writer, sheet_name=name)