def reportGS_Details_All(displayParams, DB_Values, FORMAT_Values, PROCESS_VALUES, RFE_VALUES, GS_VALUES, BLE_VALUES,
                         rdat, dat, df, learningDf, baseFormatedDf, FiltersLs, RFEs, GSlist, GSwithFS = True):


    if displayParams['archive']:

        import os

        reference = displayParams['reference']
        outputPathStudy = DB_Values['DBpath'] + "RESULTS/" + reference + 'RECORDS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        import csv

        name = reference.rstrip(reference[-1])

        with open(outputPathStudy + name + '_GS_Records_All' + ".csv", 'w', encoding='UTF8', newline='') as e:
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
            writer.writerow(['BLE_VALUES', BLE_VALUES])
            writer.writerow('')

            writer.writerow(['PREPROCESSED DATA'])

            writer.writerow(["Full df ", df.shape])
            writer.writerow(["Outliers removed ", learningDf.shape])
            ylabel = [k for k in rdat.y]
            ycol = learningDf.loc[:, ylabel[0]]
            writer.writerow(["Target min, max, mean, std ", ycol.min(), ycol.max(), ycol.mean(), ycol.std()])

            writer.writerow(["All labels", dat.allLabels])
            writer.writerow(["Selected labels", dat.remainingLabels])
            writer.writerow(["Removed dictionnary ", dat.removedDict])
            writer.writerow(["Removed labels", dat.droppedLabels])

            writer.writerow('')
            writer.writerow(['FORMAT'])
            writer.writerow(["train", baseFormatedDf.trainDf.shape])
            writer.writerow(["test", baseFormatedDf.testDf.shape])
            writer.writerow(["validate", baseFormatedDf.valDf.shape])
            writer.writerow(["check", baseFormatedDf.checkDf.shape])

            writer.writerow([''])

            if len(FiltersLs)>0:
                writer.writerow(['FILTER'])
                for filter in FiltersLs :
                    writer.writerow(['FILTER ', filter.method])
                    writer.writerow(['LABELS ', filter.trainDf.shape[1]-1]) #todo this was changed - check
                    writer.writerow([filter.selectedLabels])
                    writer.writerow('')

            if len(RFEs) > 0:
                writer.writerow(['RFE'])
                for RFE in RFEs:
                    writer.writerow(["RFE with  ", RFE.method])
                    writer.writerow(["Number of features fixed ", RFE.n_features_to_select])
                    writer.writerow(['Selected feature labels ', list(RFE.selectedLabels)])
                    writer.writerow(["Score on training ", RFE.rfe_valScore])
                    writer.writerow(["Score on validation ", RFE.rfe_checkScore])
                    writer.writerow('')

            writer.writerow('')

            writer.writerow(['GRIDSEARCH DATA'])

            writer.writerow(['BASE REFIT', PROCESS_VALUES['refit']])
            writer.writerow(['BASE SELECT', PROCESS_VALUES['grid_select']])
            writer.writerow(['BLENDER REFIT', BLE_VALUES['refit']])
            writer.writerow(['BLENDER SELECT', BLE_VALUES['grid_select']])

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
                sortedModels_R2 = sorted(allModels, key=lambda x: x[-2], reverse=True)
                sortedModels_MSE = sorted(allModels, key=lambda x: x[-3], reverse=True)

            else : # then GSlist should be GSs
                allModels = []
                for Model in GSlist:
                    v = [Model.__getattribute__(keys[i]) for i in range(len(keys))]
                    writer.writerow(v)
                    allModels.append(v)
                # sortedModels = sorted(allModels, key=lambda x: x[-1], reverse=True)
                sortedModels_Acc = sorted(allModels, key=lambda x: x[-1], reverse=True)
                sortedModels_R2 = sorted(allModels, key=lambda x: x[-2], reverse=True)
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

            writer.writerow(['SORTED GRIDSEARCH DATA - R2'])
            writer.writerow(keys)
            for elem in sortedModels_R2:
                writer.writerow(elem)

            writer.writerow('')

        e.close()



def reportCV_ModelRanking_NBest(CV_AllModels, CV_BlenderNBest, seeds, displayParams, DBpath,
                                numericLabels = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore', 'TestAcc_mean', 'TestAcc_std', 'ResidMean', 'ResidVariance']
                                , ordinalCountLabels = ['selectedLabels'],
                                ordinalLabels = ['selectorName'], n = 10, NBestScore = 'TestR2'):

    import pandas as pd
    allEvaluatedLabels = numericLabels + ordinalCountLabels + ordinalLabels

    #dataframe labels
    horizTitles = []
    horizLabels = []
    vertiLabels = []

    #vertilabels
    for predictor in CV_AllModels[0]:
        for learningDflabel in predictor.learningDfsList:
            Model = predictor.__getattribute__(learningDflabel)
            name = Model.GSName  # LR_RFR
            vertiLabels.append(name) #verti = models
    #horizlabels
    for BlenderNBest, seed in zip(CV_BlenderNBest, seeds):  # 10studies
        horizTitle = seed  # ex : 38
        horizLabel = []
        for Model in BlenderNBest.modelList:  # 10best
            name = Model.GSName
            horizLabel.append(name)  # ex : [LR_RFR, LR_DTR, LR_GBR]
        horizTitles.append(horizTitle)  # ex : [38 38 38]
        horizLabels.append(
            horizLabel)  # ex : [[LR_RFR, LR_DTR, LR_GBR][LR_RFR, LR_DTR, LR_GBR][LR_RFR, LR_DTR, LR_GBR]]

    AllDfs = []
    for label in allEvaluatedLabels:
        horizValues = []
        for BlenderNBest, seed in zip(CV_BlenderNBest, seeds): #10studies
            horizValue1 = []
            for Model in BlenderNBest.modelList: #10best
                if label in ordinalCountLabels:
                    val = len(Model.__getattribute__(label))
                else:
                    val = Model.__getattribute__(label)
                horizValue1.append(val) #ex : [0.8, 0.8, 0.8]
            horizValues.append(horizValue1) #ex : [[0.8, 0.8, 0.8][0.8, 0.8, 0.8][0.8, 0.8, 0.8]]

        # create empty dfs
        ScoresDf = pd.DataFrame(columns=horizTitles, index=vertiLabels)
        # fill in values
        for i in range(len(horizLabels)): #col par col #ex i = 4 : [[Gifa, floors_bg, structure][Gifa, floors_bg, structure][Gifa, floors_bg, structure][Gifa, floors_bg, structure]]
            for j in range(len(horizLabels[i])): #ex : j = 3
                ScoresDf.loc[[horizLabels[i][j]], [horizTitles[i]]] = horizValues[i][j] #ex : HAPDf.loc[[structure], [LR_RFR_test1_seed38]] = 3
        AllDfs.append(ScoresDf)

    sortedDfs = []
    #compute row totals
    for df in AllDfs:
        df.loc[:, 'Occurences'] = df.notnull().sum(axis=1)
        slice = df.iloc[:, 0:len(horizTitles)]
        try:
            df.loc[:, 'Total'] = slice.abs().sum(axis=1)
            df.loc[:, 'Total/Occurences'] = df['Total'] / df['Occurences']  # check this
        # deal with ordinalCount Labels - compute column totals
        except:
            allSelectorsDf = []
            df.loc[:, 'Total'] = slice.notnull().sum(axis=1)
            df.loc[:, 'Total/Occurences'] = df['Total'] / df['Occurences']
            SelectorsDf = pd.DataFrame(columns=horizTitles,
                                       index=['NoSelector', 'fl_pearson', 'fl_spearman', 'RFE_DTR', 'RFE_RFR', 'RFE_GBR'])
            for col in horizTitles: #40 df.columns
                for myk, myv in zip(df[int(col)].value_counts().keys(), list(df[int(col)].value_counts())): #noselector, #5
                    SelectorsDf.loc[myk, col] = myv
            SelectorsDf.loc[:, 'Occurences'] = SelectorsDf.notnull().sum(axis=1)
            SelectorsDfslice = SelectorsDf.iloc[:, 0:len(horizTitles)]
            SelectorsDf.loc[:, 'Total'] = SelectorsDfslice.abs().sum(axis=1)
            SelectorsDf.loc[:, 'Total/Occurences'] = SelectorsDf['Total'] / SelectorsDf['Occurences']  # check this

        df.loc['Occurences', :] = df.notnull().sum(axis=0)
    sortedDfMain = AllDfs[0].sort_values('Total', ascending=False)

    #sort all df according to increasing accuracy
    for df in AllDfs:
        df.loc[:, 'TotalAcc'] = AllDfs[0].loc[:, 'Total']
        sortedDf = df.sort_values('TotalAcc', ascending=False)
        sortedDfs.append(sortedDf)
    AllDfs += sortedDfs
    # add column totals for ordinalcount labels
    AllDfs[-1] = AllDfs[-1].append(SelectorsDf)

    sheetNames = allEvaluatedLabels + [l+'sorted' for l in allEvaluatedLabels]
    # export
    if displayParams['archive']:
        import os
        reference = displayParams['ref_prefix'] + '_Combined/'
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'
        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)
        with pd.ExcelWriter(outputPathStudy + displayParams['ref_prefix'] + "_CV_ModelRanking_NBest" +'_' + str(n) + '_' + NBestScore + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name, freeze_panes=(0, 1))
                # a = df.to_excel(writer, sheet_name=name, freeze_panes=(0,1))
                # a.column_dimensions["A"].width = 20
                # # a = writer.sheets #column_dimensions["A"]
                # print(type(a), a)


