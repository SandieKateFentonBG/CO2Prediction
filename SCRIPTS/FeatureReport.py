import pandas as pd


def reportGS_FeatureWeights(DBpath, displayParams, GS_FSs, NBestModel = None):

    yLabels_all = GS_FSs[0].__getattribute__('NoSelector').selectedLabels #ex bldg_area
    xLabels = [] #ex LR_LASSO_Fl_Spearman

    weightLabelsLs = []
    weightsLs = []
    weightsScaledLs = []

    #"query labels and values"
    if NBestModel: #GS_FSs. should be a Blender
        title = "_GS_FeatureWeights_NBest_" + str(NBestModel.N) + '_' + NBestModel.NBestScore
        for Model in NBestModel.modelList: #10best
            name = Model.GSName
            xLabels.append(name)

            weightLabelsLs.append(Model.selectedLabels)
            weightsLs.append(Model.Weights) #replace with Model.ModelWeights
            weightsScaledLs.append(Model.WeightsScaled)

    else :
        title = "_GS_FeatureWeights_All"
        for GS_FS in GS_FSs:
            for DfLabel in GS_FS.learningDfsList:
                Model = GS_FS.__getattribute__(DfLabel)
                name = Model.GSName
                xLabels.append(name)

                weightLabelsLs.append(Model.selectedLabels)
                weightsLs.append(Model.Weights)
                weightsScaledLs.append(Model.WeightsScaled)

    #create empty dfs
    weightsDf = pd.DataFrame(columns=xLabels, index=yLabels_all)
    WeightsScaledDf = pd.DataFrame(columns=xLabels, index=yLabels_all)

    for i in range(len(xLabels)):
        # fill in weights, weightsScaled
        for sLabel, weight, weightsc in zip(weightLabelsLs[i], weightsLs[i], weightsScaledLs[i]):
            weightsDf.loc[[sLabel], [xLabels[i]]] = weight
            WeightsScaledDf.loc[[sLabel], [xLabels[i]]] = weightsc

    allDfs = [weightsDf, WeightsScaledDf]
    sortedDfs = []
    for df in allDfs:
        df.loc[:,'Total'] = df.sum(axis=1)
        df.loc[:,'Occurences'] = df.notnull().sum(axis=1) - 1
        df.loc[:,'Total/Occurences'] = df['Total']/ df['Occurences'] #check this
        if NBestModel :
            df.loc[:,'Total/N'] = df['Total'] / NBestModel.N
        else :
            df.loc[:, 'Total/N'] = df['Total'] / len(xLabels)

        sortedDf = df.sort_values('Total/N', ascending = True)
        sortedDfs.append(sortedDf)
    allDfs += sortedDfs

    featureScoreDfName = ['weightsDf', 'WeightsScaledDf','weightsDfsorted', 'WeightsScaledDfsorted']

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'
        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)
        with pd.ExcelWriter(outputPathStudy + reference[:-1] + title + ".xlsx", mode='w') as writer:
            for df, name in zip(allDfs, featureScoreDfName):
                df.to_excel(writer, sheet_name=name)


def reportGS_FeatureSHAP(DBpath, displayParams, GS_FSs, xQuantLabels, xQualLabels, NBestModel = None):

    yLabels_all = GS_FSs[0].__getattribute__('NoSelector').selectedLabels #ex bldg_area
    yLabels_cat = xQuantLabels + xQualLabels
    xLabels = [] #ex LR_LASSO_Fl_Spearman

    shapLabelsLs = []
    shapLs = []

    shapGroupLabels = []
    shapGroupLs = []

    shapScoreLabels = []
    shapScoreLs = []

    shapGroupScoreLabels = []
    shapGroupScoreLs = []

    #"query labels and values"
    if NBestModel: #GS_FSs. should be a Blender
        title = "_GS_FeatureSHAP_NBest_" + str(NBestModel.N) + '_' + NBestModel.NBestScore
        for Model in NBestModel.modelList: #10best

            name = Model.GSName
            xLabels.append(name)

            shapLabelsLs.append(Model.SHAPdf['feature'])
            shapLs.append(Model.SHAPdf['importance'])

            shapGroupLabels.append(Model.SHAPGroupDf['feature'])
            shapGroupLs.append(Model.SHAPGroupDf['importance'])

            shapScoreLabels.append(list(Model.SHAPScoreDict.keys()))
            shapScoreLs.append(list(Model.SHAPScoreDict.values()))

            shapGroupScoreLabels.append(list(Model.SHAPGroupScoreDict.keys()))
            shapGroupScoreLs.append(list(Model.SHAPGroupScoreDict.values()))
    else:
        title = "_GS_FeatureSHAP_All"
        for GS_FS in GS_FSs:
            for DfLabel in GS_FS.learningDfsList:
                Model = GS_FS.__getattribute__(DfLabel)
                name = Model.GSName
                xLabels.append(name)

                shapLabelsLs.append(Model.SHAPdf['feature'])
                shapLs.append(Model.SHAPdf['importance'])

                shapGroupLabels.append(Model.SHAPGroupDf['feature'])
                shapGroupLs.append(Model.SHAPGroupDf['importance'])

                shapScoreLabels.append(list(Model.SHAPScoreDict.keys()))
                shapScoreLs.append(list(Model.SHAPScoreDict.values()))

                shapGroupScoreLabels.append(list(Model.SHAPGroupScoreDict.keys()))
                shapGroupScoreLs.append(list(Model.SHAPGroupScoreDict.values()))

    #create empty dfs
    SHAPDf = pd.DataFrame(columns=xLabels, index=yLabels_all)
    SHAPGroupDf = pd.DataFrame(columns=xLabels, index=yLabels_cat)
    SHAPScoreDf = pd.DataFrame(columns=xLabels, index=yLabels_all)
    SHAPGroupScoreDf = pd.DataFrame(columns=xLabels, index=yLabels_cat)

    for i in range(len(xLabels)):
        # fill in SHAP
        for shapLab, shapVal in zip(shapLabelsLs[i], shapLs[i]):
            SHAPDf.loc[[shapLab], [xLabels[i]]] = shapVal
        # fill in SHAPGroup
        for shapLab, shapVal in zip(shapGroupLabels[i], shapGroupLs[i]):
            SHAPGroupDf.loc[[shapLab], [xLabels[i]]] = shapVal
        # fill in SHAPScore
        for shapLab, shapVal in zip(shapScoreLabels[i], shapScoreLs[i]):
            SHAPScoreDf.loc[[shapLab], [xLabels[i]]] = shapVal
        # fill in SHAPGroupScore
        for shapLab, shapVal in zip(shapGroupScoreLabels[i], shapGroupScoreLs[i]):
            SHAPGroupScoreDf.loc[[shapLab], [xLabels[i]]] = shapVal

    allDfs = [SHAPDf,SHAPGroupDf,SHAPScoreDf, SHAPGroupScoreDf]

    #sort by row sum
    sortedDfs = []
    for df in allDfs:
        df.loc[:,'Total'] = df.sum(axis=1)
        df.loc[:,'Occurences'] = df.notnull().sum(axis=1) - 1
        df.loc[:,'Total/Occurences'] = df['Total']/ df['Occurences'] #check this
        sortedDf = df.sort_values('Total', ascending = False)
        sortedDfs.append(sortedDf)
    allDfs += sortedDfs

    featureScoreDfName = ['SHAPDf','SHAPGroupDf','SHAPScoreDf', 'SHAPGroupScoreDf',
        'SHAPDfsorted','SHAPGroupDfsorted','SHAPScoreDfsorted', 'SHAPGroupScoreDfsorted']

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)
        with pd.ExcelWriter(outputPathStudy + reference[:-1] + title + ".xlsx", mode='w') as writer:
            for df, name in zip(allDfs, featureScoreDfName):
                df.to_excel(writer, sheet_name=name)

