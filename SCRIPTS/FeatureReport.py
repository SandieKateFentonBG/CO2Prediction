import pandas as pd

# def reportGS_FeatureWeights_All(DBpath, displayParams, GS_FSs, xQuantLabels, xQualLabels):
#
#     yLabels_all = GS_FSs[0].__getattribute__('NoSelector').selectedLabels #ex bldg_area
#     yLabels_cat = xQuantLabels + xQualLabels
#     xLabels = [] #ex LR_LASSO_Fl_Spearman
#
#     weightLabelsLs = []
#     weightsLs = []
#     weightsScaledLs = []
#
#     shapLabelsLs = []
#     shapLs = []
#
#     shapGroupLabels = []
#     shapGroupLs = []
#
#     shapScoreLabels = []
#     shapScoreLs = []
#
#     shapGroupScoreLabels = []
#     shapGroupScoreLs = []
#
#     #"query labels and values"
#     for GS_FS in GS_FSs:
#         for DfLabel in GS_FS.learningDfsList:
#             GS = GS_FS.__getattribute__(DfLabel)
#             name = GS.GSName
#             xLabels.append(name)
#
#             weightLabelsLs.append(GS.selectedLabels)
#             weightsLs.append(GS.Weights)
#             weightsScaledLs.append(GS.WeightsScaled)
#
#             shapLabelsLs.append(GS.SHAPdf['feature'])
#             shapLs.append(GS.SHAPdf['importance'])
#
#             shapGroupLabels.append(GS.SHAPGroupDf['feature'])
#             shapGroupLs.append(GS.SHAPGroupDf['importance'])
#
#             shapScoreLabels.append(list(GS.SHAPScoreDict.keys()))
#             shapScoreLs.append(list(GS.SHAPScoreDict.values()))
#
#             shapGroupScoreLabels.append(list(GS.SHAPGroupScoreDict.keys()))
#             shapGroupScoreLs.append(list(GS.SHAPGroupScoreDict.values()))
#
#     #create empty dfs
#     weightsDf = pd.DataFrame(columns=xLabels, index=yLabels_all)
#     WeightsScaledDf = pd.DataFrame(columns=xLabels, index=yLabels_all)
#     SHAPDf = pd.DataFrame(columns=xLabels, index=yLabels_all)
#     SHAPGroupDf = pd.DataFrame(columns=xLabels, index=yLabels_cat)
#     SHAPScoreDf = pd.DataFrame(columns=xLabels, index=yLabels_all)
#     SHAPGroupScoreDf = pd.DataFrame(columns=xLabels, index=yLabels_cat)
#
#     for i in range(len(xLabels)):
#         # print(xLabels[i])
#         # fill in weights, weightsScaled
#         for sLabel, weight, weightsc in zip(weightLabelsLs[i], weightsLs[i], weightsScaledLs[i]):
#             weightsDf.loc[[sLabel], [xLabels[i]]] = weight
#             WeightsScaledDf.loc[[sLabel], [xLabels[i]]] = weightsc
#         # fill in SHAP
#         for shapLab, shapVal in zip(shapLabelsLs[i], shapLs[i]):
#             SHAPDf.loc[[shapLab], [xLabels[i]]] = shapVal
#         # fill in SHAPGroup
#         for shapLab, shapVal in zip(shapGroupLabels[i], shapGroupLs[i]):
#             SHAPGroupDf.loc[[shapLab], [xLabels[i]]] = shapVal
#         # fill in SHAPScore
#         for shapLab, shapVal in zip(shapScoreLabels[i], shapScoreLs[i]):
#             SHAPScoreDf.loc[[shapLab], [xLabels[i]]] = shapVal
#
#         # fill in SHAPGroupScore
#         for shapLab, shapVal in zip(shapGroupScoreLabels[i], shapGroupScoreLs[i]):
#             SHAPGroupScoreDf.loc[[shapLab], [xLabels[i]]] = shapVal
#
#     # SHAPScoreDf.loc[:,'Total'] = SHAPScoreDf.sum(axis=1)
#
#     allDfs = [weightsDf, WeightsScaledDf,SHAPDf,SHAPGroupDf,SHAPScoreDf, SHAPGroupScoreDf]
#     sortedDfs = []
#     for df in allDfs:
#         df.loc[:,'Total'] = df.sum(axis=1)
#         df.loc[:,'Occurences'] = df.notnull().sum(axis=1) - 1
#         # df.loc[:,'Total/Occurences'] = df.sum(axis=1) / df.notnull().sum(axis=1)
#         df.loc[:,'Total/Occurences'] = df['Total']/ df['Occurences'] #check this
#
#         sortedDf = df.sort_values('Total/Occurences', ascending = False)
#
#         sortedDfs.append(sortedDf)
#     allDfs += sortedDfs
#
#
#     featureScoreDfName = ['weightsDf', 'WeightsScaledDf','SHAPDf','SHAPGroupDf','SHAPScoreDf', 'SHAPGroupScoreDf',
#         'weightsDfsorted', 'WeightsScaledDfsorted','SHAPDfsorted','SHAPGroupDfsorted','SHAPScoreDfsorted', 'SHAPGroupScoreDfsorted']
#
#     if displayParams['archive']:
#         import os
#         reference = displayParams['reference']
#         outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'
#         # outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'
#
#         if not os.path.isdir(outputPathStudy):
#             os.makedirs(outputPathStudy)
#         # with pd.ExcelWriter(outputPathStudy + reference[:-1] + "FeatureReport" + "_GS_FS" ".xlsx", mode='w') as writer:
#         with pd.ExcelWriter(outputPathStudy + reference[:-1] + "_GS_FeatureWeights_All" + ".xlsx", mode='w') as writer:
#             for df, name in zip(allDfs, featureScoreDfName):
#                 df.to_excel(writer, sheet_name=name)

def reportGS_FeatureWeights(DBpath, displayParams, GS_FSs, blender = None):

    yLabels_all = GS_FSs[0].__getattribute__('NoSelector').selectedLabels #ex bldg_area
    xLabels = [] #ex LR_LASSO_Fl_Spearman

    weightLabelsLs = []
    weightsLs = []
    weightsScaledLs = []

    #"query labels and values"
    if blender: #GS_FSs. should be a Blender
        title = "_GS_FeatureWeights_NBest"
        for Model in blender.modelList: #10best
            name = Model.GSName
            xLabels.append(name)

            weightLabelsLs.append(Model.selectedLabels)
            weightsLs.append(Model.Weights)
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
        sortedDf = df.sort_values('Total/Occurences', ascending = False)
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


def reportGS_FeatureSHAP(DBpath, displayParams, GS_FSs, xQuantLabels, xQualLabels, blender = None):

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
    if blender: #GS_FSs. should be a Blender
        title = "_GS_FeatureSHAP_NBest"
        for Model in blender.modelList: #10best

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

