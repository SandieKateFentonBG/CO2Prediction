import pandas as pd
import shap
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from Main_GS_FS_Steps import unpackGS_FSs

def CombineAbsSHAP(studies_Blender, studies_GS_FS, xQuantLabels, xQualLabels):
    yLabels_all = studies_GS_FS[0][0].__getattribute__('NoSelector').selectedDict #ex bldg_area

    yLabels_cat = xQuantLabels + xQualLabels
    xLabels = [] #ex LR_LASSO_Fl_Spearman

    shapLabelsLs = []
    shapLs = []

    shapGroupLabels = []
    shapGroupLs = []

    #"query labels and values"
    for blendModel in studies_Blender:
        for GS in blendModel.modelList:

            name = GS.GSName
            xLabels.append(name)
            print()

            shapLabelsLs.append(GS.SHAPdf['feature'])
            shapLs.append(GS.SHAPdf['importance'])

            shapGroupLabels.append(GS.SHAPGroupDf['feature'])
            shapGroupLs.append(GS.SHAPGroupDf['importance'])

    #create empty dfs

    SHAPDf = pd.DataFrame(columns=xLabels, index=yLabels_all)

    SHAPGroupDf = pd.DataFrame(columns=xLabels, index=yLabels_cat)

    for i in range(len(xLabels)):

        # fill in SHAP
        for shapLab, shapVal in zip(shapLabelsLs[i], shapLs[i]):
            SHAPDf.loc[[shapLab], [xLabels[i]]] = shapVal
        # fill in SHAPGroup
        for shapLab, shapVal in zip(shapGroupLabels[i], shapGroupLs[i]):
            SHAPGroupDf.loc[[shapLab], [xLabels[i]]] = shapVal

    return SHAPDf,SHAPGroupDf

def formatCV_SHAP_toDf(CV_Assembly, studies_GS_FS, xQuantLabels, xQualLabels, randomValues = None, NBest = True):
    yLabels_all = studies_GS_FS[0][0].__getattribute__('NoSelector').selectedDict #ex bldg_area - this retrieves all y labels

    yLabels_cat = xQuantLabels + xQualLabels
    xLabels = [] #ex LR_LASSO_Fl_Spearman

    xGroupLabels = []

    shapLabelsLs = []
    shapValuesLs = []

    shapGroupLabels = []
    shapGroupValuesLs = []

    #"query labels and values"

    if randomValues:
        seeds = randomValues
    else:
        seeds = list(range(len(CV_Assembly)))

    if NBest:
        for BlenderNBest, seed in zip(CV_Assembly, seeds): #10studies
            for Model in BlenderNBest.modelList: #10best

                name = Model.GSName #LR_RFR

                df_shap_values = pd.DataFrame(data=Model.SHAPvalues.T, index=Model.SHAPdf['feature']) #index = rows

                for i in range(len(df_shap_values.columns)):
                    xLabels.append(name + '_sample' +str(i) + '_seed' + str(seed)) # ex LR_RFR_test1_seed38
                    shapLabelsLs.append(list(Model.SHAPdf['feature'])) #ex : [Gifa, floors_bg, structure]
                    shapValuesLs.append(list(df_shap_values.iloc[:, i])) # ex [1,2,3] #1 list per tested sample

                df_shapGroup_values = pd.DataFrame(data=Model.SHAPGroupvalues.T, index=Model.SHAPGroupDf['feature'])

                for i in range(len(df_shapGroup_values.columns)):
                    xGroupLabels.append(name + '_sample' + str(i) + '_seed' + str(seed))
                    shapGroupLabels.append(list(Model.SHAPGroupDf['feature']))
                    shapGroupValuesLs.append(list(df_shapGroup_values.iloc[:, i]))

    else: # if you want to summarize a regressor category - ex all LR
        for GS_FSs, seed in zip(CV_Assembly, seeds):

            Model_List = unpackGS_FSs(GS_FSs)
            for Model in Model_List:  # 10best

                name = Model.GSName  # LR_RFR

                df_shap_values = pd.DataFrame(data=Model.SHAPvalues.T, index=Model.SHAPdf['feature'])  # index = rows

                for i in range(len(df_shap_values.columns)):
                    xLabels.append(name + '_sample' + str(i) + '_seed' + str(seed))  # ex LR_RFR_test1_seed38
                    shapLabelsLs.append(list(Model.SHAPdf['feature']))  # ex : [Gifa, floors_bg, structure]
                    shapValuesLs.append(list(df_shap_values.iloc[:, i]))  # ex [1,2,3] #1 list per tested sample

                df_shapGroup_values = pd.DataFrame(data=Model.SHAPGroupvalues.T, index=Model.SHAPGroupDf['feature'])

                for i in range(len(df_shapGroup_values.columns)):
                    xGroupLabels.append(name + '_sample' + str(i) + '_seed' + str(seed))
                    shapGroupLabels.append(list(Model.SHAPGroupDf['feature']))
                    shapGroupValuesLs.append(list(df_shapGroup_values.iloc[:, i]))

    #create empty dfs
    SHAPDf = pd.DataFrame(columns=xLabels, index=yLabels_all)
    SHAPGroupDf = pd.DataFrame(columns=xGroupLabels, index=yLabels_cat)


    for i in range(len(shapLabelsLs)): #col par col #ex i = 4 : [[Gifa, floors_bg, structure][Gifa, floors_bg, structure][Gifa, floors_bg, structure][Gifa, floors_bg, structure]]
        for j in range(len(shapLabelsLs[i])): #ex : j = 3
            SHAPDf.loc[[shapLabelsLs[i][j]], [xLabels[i]]] = shapValuesLs[i][j] #ex : HAPDf.loc[[structure], [LR_RFR_test1_seed38]] = 3

    for i in range(len(shapGroupLabels)): #col par col
        for j in range(len(shapGroupLabels[i])):
            SHAPGroupDf.loc[[shapGroupLabels[i][j]], [xGroupLabels[i]]] = shapGroupValuesLs[i][j]

    return SHAPDf,SHAPGroupDf

def reportCV_SHAP(SHAPDf, SHAPGroupDf, displayParams, DBpath, NBest = True, n=None, NBestScore=None, GSName = None):

    AllDfs = [SHAPDf, SHAPGroupDf]
    sheetNames = ['SHAPDf', 'SHAPGroupDf', 'SHAPDfsorted', 'SHAPGroupDfsorted']

    sortedDfs = []
    for df in AllDfs:

        dfslice = df.astype(float)
        df.loc[:, 'Total'] = df.abs().sum(axis=1)
        df.loc[:, 'Occurences'] = df.notnull().sum(axis=1) - 1
        df.loc[:, 'Total/Occurences'] = df['Total'] / df['Occurences']  # check this
        df.loc[:, 'Total/N'] = df['Total'] / len(df.columns)

        df.loc[:, 'MaxValue'] = dfslice.max(axis=1)
        idmax = dfslice.idxmax(axis=1)
        df.loc[:, 'MaxRegressor'] = idmax.str.split('_').str[0] + '_' + idmax.str.split('_').str[1]
        df.loc[:, 'MaxFilter'] = idmax.str.split('_').str[-3]
        df.loc[:, 'MaxSample'] = idmax.str.split('_').str[-2]
        df.loc[:, 'MaxSeed'] = idmax.str.split('_').str[-1]

        df.loc[:, 'MinValue'] = dfslice.min(axis=1)
        idmin = dfslice.idxmin(axis=1)
        df.loc[:, 'MinRegressor'] = idmin.str.split('_').str[0] + '_' + idmin.str.split('_').str[1]
        df.loc[:, 'MinFilter'] = idmin.str.split('_').str[-3]
        df.loc[:, 'MinSample'] = idmin.str.split('_').str[-2]
        df.loc[:, 'MinSeed'] = idmin.str.split('_').str[-1]

        sortedDf = df.sort_values('Total/N', ascending=False)

        sortedDfs.append(sortedDf)
    AllDfs += sortedDfs
    if NBest:
        content = "NBest_" + str(n) + '_' + NBestScore
    else:
        content = GSName

    if displayParams['archive']:
        import os
        reference = displayParams['ref_prefix'] + '_Combined/'
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)
        with pd.ExcelWriter(outputPathStudy + displayParams['ref_prefix'] + "_CV_SHAP_" + content + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)

def plotSHAPSummary(SHAPDf, displayParams, DBpath, content='', NBest = True, n=None, NBestScore=None):
    """Plot combined shap summary for fitted estimators and a set of test with its labels"""

    newDf = SHAPDf.fillna(0)
    shap_values = np.array(newDf).T  # 38(test)*20(best models)
    features = newDf.index

    # plot & save SHAP values
    shap_summary = shap.summary_plot(shap_values=shap_values, features=features, plot_type="dot", show=False)


    plt.suptitle('Summary of SHAP values - ' + content, ha="right", size = 'large' )
    plt.gcf().set_size_inches(12, 6)
    reference = displayParams['ref_prefix'] + '_Combined/'
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/SHAP/'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        if NBest:
            plt.savefig(outputFigPath + '/SHAPCombined_NBest_' + str(n) + '_' + NBestScore + content + '.png')
        else :
            plt.savefig(outputFigPath + '/SHAPCombined_' + content + '.png')


        if displayParams['showPlot']:
            plt.show()

        plt.close()




def RUN_SHAP_Combined_NBest(displayParams, DBpath, studies_NBest, studies_GS_FS, xQuantLabels, xQualLabels, n, NBestScore, randomValues = None):

    SHAPDf,SHAPGroupDf = formatCV_SHAP_toDf(studies_NBest, studies_GS_FS, xQuantLabels, xQualLabels, randomValues = randomValues)

    plotSHAPSummary(SHAPDf, displayParams, DBpath, content='Ungrouped', n=n, NBestScore=NBestScore)
    plotSHAPSummary(SHAPGroupDf, displayParams, DBpath, content='Grouped', n=n, NBestScore=NBestScore)
    reportCV_SHAP(SHAPDf, SHAPGroupDf, displayParams, DBpath, NBest = True, n=n, NBestScore=NBestScore, GSName = None)

def RUN_SHAP_Combined_All(displayParams, DBpath, studies_GS_FS, GSName,xQuantLabels, xQualLabels, randomValues=None):
    print('formatCV_SHAP_toDf')
    SHAPDf, SHAPGroupDf = formatCV_SHAP_toDf(studies_GS_FS, studies_GS_FS, xQuantLabels, xQualLabels,
                                             randomValues=randomValues, NBest = False)
    print('Ungrouped_plotSHAPSummary')
    plotSHAPSummary(SHAPDf, displayParams, DBpath, content='Ungrouped_' + GSName, NBest = False)
    print('Grouped_plotSHAPSummary')
    plotSHAPSummary(SHAPGroupDf, displayParams, DBpath, content='Grouped_' + GSName, NBest = False)

    # reportCV_SHAP(SHAPDf, SHAPGroupDf, displayParams, DBpath, NBest = False, GSName = GSName)