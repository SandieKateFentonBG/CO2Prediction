import pandas as pd
import shap
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def CombineAbsSHAP(studies_Blender, studies_GS_FS, xQuantLabels, xQualLabels):
    yLabels_all = studies_GS_FS[0][0].__getattribute__('NoSelector').selectedLabels #ex bldg_area

    yLabels_cat = xQuantLabels + xQualLabels
    xLabels = [] #ex LR_LASSO_Fl_Spearman

    shapLabelsLs = []
    shapLs = []

    shapGroupLabels = []
    shapGroupLs = []

    #"query labels and values"
    for blendModel in studies_Blender:
        for GS in blendModel.modelList:
    # for GS_FSs in studies_GS_FS:
    #     for GS_FS in GS_FSs:
    #         for DfLabel in GS_FS.learningDfsList:
    #             GS = GS_FS.__getattribute__(DfLabel)
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

def CombineAllSHAP(studies_Blender, studies_GS_FS, xQuantLabels, xQualLabels):
    yLabels_all = studies_GS_FS[0][0].__getattribute__('NoSelector').selectedLabels #ex bldg_area

    yLabels_cat = xQuantLabels + xQualLabels
    xLabels = [] #ex LR_LASSO_Fl_Spearman

    xGroupLabels = []

    shapLabelsLs = []
    shapLs = []

    shapGroupLabels = []
    shapGroupLs = []

    #"query labels and values"
    for blendModel in studies_Blender:
        for GS in blendModel.modelList:

            name = GS.GSName

            df_shap_values = pd.DataFrame(data=GS.SHAPvalues.T, index=GS.SHAPdf['feature'])

            for i in range(len(df_shap_values.columns)):
                xLabels.append(name + str(i))
                shapLabelsLs.append(list(GS.SHAPdf['feature']))
                shapLs.append(list(df_shap_values.iloc[:, i]))

            df_shapGroup_values = pd.DataFrame(data=GS.SHAPGroupvalues.T, index=GS.SHAPGroupDf['feature'])

            for i in range(len(df_shapGroup_values.columns)):
                xGroupLabels.append(name + str(i))
                shapGroupLabels.append(list(GS.SHAPGroupDf['feature']))
                shapGroupLs.append(list(df_shapGroup_values.iloc[:, i]))


    #create empty dfs
    SHAPDf = pd.DataFrame(columns=xLabels, index=yLabels_all)
    SHAPGroupDf = pd.DataFrame(columns=xGroupLabels, index=yLabels_cat)


    for i in range(len(shapLabelsLs)): #col par col
        for j in range(len(shapLabelsLs[i])):
            SHAPDf.loc[[shapLabelsLs[i][j]], [xLabels[i]]] = shapLs[i][j]

    for i in range(len(shapGroupLabels)): #col par col
        for j in range(len(shapGroupLabels[i])):
            SHAPGroupDf.loc[[shapGroupLabels[i][j]], [xGroupLabels[i]]] = shapGroupLs[i][j]

    return SHAPDf,SHAPGroupDf

def Report_shap_CombinedSummaryPlot(SHAPDf,SHAPGroupDf, displayParams, DBpath):

    AllDfs = [SHAPDf, SHAPGroupDf]
    sheetNames = ['SHAPDf', 'SHAPGroupDf', 'SHAPDfsorted', 'SHAPGroupDfsorted']

    sortedDfs = []
    for df in AllDfs:
        df.loc[:, 'Total'] = df.abs().sum(axis=1)
        df.loc[:, 'Occurences'] = df.notnull().sum(axis=1) - 1
        df.loc[:, 'Total/Occurences'] = df['Total'] / df['Occurences']  # check this

        sortedDf = df.sort_values('Total/Occurences', ascending=False)

        sortedDfs.append(sortedDf)
    AllDfs += sortedDfs

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference[:-6] + '_Combined/' + 'RECORDS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)
        with pd.ExcelWriter(outputPathStudy + reference[:-6] + "_CombinedSHAP" + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)

def plot_shap_CombinedSummaryPlot(SHAPDf, displayParams, DBpath, content=''):
    """Plot combined shap summary for fitted estimators and a set of test with its labels"""

    newDf = SHAPDf.fillna(0)
    shap_values = np.array(newDf).T  # 38(test)*20(best models)
    features = newDf.index

    # plot & save SHAP values
    shap_summary = shap.summary_plot(shap_values=shap_values, features=features, plot_type="dot", show=False)

    plt.suptitle('Combined SHAP', ha="right", size = 'large' )
    plt.gcf().set_size_inches(12, 6)
    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'VISU/SHAP/'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/SHAPCombined_' + content + '.png')
    if displayParams['showPlot']:
        plt.show()

    plt.close()


def RUN_SHAP_Combined(displayParams, DBpath, studies_Blender, studies_GS_FS, xQuantLabels, xQualLabels):

    SHAPDf,SHAPGroupDf = CombineAllSHAP(studies_Blender, studies_GS_FS, xQuantLabels, xQualLabels)

    plot_shap_CombinedSummaryPlot(SHAPDf, displayParams, DBpath, content='SHAPDf')
    plot_shap_CombinedSummaryPlot(SHAPGroupDf, displayParams, DBpath, content='SHAPGroupDf')

    Report_shap_CombinedSummaryPlot(SHAPDf, SHAPGroupDf, displayParams, DBpath)