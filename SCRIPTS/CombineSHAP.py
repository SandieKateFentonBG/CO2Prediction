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

def formatCV_SHAP_NBest(CV_BlenderNBest, studies_GS_FS, xQuantLabels, xQualLabels, randomValues = None):
    yLabels_all = studies_GS_FS[0][0].__getattribute__('NoSelector').selectedLabels #ex bldg_area - this retrieves all y labels

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
        seeds = list(range(len(CV_BlenderNBest)))

    for BlenderNBest, seed in zip(CV_BlenderNBest, seeds): #10studies
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

def reportCV_SHAP_NBest(SHAPDf, SHAPGroupDf, NBestscore, displayParams, DBpath):

    AllDfs = [SHAPDf, SHAPGroupDf]
    sheetNames = ['SHAPDf', 'SHAPGroupDf', 'SHAPDfsorted', 'SHAPGroupDfsorted']

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
        with pd.ExcelWriter(outputPathStudy + reference[:-6] + "_CV_SHAP_NBest_" + NBestscore + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)

def plotSHAPSummary_NBest(SHAPDf, NBestScore, displayParams, DBpath, content=''):
    """Plot combined shap summary for fitted estimators and a set of test with its labels"""

    newDf = SHAPDf.fillna(0)
    shap_values = np.array(newDf).T  # 38(test)*20(best models)
    features = newDf.index

    # plot & save SHAP values
    shap_summary = shap.summary_plot(shap_values=shap_values, features=features, plot_type="dot", show=False)

    plt.suptitle('Summary of SHAP values - ' + content, ha="right", size = 'large' )
    plt.gcf().set_size_inches(12, 6)
    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'VISU/SHAP/'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/SHAPCombined_NBest_' + NBestScore + content + '.png')
    if displayParams['showPlot']:
        plt.show()

    plt.close()


def RUN_SHAP_Combined(displayParams, DBpath, studies_Blender, studies_GS_FS, xQuantLabels, xQualLabels, NBestScore, randomValues = None):

    SHAPDf,SHAPGroupDf = formatCV_SHAP_NBest(studies_Blender, studies_GS_FS, xQuantLabels, xQualLabels, randomValues = randomValues)

    plotSHAPSummary_NBest(SHAPDf, NBestScore, displayParams, DBpath, content='Ungrouped')
    plotSHAPSummary_NBest(SHAPGroupDf, NBestScore, displayParams, DBpath, content='Grouped')

    reportCV_SHAP_NBest(SHAPDf, SHAPGroupDf, NBestScore, displayParams, DBpath)