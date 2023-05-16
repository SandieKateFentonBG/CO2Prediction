
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from HelpersArchiver import *


def mergeList(list):
    #merge multiplelists in single list

    return [j for i in list for j in i]

def AssembleCVResiduals(studies):
    residualsDict = dict()

    for predictor in studies[0]:
        for learningDflabel in predictor.learningDfsList:
            model = predictor.__getattribute__(learningDflabel)
            residualsDict[model.GSName] = []

    for study in studies:
        for predictor in study:
            for learningDflabel in predictor.learningDfsList:
                model = predictor.__getattribute__(learningDflabel)
                residualsDict[model.GSName].append(list(model.Resid))

    for k, v in residualsDict.items():
        residualsDict[k] = mergeList(v)

    return residualsDict

def AssembleCVResiduals_NBest(studies, studies_Blender):
    residualsDict = dict()

    for predictor in studies[0]:
        for learningDflabel in predictor.learningDfsList:
            model = predictor.__getattribute__(learningDflabel)
            residualsDict[model.GSName] = []

    residualsDict["all"] = []
    for blender in studies_Blender:
        for model in blender.modelList:
            residualsDict[model.GSName].append(list(model.Resid))

    for k, v in residualsDict.items():
        residualsDict[k] = mergeList(v)

    return residualsDict


def AssembleCVResults(studies, label):
    resultsDict = dict()

    #create empty dictionary
    for predictor in studies[0]:
        for learningDflabel in predictor.learningDfsList:
            model = predictor.__getattribute__(learningDflabel)
            resultsDict[model.GSName] = []
    #fill in with all studies
    for study in studies:
        for predictor in study:
            for learningDflabel in predictor.learningDfsList:
                model = predictor.__getattribute__(learningDflabel)
                resultsDict[model.GSName].append(model.__getattribute__(label))

    return resultsDict


def computeCV_Scores_Avg_All(studies):
    import numpy as np
    SummaryDict = dict()
    #

    for predictor in studies[0]:
        for learningDflabel in predictor.learningDfsList:
            model = predictor.__getattribute__(learningDflabel)
            SummaryDict[model.GSName] = []

    TestAccDict = AssembleCVResults(studies, 'TestAcc')
    TestMSEDict = AssembleCVResults(studies, 'TestMSE')
    ResidMeanDict = AssembleCVResults(studies, 'ResidMean')
    ResidVarianceDict = AssembleCVResults(studies, 'ResidVariance')
    TrainScoreDict = AssembleCVResults(studies, 'TrainScore')
    TestScoreDict = AssembleCVResults(studies, 'TestScore')

    for k in TestAccDict.keys():
        avgAcc1 = round(np.mean(TestAccDict[k]), 3)
        stdAcc1 = round(np.std(TestAccDict[k]), 3)
        avgAcc2 = round(np.mean(TestMSEDict[k]), 3)
        stdAcc2 = round(np.std(TestMSEDict[k]), 3)
        avgAcc3 = round(np.mean(ResidMeanDict[k]), 3)
        stdAcc3 = round(np.std(ResidMeanDict[k]), 3)

        avgAcc4 = round(np.mean(ResidVarianceDict[k]), 3)
        stdAcc4 = round(np.std(ResidVarianceDict[k]), 3)
        avgAcc5 = round(np.mean(TrainScoreDict[k]), 3)
        stdAcc5 = round(np.std(TrainScoreDict[k]), 3)
        avgAcc6 = round(np.mean(TestScoreDict[k]), 3)
        stdAcc6 = round(np.std(TestScoreDict[k]), 3)
        SummaryDict[k] = [avgAcc1, stdAcc1, avgAcc2, stdAcc2, avgAcc3, stdAcc3,
                          avgAcc4, stdAcc4, avgAcc5, stdAcc5, avgAcc6, stdAcc6]

    # track results
    columns = ['TestAcc-Mean', 'TestAcc-Std', 'TestMSE-Mean', 'TestMSE-Std', 'Resid-Mean', 'Resid-Std',
               'ResidVariance-Mean', 'ResidVariance-Std', 'TrainScore-Mean', 'TrainScore-Std', 'TestR2-Mean',
               'TestR2-Std']
    ResultsDf = pd.DataFrame(columns=columns, index=SummaryDict.keys())
    for i in range(len(columns)):
        ResultsDf[columns[i]] = [SummaryDict[k][i] for k in SummaryDict.keys()]

    return ResultsDf


def find_Overall_Best_Models(DBpath, displayParams, ResultsDf, n=10, NBestScore='TestR2'):

    sortedDf = ResultsDf.sort_values(NBestScore + '-Mean', ascending=False)
    BestModelNames = list(sortedDf.index[0:n])
    print("The ", str(n), 'Models with Best ', NBestScore, "are:")
    print(BestModelNames)

    pickleDumpMe(DBpath, displayParams, BestModelNames, 'GS_FS', 'BestModelNames', combined=True)


    return BestModelNames






def reportCV_ScoresAvg_All(ResultsDf, displayParams, DBpath, NBestScore='TestR2'): #ResultsDf

    """    create a dictionary compiling model accuracies for all 10 studies,
    as well as average accuracy, residual Mean and Residual Variance"""


    sortedDf = ResultsDf.sort_values(NBestScore + '-Mean', ascending=False)

    AllDfs = [ResultsDf, sortedDf]
    sheetNames = ['GridsearchResults', 'Sorted_GridsearchResults']

    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'RECORDS/'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        with pd.ExcelWriter(outputFigPath + reference[:-6] + '_CV_ScoresAvg_All' + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)


    return ResultsDf


def AssembleNBestResiduals(studies_Blender): #todo : naming was changed
    residualsDict = dict()
    residualsDict["all"] = []
    for blender in studies_Blender:
        for model in blender.modelList:
            residualsDict["all"].append(list(model.Resid))

    for k, v in residualsDict.items():
        residualsDict[k] = mergeList(v)

    return residualsDict

def AssembleBlenderResiduals(studies_Blender):
    residualsDict = dict()
    residualsDict["all"] = []
    for blender in studies_Blender:
        residualsDict["all"].append(list(blender.Resid))

    for k, v in residualsDict.items():
        residualsDict[k] = mergeList(v)

    return residualsDict

def plotCVResidualsHistogram(studies, displayParams, FORMAT_Values, DBpath, studyFolder ='Histplot'):

    residualsDict = AssembleCVResiduals(studies)

    for k, v in residualsDict.items():
        title = 'Residuals distribution for ' + k
        x = "Residuals [%s]" % FORMAT_Values['targetLabels']
        fig, ax = plt.subplots()

        ax = sns.histplot(v, kde=True, bins=14, binrange = (-100, 100), legend = False)
        plt.setp(ax.patches, linewidth=0)

        plt.title(title, fontsize=14)
        plt.xlabel(x, fontsize=14)

        reference = displayParams['reference']
        if displayParams['archive']:
            path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'VISU/Residuals/' + studyFolder
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + k + '-'+ studyFolder +'.png')

        if displayParams['showPlot']:
            plt.show()

        plt.close()

def analyzeCVResiduals(studies):

    # assemble residuals
    residualsDict = AssembleCVResiduals(studies)
    models, means, variances = [], [], []

    for k, v in residualsDict.items():
        arr = np.array(v)
        mean = np.mean(arr)  #
        variance = np.var(arr)
        models.append(k)
        means.append(round(np.abs(mean), 2))
        variances.append(round(variance, 2))

    return models, means, variances







def plotCVResidualsGaussian_indiv(studies, displayParams, FORMAT_Values, DBpath, studyFolder='GaussianPlot', binwidth=25,
                                  setxLim=[-300, 300], fontsize=12, studies_Blender = None):

    from scipy.stats import norm
    import seaborn as sns


    if studies_Blender : #only takes nBestmodels
        residualsDict = AssembleCVResiduals_NBest(studies, studies_Blender)
        extra = 'NBest'
        a = ''
    else : #takes all models
        residualsDict = AssembleCVResiduals(studies)
        extra = ''

    # assemble residuals
    # residualsDict = AssembleCVResiduals(studies)
    models, means, variances = [], [], []

    listResVal = mergeList(list(residualsDict.values()))
    resmin, resmax = min(listResVal), max(listResVal)

    #TODO : toggle this if you want to have identical boundary values on x axis for all graphs

    if resmax > setxLim[1]:
        import math
        setxLim[1] = math.ceil(resmax / 100) * 100
        print("residuals out of binrange  :", resmax)
        print("bin max changed to :", setxLim[1])
    if resmin < setxLim[0]:
        import math
        setxLim[0] = math.floor(resmin / 100) * 100
        print("residuals out of binrange  :", resmin)
        print("bin min changed to :", setxLim[0])

    #TODO :

    for k, v in residualsDict.items():
        if len(v) > 0:
            title = 'Residuals distribution for ' + k + '' + extra
            x = "Residuals %s" % FORMAT_Values['targetLabels']
            fig, ax = plt.subplots()

            # plot the histplot and the kde
            ax = sns.histplot(v, kde=True, legend=False, binwidth=binwidth, label="Residuals kde curve")
            for u in ['right', 'top']:
                ax.spines[u].set_visible(False)

            plt.setp(ax.patches, linewidth=0)
            # plt.title(title, fontsize=fontsize)
            plt.xlabel(x, fontsize=fontsize)
            plt.ylabel("Tested samples" + "(" + k + ")", fontsize=fontsize)
            arr = np.array(v)

            plt.figure(1)
            if setxLim:
                xLim = (setxLim[0], setxLim[1])
            else:
                xLim = (min(arr), max(arr))
            plt.xlim(xLim)
            mean = np.mean(arr)  #
            variance = np.var(arr)
            models.append(k)
            means.append(round(np.abs(mean), 2))
            variances.append(round(variance, 2))
            sigma = np.sqrt(variance)
            x = np.linspace(min(arr), max(arr), 100)
            t = np.linspace(-300, 300, 100)
            dx = binwidth
            scale = len(arr) * dx

            # plot the gaussian
            plt.plot(t, norm.pdf(t, mean, sigma) * scale, color='red', linestyle='dashed', label="Gaussian curve")
            plt.legend()

            reference = displayParams['reference']
            if displayParams['archive']:
                path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'VISU/Residuals/' + studyFolder
                import os
                outputFigPath = path + folder + subFolder
                if not os.path.isdir(outputFigPath):
                    os.makedirs(outputFigPath)

                plt.savefig(outputFigPath + '/' + k + '-' + studyFolder + extra + '.png')

            if displayParams['showPlot']:
                plt.show()
            plt.close()

    # return models, means, variances

def plotCVResidualsHistogram_Combined(studies, displayParams, FORMAT_Values, DBpath, studyFolder ='Histplot', blended = False):

    if blended : #only takes nBestmodels
        residualsDict = AssembleNBestResiduals(studies)
        title = 'Residuals distribution for 10 best models over 10 runs'
    else : #takes all models
        residualsDict = AssembleCVResiduals(studies)
        title = 'Residuals distribution for all models over 10 runs'

    mergedList = mergeList(list(residualsDict.values()))

    # title = 'Residuals distribution'
    x = "Residuals [%s]" % FORMAT_Values['targetLabels']
    fig, ax = plt.subplots()

    ax = sns.histplot(mergedList, kde=True, bins=14, binrange=(-100, 100), legend=False)
    plt.setp(ax.patches, linewidth=0)

    plt.title(title, fontsize=14)
    plt.xlabel(x, fontsize=14)

    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'VISU/Residuals/' + studyFolder
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + 'Combined' + '-' + studyFolder + '.png')

    if displayParams['showPlot']:
        plt.show()

    plt.close()




def plotCVResidualsGaussian_Combined(studies, displayParams, FORMAT_Values, DBpath, studyFolder='GaussianPlot',
                                     binwidth=25,
                                     setxLim=[-300, 300], fontsize=12, NBest = False, Blender = False, CV = False):
    from scipy.stats import norm
    import seaborn as sns

    # assemble residuals
    if NBest : #only takes nBestmodels
        residualsDict = AssembleNBestResiduals(studies)
        title = 'Residuals distribution for 10 best models over 10 runs'
        a = '10 selected models'
        extra = '_'
    elif Blender:  # only takes Blender results
        residualsDict = AssembleBlenderResiduals(studies)
        title = 'Residuals distribution for Blender Models over 10 runs ' + studies[0].GSName
        extra = '_' + studies[0].GSName
        a = studies[0].GSName
    else : #takes all models
        residualsDict = AssembleCVResiduals(studies)
        title = 'Residuals distribution for all models over 10 runs'
        extra = '_'
        a = 'All Models'

    listResVal = mergeList(list(residualsDict.values()))
    arr = np.array(listResVal)

    mean = np.mean(arr)

    variance = np.var(arr)
    sigma = np.sqrt(variance)

    resmin, resmax = min(listResVal), max(listResVal)

    #TODO : toggle this if you want to have identical boundary values on x axis for all graphs

    if resmax > setxLim[1]:
        import math
        setxLim[1] = math.ceil(resmax / 100) * 100
        print("residuals out of binrange  :", resmax)
        print("bin max changed to :", setxLim[1])
    if resmin < setxLim[0]:
        import math
        setxLim[0] = math.floor(resmin / 100) * 100
        print("residuals out of binrange  :", resmin)
        print("bin min changed to :", setxLim[0])

    #TODO :

    # for k, v in residualsDict.items():
    # title = 'Residuals distribution for '
    x = "Residuals %s" % FORMAT_Values['targetLabels']
    fig, ax = plt.subplots()

    # plot the histplot and the kde
    ax = sns.histplot(listResVal, kde=True, legend=False, binwidth=binwidth, label="Residuals kde curve") #
    for k in ['right', 'top']:
        ax.spines[k].set_visible(False)

    plt.setp(ax.patches, linewidth=0)
    # plt.title(title, fontsize=fontsize)
    plt.xlabel(x, fontsize=fontsize)
    plt.ylabel("Tested samples (" + a + ")", fontsize=fontsize)

    plt.figure(1)
    if setxLim:
        xLim = (setxLim[0], setxLim[1])
    else:
        xLim = (min(arr), max(arr))
    plt.xlim(xLim)

    x = np.linspace(min(arr), max(arr), 100)
    t = np.linspace(-300, 300, 100)
    dx = binwidth
    scale = len(arr) * dx

    # plot the gaussian
    plt.plot(t, norm.pdf(t, mean, sigma) * scale, color='red', linestyle='dashed', label="Gaussian curve")
    plt.legend()

    ref_prefix = displayParams["ref_prefix"]

    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'VISU/Residuals/' + studyFolder
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/' + 'Combined' + '-' + studyFolder + extra + '.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()

    return mean, variance

def reportCV_Residuals_All(models, means, variances, displayParams, DBpath):

    # track mean and variance of residuals
    ResidualsDf = pd.DataFrame(columns=['mean', 'variance'], index=models)
    ResidualsDf['mean'] = means
    ResidualsDf['variance'] = variances
    sortedDf = ResidualsDf.sort_values('mean', ascending=False)
    AllDfs = [ResidualsDf, sortedDf]
    sheetNames = ['Residuals_MeanVar', 'Sorted_Residuals_MeanVar']

    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'RECORDS/'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        with pd.ExcelWriter(outputFigPath + reference[:-6] + '_CV_Residuals_All' + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)

def RUN_CombinedResiduals(studies_GS_FS, studies_NBest, studies_Blender, displayParams, FORMAT_Values, DBpath, n, NBestScore):

    models, means, variances = analyzeCVResiduals(studies_GS_FS)
    reportCV_Residuals_All(models, means, variances, displayParams, DBpath)

    BLName = studies_Blender[0].GSName

    plotCVResidualsGaussian_Combined(studies_NBest, displayParams, FORMAT_Values, DBpath,
                                     studyFolder='GaussianPlot_NBest', NBest=True)
    plotCVResidualsGaussian_Combined(studies_Blender, displayParams, FORMAT_Values, DBpath,
                                     studyFolder='GaussianPlot_Combined_' + BLName, Blender=True)
    plotCVResidualsGaussian_Combined(studies_GS_FS, displayParams, FORMAT_Values, DBpath,
                                     studyFolder='GaussianPlot_groupedModels')
    plotCVResidualsGaussian_indiv(studies_GS_FS, displayParams, FORMAT_Values, DBpath,
                                  studyFolder='GaussianPlot_indivModels')
    plotCVResidualsGaussian_indiv(studies_GS_FS, displayParams, FORMAT_Values, DBpath,
                                  studyFolder='GaussianPlot_indiv_'+ BLName, studies_Blender = studies_Blender)


    if displayParams['plot_all']:
        plotCVResidualsHistogram(studies_GS_FS, displayParams, FORMAT_Values, DBpath,
                                 studyFolder='Histplot_indivModels')
        plotCVResidualsHistogram_Combined(studies_GS_FS, displayParams, FORMAT_Values, DBpath,
                                          studyFolder='Histplot_groupedModels')
        plotCVResidualsHistogram_Combined(studies_Blender, displayParams, FORMAT_Values, DBpath,
                                          studyFolder='Histplot_NBest_' + str(n) + '_' + NBestScore, blended=True)



