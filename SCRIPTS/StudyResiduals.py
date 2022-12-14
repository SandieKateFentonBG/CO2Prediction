
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def mergeList(list):
    #merge multiplelists in single list

    return [j for i in list for j in i]

def AssembleStudyResiduals(studies):
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

def AssembleBlenderResiduals(studies_Blender):
    residualsDict = dict()
    residualsDict["all"] = []
    for blender in studies_Blender:
        for model in blender.modelList:
            residualsDict["all"].append(list(model.Resid))

    for k, v in residualsDict.items():
        residualsDict[k] = mergeList(v)

    return residualsDict

def plotResidualsHistogram(studies, displayParams, FORMAT_Values, DBpath, studyFolder ='Histplot'):

    residualsDict = AssembleStudyResiduals(studies)

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
            path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'VISU/' + studyFolder
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + k + '-'+ studyFolder +'.png')

        if displayParams['showPlot']:
            plt.show()

        plt.close()

def plotResidualsGaussian(studies, displayParams, FORMAT_Values, DBpath, studyFolder='GaussianPlot', binwidth=25,
                          setxLim=[-300, 300], fontsize=14):

    from scipy.stats import norm
    import seaborn as sns

    # assemble residuals
    residualsDict = AssembleStudyResiduals(studies)
    models, means, variances = [], [], []

    listResVal = mergeList(list(residualsDict.values()))
    resmin, resmax = min(listResVal), max(listResVal)
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

    for k, v in residualsDict.items():
        title = 'Residuals distribution for ' + k
        x = "Residuals [%s]" % FORMAT_Values['targetLabels']
        fig, ax = plt.subplots()

        # plot the histplot and the kde
        ax = sns.histplot(v, kde=True, legend=False, binwidth=binwidth, label="Residuals kde curve")
        plt.setp(ax.patches, linewidth=0)
        plt.title(title, fontsize=fontsize)
        plt.xlabel(x, fontsize=fontsize)
        plt.ylabel("Count", fontsize=fontsize)
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
            path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'VISU/' + studyFolder
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + k + '-' + studyFolder + '.png')

        if displayParams['showPlot']:
            plt.show()
        plt.close()

    return models, means, variances

def plotCombinedResidualsHistogram(studies, displayParams, FORMAT_Values, DBpath, studyFolder ='Histplot', blended = False):

    if blended : #only takes nBestmodels
        residualsDict = AssembleBlenderResiduals(studies)
    else : #takes all models
        residualsDict = AssembleStudyResiduals(studies)

    mergedList = mergeList(list(residualsDict.values()))

    title = 'Residuals distribution'
    x = "Residuals [%s]" % FORMAT_Values['targetLabels']
    fig, ax = plt.subplots()

    ax = sns.histplot(mergedList, kde=True, bins=14, binrange=(-100, 100), legend=False)
    plt.setp(ax.patches, linewidth=0)

    plt.title(title, fontsize=14)
    plt.xlabel(x, fontsize=14)

    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'VISU/' + studyFolder
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + 'Combined' + '-' + studyFolder + '.png')

    if displayParams['showPlot']:
        plt.show()

    plt.close()

def plotCombinedResidualsGaussian(studies, displayParams, FORMAT_Values, DBpath, studyFolder='GaussianPlot',
                                  binwidth=25,
                                  setxLim=[-300, 300], fontsize=14, blended = False):
    from scipy.stats import norm
    import seaborn as sns

    # assemble residuals
    if blended : #only takes nBestmodels
        residualsDict = AssembleBlenderResiduals(studies)
    else : #takes all models
        residualsDict = AssembleStudyResiduals(studies)

    listResVal = mergeList(list(residualsDict.values()))
    arr = np.array(listResVal)

    mean = np.mean(arr)

    variance = np.var(arr)
    sigma = np.sqrt(variance)

    resmin, resmax = min(listResVal), max(listResVal)
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

    # for k, v in residualsDict.items():
    title = 'Residuals distribution for '
    x = "Residuals [%s]" % FORMAT_Values['targetLabels']
    fig, ax = plt.subplots()

    # plot the histplot and the kde
    ax = sns.histplot(listResVal, kde=True, legend=False, binwidth=binwidth, label="Residuals kde curve")
    plt.setp(ax.patches, linewidth=0)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(x, fontsize=fontsize)
    plt.ylabel("Count", fontsize=fontsize)

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

    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'VISU/' + studyFolder
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + 'Combined' + '-' + studyFolder + '.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()

    return mean, variance

def ReportResiduals(models, means, variances, displayParams, DBpath):

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

        with pd.ExcelWriter(outputFigPath + reference[:-6] + '_ResidualsCombined' + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)

