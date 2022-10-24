import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def averageWeight(GSs, key = 'WeightsScaled'):
    means = []
    stdvs = []
    for i in range(len(GSs[0].__getattribute__(key))):
        print(len(GSs[0].__getattribute__(key)))
        single = []
        for m in GSs:
            single.append(m.__getattribute__(key)[i])
        av = np.mean(single)
        st = np.std(single)
        means.append(av)
        stdvs.append(st)
        print(i, single)
    print(means, stdvs)
    return means, stdvs

def emptyWeights(df, target): #keys = df.keys()
    # weights = list(df.keys()[0: -1])+['intercept']
    weights = list(df.keys()) #+['intercept']
    weights.remove(target)
    weightsDict = dict()
    for w in weights:
        weightsDict[w]= 0

    return weightsDict

def modelWeightsDict(target, features, weights, df=None):
    if df:
        weightsDict = emptyWeights(df, target)
    else:
        weightsDict = dict()
    for i, j in zip(features, weights):
        weightsDict[i] = j
    return weightsDict

def modelWeightsList(target, features, weights, df=None):
    weightsDict = modelWeightsDict(target, features, weights, df=None)
    ks = list(weightsDict.keys())
    ws = list(weightsDict.values())
    return ks, ws

def sortedListAccordingToGuide(guide, list1, list2=None):

    sortedG = sorted(guide)
    sortedL1 = [x for _, x in sorted(zip(guide, list1))]
    if list2:
        sortedL2 = [x for _, x in sorted(zip(guide, list2))]
        return sortedG, sortedL1, sortedL2
    return sortedG, sortedL1

def WeightsBarplotAll(GSs, DBpath, displayParams, target , df=None, yLim = None, sorted = True, key = 'WeightsScaled' ):
    import numpy
    import math
    # linModels = [GS for GS in GSs if GS.isLinear==True] #only works/makes sense for linear models
    linModels = GSs

    fig = plt.figure(figsize=(20, 10))
    idx = 1
    count = math.ceil(len(linModels)/2)
    meanWeights, _ = averageWeight(linModels)

    for m in linModels:
        f, v = modelWeightsList(target, m.selectedLabels, m.__getattribute__(key), df)
        if sorted:
            _, v, f = sortedListAccordingToGuide(meanWeights, v, f)
        plt.subplot(count, 2, idx)

        plt.bar(numpy.arange(len(f)), v, align='center', color="Blue", data=f)
        plt.xticks(numpy.arange(len(f)), f, rotation=25, ha="right",
                 rotation_mode="anchor", size = 5)

        if yLim:
            plt.ylim(-yLim, yLim)
        plt.ylabel(m.predictorName) #+ 'Weights'
        idx += 1
    title = 'Feature Importance'
    fig.suptitle(title, fontsize="x-large")

    reference = displayParams['reference']
    if displayParams['archive']:

        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/GS/Weights'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/ModelsCoefImportance.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()


def listWeight(GSs, key = 'WeightsScaled'):
    weights = []
    labels = []
    for i in range(len(GSs[0].__getattribute__(key))):
        single = []
        for m in GSs:
            single.append(m.__getattribute__(key)[i])

        weights.append(single)
    for m in GSs:
        labels.append(m.modelPredictor)

    return weights, labels

def WeightsSummaryPlot(GSs, displayParams, DBpath, sorted=True, yLim=None, fontsize=14):

    import pandas as pd
    import numpy
    import seaborn as sns

    # linModels = [GS for GS in GSs if GS.isLinear==True] #only works/makes sense for linear models
    linModels = GSs

    weights, modelLabels = listWeight(linModels)
    meanWeights, stdvs = averageWeight(linModels)

    features = linModels[0].selectedLabels
    if sorted:
        meanWeights, weights, features = sortedListAccordingToGuide(meanWeights, weights, features)
    lineTable = pd.DataFrame(weights, columns=modelLabels, index=features)
    barTable = lineTable.T

    fig = plt.figure(figsize=(12, 10))
    plt.title("Feature relative weights in calibrated linear models.", fontsize=fontsize)
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=lineTable)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.barplot(data=barTable, palette="bwr", ci=None) #cmap="Blues_d"palette="gwr"
    plt.xticks(numpy.arange(len(features)), features, rotation=90, ha="right", rotation_mode="anchor", size=fontsize)
    if yLim:
        plt.ylim(-yLim, yLim)
    plt.ylabel('Weights', fontsize=fontsize)
    plt.xlabel('Features', fontsize=fontsize)
    fig.tight_layout()

    reference = displayParams['reference']
    if displayParams['archive']:

        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/GS/Weights'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/meanCoefImportance.png')
    if displayParams['showPlot']:
        plt.show()

    plt.close()
