import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def coefPdDisplay(features,values, displayParams):

    table = pd.DataFrame(values, columns=["Coefficients"], index=features)
    if displayParams['showResults']:
        print(table)

    return table

def coefBarDisplay(coefPd,displayParams):

    if displayParams['showResults']:
        coefPd.plot(kind="barh", figsize=(9, 7))
        plt.title("Assessing coefficients")
        plt.axvline(x=0, color=".5")
        plt.subplots_adjust(left=0.3)
        plt.show()

def coefBarDisplayAll(models, displayParams, df=None, yLim = 0.05):
    import numpy
    linModels = [m for m in models if m['Linear']==True] #only works/makes sense for linear models
    fig = plt.figure(figsize=(20, 10))
    idx = 1
    count = int(len(linModels)/2)
    for m in linModels:
        f, v = modelWeightsList(displayParams['Target'], m['features'], m['bModelWeights'], df)
        plt.subplot(count, 2, idx)
        plt.bar(numpy.arange(len(f)),
                v,
                align='center',
                color='red')
        plt.xticks(numpy.arange(len(f)), f, rotation=25, ha="right",
                 rotation_mode="anchor", size = 5)
        if yLim:
            plt.ylim(-yLim, yLim)
        plt.ylabel(m['model']) #+ 'Weights'
        idx += 1
    title = 'Feature Importance'
    fig.suptitle(title, fontsize="x-large")

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Coef'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/ModelsCoefImportance.png')
    if displayParams['showPlot']:
        plt.show()
    plt.close()

def listWeight(models):
    weights = []
    labels = []
    for i in range(len(models[0]['bModelWeights'])):
        single = []
        for m in models:
            single.append(m['bModelWeights'][i])
        weights.append(single)
    for m in models:
        labels.append(m['model'])

    return weights, labels

def averageWeight(models):
    means = []
    stdvs = []
    for i in range(len(models[0]['bModelWeights'])):
        single = []
        for m in models:
            single.append(m['bModelWeights'][i])
        av = np.mean(single)
        st = np.std(single)
        means.append(av)
        stdvs.append(st)

    return means, stdvs

def sortedListAccordingToGuide(guide, list1, list2=None):

    sortedG = sorted(guide)
    sortedL1 = [x for _, x in sorted(zip(guide, list1))]
    if list2:
        sortedL2 = [x for _, x in sorted(zip(guide, list2))]
        return sortedG, sortedL1, sortedL2
    return sortedG, sortedL1


def coefBarDisplayMean(models, displayParams, sorted = True, yLim = None):
    import numpy
    linModels = [m for m in models if m['Linear']==True] #only works/makes sense for linear models
    means, stdvs = averageWeight(linModels)
    labels = linModels[0]['features']
    if sorted:
        means, stdvs, labels = sortedListAccordingToGuide(means, stdvs, labels)

    fig = plt.figure(figsize=(20, 10))
    # plt.grid()
    plt.bar(numpy.arange(len(means)), means, align='center', color='red', yerr = stdvs)
    plt.xticks(numpy.arange(len(labels)), labels, rotation=25, ha="right",rotation_mode="anchor", size = 8)
    if yLim:
        plt.ylim(-yLim, yLim)
    plt.ylabel('Weights')

    title = 'Feature Importance'
    fig.suptitle(title, fontsize="x-large")

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Coef'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/CoefImportance.png')
    if displayParams['showPlot']:
        plt.show()
    plt.close()


def WeightsPlot(models):

    import pandas as pd

    import numpy
    import seaborn as sns


    linModels = [m for m in models if m['Linear']==True] #only works/makes sense for linear models
    weights, modelLabels = listWeight(linModels)
    features = linModels[0]['features']
    lineTable = pd.DataFrame(weights, columns=modelLabels, index=features)
    barTable = lineTable.T
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=lineTable)
    sns.barplot(data=barTable,  palette="Blues_d")
    plt.xticks(numpy.arange(len(features)), features, rotation=25, ha="right", rotation_mode="anchor", size=8)

    plt.show()
    plt.close()

    #todo ; add multiple bars/ hue / order along increasing mean value
    # tab = barTable.loc[modelLabels[3]]
    # sns.barplot(x=features, y = weights, hue =modelLabels, data=table) #", x=modelLabels, y=weights, hue=weights
    # sns.catplot(data=table)

