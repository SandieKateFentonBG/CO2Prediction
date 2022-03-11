import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

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

def WeightsDf(features, values, displayParams):

    table = pd.DataFrame(values, columns=["Coefficients"], index=features)
    if displayParams['showResults']:
        print(table)

    return table

def WeightsBarplot(coefPd, displayParams):

    if displayParams['showResults']:
        coefPd.plot(kind="barh", figsize=(9, 7))
        plt.title("Assessing coefficients")
        plt.axvline(x=0, color=".5")
        plt.subplots_adjust(left=0.3)
        plt.show()

def WeightsBarplotAll(models, displayParams, df=None, yLim = None, sorted = True, key = 'bModelWeightsScaled'):
    import numpy
    import math
    linModels = [m for m in models if m['Linear']==True] #only works/makes sense for linear models
    fig = plt.figure(figsize=(20, 10))
    idx = 1
    count = math.ceil(len(linModels)/2)
    meanWeights, _ = averageWeight(linModels)

    for m in linModels:
        f, v = modelWeightsList(displayParams['Target'], m['features'], m[key], df)
        # print('vl before', v)
        # if scaled:
        #     v = scaledList(v)
        #     print('vl after', v)
        if sorted:
            _, v, f = sortedListAccordingToGuide(meanWeights, v, f)
        plt.subplot(count, 2, idx)

        # plt.subplot(count, 2, idx)
        plt.bar(numpy.arange(len(f)), v, align='center', color="Blue", data=f)
        plt.xticks(numpy.arange(len(f)), f, rotation=25, ha="right",
                 rotation_mode="anchor", size = 5)
        #todo : ax.bar_label(f)#, loc
        if yLim:
            plt.ylim(-yLim, yLim)
        plt.ylabel(m['model']) #+ 'Weights'
        idx += 1
    title = 'Feature Importance'
    fig.suptitle(title, fontsize="x-large")

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + str(displayParams['random_state']) + '/Coef'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/ModelsCoefImportance.png')
    if displayParams['showPlot']:
        plt.show()
    plt.close()

def listWeight(models, key = 'bModelWeightsScaled'):
    weights = []
    labels = []
    for i in range(len(models[0][key])):
        single = []
        for m in models:
            single.append(m[key][i])
            # print(len(single))
            # # if scaled:
            # #     single = scaledList(single)
        weights.append(single)
    for m in models:
        labels.append(m['model'])

    return weights, labels

def averageWeight(models, key = 'bModelWeightsScaled'):
    means = []
    stdvs = []
    for i in range(len(models[0][key])):
        single = []
        for m in models:
            single.append(m[key][i])
        av = np.mean(single)
        st = np.std(single)
        means.append(av)
        stdvs.append(st)

    return means, stdvs

# def averageWeight(models, scaled = True):
#     means = []
#     stdvs = []
#     for i in range(len(models[0]['bModelWeights'])):
#         single = []
#         for m in models:
#             print(m['bModelWeights'])
#             scaledWeights = scaledList(m['bModelWeights'])
#             single.append(m['bModelWeights'][i])
#         av = np.mean(single)
#         st = np.std(single)
#         means.append(av)
#         stdvs.append(st)
#         if scaled :
#             means = scaledList(means)
#
#     return means, stdvs

# def scaledList(means):
#
#     vScaler = preprocessing.MinMaxScaler()
#     v_normalized = vScaler.fit_transform(np.array(means).reshape(-1, 1)).reshape(1, -1)
#     return v_normalized.tolist()[0]

def sortedListAccordingToGuide(guide, list1, list2=None):

    sortedG = sorted(guide)
    sortedL1 = [x for _, x in sorted(zip(guide, list1))]
    if list2:
        sortedL2 = [x for _, x in sorted(zip(guide, list2))]
        return sortedG, sortedL1, sortedL2
    return sortedG, sortedL1


def WeightsSummaryPlot(models, displayParams, sorted=True, yLim=None, fontsize=14):

    import pandas as pd
    import numpy
    import seaborn as sns

    linModels = [m for m in models if m['Linear'] == True]  # only works/makes sense for linear models
    weights, modelLabels = listWeight(linModels)
    meanWeights, stdvs = averageWeight(linModels)

    features = linModels[0]['features']
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
    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + str(displayParams['random_state']) + '/Coef'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/meanCoefImportance.png')
    if displayParams['showPlot']:
        plt.show()

    plt.close()



