import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def xGS_averageWeight(GSs, key = 'WeightsScaled'):
    means = []
    stdvs = []
    for i in range(len(GSs[0].__getattribute__(key))):
        single = []
        for m in GSs:
            single.append(m.__getattribute__(key)[i])
        av = np.mean(single)
        st = np.std(single)
        means.append(av)
        stdvs.append(st)

    return means, stdvs

def GS_emptyWeights(df, target): #keys = df.keys()
    # weights = list(df.keys()[0: -1])+['intercept']
    weights = list(df.keys()) #+['intercept']

    weights.remove(target[0])
    weightsDict = dict()
    for w in weights:
        weightsDict[w]= 0

    return weightsDict

def GS_modelWeightsDict(target, features, weights, df):

    weightsDict = GS_emptyWeights(df, target)

    for i, j in zip(features, weights):
        weightsDict[i] = j
    return weightsDict

def GS_modelWeightsList(target, features, weights, df):
    weightsDict = GS_modelWeightsDict(target, features, weights, df=df)
    ks = list(weightsDict.keys())
    ws = list(weightsDict.values())

    return ks, ws

def GS_sortedListAccordingToGuide(guide, list1, list2=None):

    sortedG = sorted(guide)
    sortedL1 = [x for _, x in sorted(zip(guide, list1))]
    if list2:
        sortedL2 = [x for _, x in sorted(zip(guide, list2))]
        return sortedG, sortedL1, sortedL2
    return sortedG, sortedL1

def GS_listWeight(GSs, key = 'WeightsScaled'):
    weights = []
    labels = []
    for i in range(len(GSs[0].__getattribute__(key))):
        single = []
        for m in GSs:
            single.append(m.__getattribute__(key)[i])

        weights.append(single)

    for m in GSs:

        lab = str(m.modelPredictor) + '-' + m.selectorName
        labels.append(lab)


        plt.ylabel(m.predictorName + m.selectorName) #+ 'Weights'

    return weights, labels


def GS_averageWeight(target, GS_FSs, key = 'WeightsScaled', df = None):

    #todo : change the GS FS s for Gs Fs !!
    means = []
    stdvs = []

    LList = []
    VList =  []
    for GS_FS in GS_FSs:
        for learningDflabel in GS_FS.learningDfsList:
            GS = GS_FS.__getattribute__(learningDflabel)
            labelLs,valueLs = GS_modelWeightsList(target, GS.selectedLabels, GS.__getattribute__(key), df) #54
            VList.append(valueLs) #7

    for i in range(len(VList[0])): #todo : why vlist 0
        single = []
        for elem in VList:
            single.append(elem[i])


        av = np.mean(single)
        st = np.std(single)
        means.append(av)
        stdvs.append(st)

    return means, stdvs

def GS_averageWeight_NBest(target, BlendModel, key = 'WeightsScaled', df = None):

    #todo : change the GS FS s for Gs Fs !!
    means = []
    stdvs = []

    LList = []
    VList =  []

    for Model in BlendModel.modelList:

            labelLs,valueLs = GS_modelWeightsList(target, Model.selectedLabels, Model.__getattribute__(key), df) #54
            VList.append(valueLs) #7

    for i in range(len(VList[0])): #todo : why vlist 0
        single = []
        for elem in VList:
            single.append(elem[i])


        av = np.mean(single)
        st = np.std(single)
        means.append(av)
        stdvs.append(st)

    return means, stdvs


def GS_WeightsBarplotAll(GS_FSs, GS_FSs_for_mean, DBpath, displayParams, target, content='', df_for_empty_labels=None,
                         yLim = None, sorted = True, key = 'WeightsScaled', studyFolder='GS_FS/'):
    import numpy
    import math
    # linModels = [GS for GS in GSs if GS.isLinear==True] #only works/makes sense for linear models
    linModels = GS_FSs

    length = 0
    for GS_FS in GS_FSs:
        length += len(GS_FS.learningDfsList)

    fig = plt.figure(figsize=(20, 10))
    idx = 1
    count = math.ceil(length/2)
    if sorted:
        meanWeights, _ = GS_averageWeight(target, GS_FSs_for_mean, key = 'WeightsScaled', df = df_for_empty_labels)

    for GS_FS in GS_FSs:  # ,LR_LASSO_FS_GS, LR_RIDGE_FS_GS, LR_ELAST_FS_GS
        for learningDflabel in GS_FS.learningDfsList:
            GS = GS_FS.__getattribute__(learningDflabel)
            f, v = GS_modelWeightsList(target, GS.selectedLabels, GS.__getattribute__(key), df_for_empty_labels)

            if sorted:
                _, v, f = GS_sortedListAccordingToGuide(meanWeights, v, f)

            plt.subplot(count, 2, idx)
            plt.bar(numpy.arange(len(f)), v, align='center', color="Blue", data=f)
            plt.xticks(numpy.arange(len(f)), f, rotation=25, ha="right",
                     rotation_mode="anchor", size = 5)

            if yLim:
                plt.ylim(-yLim, yLim)

            plt.ylabel(GS.predictorName + '_' + GS.selectorName) #+ 'Weights'
            #plt.ylabel(m.predictorName) #+ 'Weights'
            idx += 1
    title = 'Feature Importance - Scaled weights'
    fig.suptitle(title, fontsize="x-large")

    reference = displayParams['reference']
    if displayParams['archive']:

        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'Weights'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/WeightsBarplot' + content +'.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()

def unpack_GS_FS(GS_FSs):

    unpacked = []
    for GS_FS in GS_FSs:
        for learningDflabel in GS_FS.learningDfsList:
            GS = GS_FS.__getattribute__(learningDflabel)
            unpacked.append(GS)

    return unpacked



def GS_WeightsSummaryPlot(GS_FSs, GS_FSs_for_mean, target, df_for_empty_labels, displayParams, DBpath, content='', sorted=True, yLim=None,
                          fontsize=14,  studyFolder='GS_FS/'):

    import pandas as pd
    import numpy
    import seaborn as sns

    inv_weights = []
    modelLabels = []
    for GS_FS in GS_FSs:
        for learningDflabel in GS_FS.learningDfsList:

            GS = GS_FS.__getattribute__(learningDflabel)
            labelLs,valueLs = GS_modelWeightsList(target, GS.selectedLabels, GS.__getattribute__('WeightsScaled'), df_for_empty_labels) #54
            inv_weights.append(valueLs) #7
            #todo - check - changed naming here for labels
            lab = GS.predictorName + '-' + GS.selectorName
            # lab = str(GS.modelPredictor) + '-' + GS.selectorName
            modelLabels.append(lab)

    weights = []
    for i in range(len(inv_weights[0])): #54
        single = []
        for j in range(len(inv_weights)) : #7
            single.append(inv_weights[j][i])
        weights.append(single)

    meanWeights, stdvs = GS_averageWeight(target, GS_FSs_for_mean, key = 'WeightsScaled', df = df_for_empty_labels)

    features = list(df_for_empty_labels.keys())
    features.remove(target[0])

    if sorted:
        meanWeights, weights, features = GS_sortedListAccordingToGuide(meanWeights, weights, features)


    lineTable = pd.DataFrame(weights, columns=modelLabels, index=features)

    test = pd.DataFrame(meanWeights, index=features)
    barTable = test.T

    fig = plt.figure(figsize=(12, 10))
    plt.title("Feature relative weights in calibrated models.", fontsize=fontsize)
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

        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'Weights'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/WeightsLinPlot' + content + '.png')
    if displayParams['showPlot']:
        plt.show()

    plt.close()

def GS_WeightsSummaryPlot_NBest(BlendModel, target, df_for_empty_labels, displayParams, DBpath, content='',
                          sorted=True, yLim=None,
                          fontsize=14, studyFolder='GS_FS/'):

    import pandas as pd
    import numpy
    import seaborn as sns

    inv_weights = []
    modelLabels = []

    for Model in BlendModel.modelList:

        labelLs, valueLs = GS_modelWeightsList(target, Model.selectedLabels, Model.__getattribute__('WeightsScaled'),
                                               df_for_empty_labels)  # 54
        inv_weights.append(valueLs)  # 7
        # todo - check - changed naming here for labels
        lab = Model.predictorName + '-' + Model.selectorName

        modelLabels.append(lab)

    weights = []
    for i in range(len(inv_weights[0])):  # 54
        single = []
        for j in range(len(inv_weights)):  # 7
            single.append(inv_weights[j][i])
        weights.append(single)

    meanWeights, stdvs = GS_averageWeight_NBest(target, BlendModel, key='WeightsScaled', df=df_for_empty_labels)

    features = list(df_for_empty_labels.keys())
    features.remove(target[0])

    if sorted:
        meanWeights, weights, features = GS_sortedListAccordingToGuide(meanWeights, weights, features)



    lineTable = pd.DataFrame(weights, columns=modelLabels, index=features)

    test = pd.DataFrame(meanWeights, index=features)
    barTable = test.T

    fig = plt.figure(figsize=(12, 10))
    plt.title("Feature relative weights in calibrated models.", fontsize=fontsize)
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=lineTable)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.barplot(data=barTable, palette="bwr", ci=None)  # cmap="Blues_d"palette="gwr"
    plt.xticks(numpy.arange(len(features)), features, rotation=90, ha="right", rotation_mode="anchor",
               size=fontsize)
    if yLim:
        plt.ylim(-yLim, yLim)
    plt.ylabel('Weights', fontsize=fontsize)
    plt.xlabel('Features', fontsize=fontsize)
    fig.tight_layout()

    reference = displayParams['reference']
    if displayParams['archive']:

        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'Weights'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/WeightsPlot' + content + '_NBest' + '.png')
    if displayParams['showPlot']:
        plt.show()

    plt.close()



