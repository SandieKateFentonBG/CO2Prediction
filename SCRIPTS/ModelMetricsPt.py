import numpy as np
import matplotlib.pyplot as plt


def MetricsSummaryPlot(GSs, displayParams, DBpath, metricLabels=['TrainScore', 'TestScore', 'TestAcc', 'TestMSE'],
                       title='Model Evaluations', ylabel='Evaluation Metric', fontsize=14, scatter=False):

    if displayParams["archive"] or displayParams["showPlot"]:
        sortedMod = sortGridResults(GSs, metric ='TestAcc', highest = True)
        MetricsPt(sortedMod, displayParams, DBpath, metricLabels=metricLabels,
              title=title, ylabel=ylabel, fontsize=fontsize, scatter=scatter)
        MetricsPt(sortedMod, displayParams, DBpath, metricLabels=metricLabels[0:2],
              title=title, ylabel=ylabel, fontsize=fontsize, scatter=scatter)
        MetricsPt(sortedMod, displayParams, DBpath, metricLabels = metricLabels[2:3],
              title=title, ylabel=ylabel, fontsize=fontsize, scatter=scatter)
        MetricsPt(sortedMod, displayParams, DBpath, metricLabels=metricLabels[3:],
              title=title, ylabel=ylabel, fontsize=fontsize, scatter=scatter)


def MetricsPt(GSs, displayParams, DBpath, metricLabels=['TrainScore', 'TestScore', 'TestAcc', 'TestMSE'],
              title='Model Evaluations', ylabel='Evaluation Metric', fontsize=14, scatter=True):
    import pandas as pd
    import seaborn as sns

    if displayParams['showPlot'] or displayParams['archive']:

        labels = []
        metrics = []
        for m in GSs:
            label = m.modelPredictor
            metric = [m.__getattribute__(lab) for lab in metricLabels]

            labels.append(label)
            metrics.append(metric)

        df = pd.DataFrame(metrics, index=list(range(len(labels))), columns=metricLabels)

        fig = plt.figure(figsize=(10, 12))

        plt.title(title, fontsize=fontsize)

        sns.lineplot(data=df)
        plt.xticks(list(range(len(labels))), labels, rotation=25, ha="right",
                   rotation_mode="anchor", size=fontsize)
        sns.set_theme(style="whitegrid")
        plt.ylabel(ylabel, fontsize=fontsize)

        name = ''
        for i in range(len(metricLabels)):
            name += metricLabels[i]

        reference = displayParams['reference']
        if displayParams['archive']:

            path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/GS/Metrics'
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + name + '.png')

        if displayParams['showPlot']:
            plt.show()
        fig.tight_layout()
        plt.close()


def averageMetric(models, metricLabels = ['bModelTrR2','bModelTeR2','bModelAcc','bModelMSE','bModelr2']):
    means = []
    stdvs = []
    for label in metricLabels:
        single = []
        for m in models:
            single.append(m[label])
        av = np.mean(single)
        st = np.std(single)
        means.append(av)
        stdvs.append(st)

    return means, stdvs



def sortGridResults(GSs, metric ='TestAcc', highest = True):

    # x is one elem of the list
    if metric == 'TestAcc':
        return sorted(GSs, key=lambda x: x.TestAcc, reverse=highest)
    if metric == 'TestMSE':
        return sorted(GSs, key=lambda x: x.TestMSE, reverse=highest)
    if metric == 'TestR2':
        return sorted(GSs, key=lambda x: x.TestR2, reverse=highest)
