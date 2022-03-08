import numpy as np
import matplotlib.pyplot as plt


def MetricsSummaryPlot(models, displayParams, metricLabels = ['bModelTrR2','bModelTeR2','bModelAcc','bModelMSE'],
            title ='Model Evaluations', ylabel='Evaluation Metric'):
    import pandas as pd
    import seaborn as sns

    means, _ = averageMetric(models, metricLabels)
    labels = []
    metrics = []
    for m in models:
        label = m['bModel']
        metric =[m[label] for label in metricLabels]
        # metric = [m['bModelTrScore'], m['bModelTeScore'], m['bModelAcc'], m['bModelMSE'],m['bModelr2']]
        labels.append(label)
        metrics.append(metric)
    df = pd.DataFrame(metrics, index=list(range(len(labels))), columns=metricLabels)

    fig = plt.figure(figsize=(10, 10))

    plt.title(title)
    # sns.scatterplot(data=df) #, y=metricLabels, x=metrics, hue=metricLabels)
    sns.lineplot(data=df)
    plt.xticks(list(range(len(labels))), labels, rotation=25, ha="right",
             rotation_mode="anchor", size = 8)
    sns.set_theme(style="whitegrid")
    plt.ylabel(ylabel)
    if len(metricLabels)>2 :
        name = 'Summary'
    else:
        name = metricLabels[0]
    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + str(displayParams['random_state']) +'/Metrics'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + name + '.png')
    if displayParams['showPlot']:
        plt.show()

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
