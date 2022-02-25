import numpy as np
import matplotlib.pyplot as plt


def MetricsSummaryPlot(models, displayParams, metricLabels = ['bModelTrScore','bModelTeScore','bModelAcc','bModelMSE','bModelr2'],
            title ='Model Evaluations', xlabel='Evaluation Metric'):
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
    df = pd.DataFrame(metrics, index=labels, columns=metricLabels)
    tf = pd.DataFrame([means], index=labels, columns=metricLabels)

    fig = plt.figure(figsize=(10, 10))
    plt.title(title)
    sns.scatterplot(data=df.T) #, y=metricLabels, x=metrics, hue=metricLabels)
    # sns.catplot(data=df.T)
    # # sns.barplot(data=df, color = 'Grey')
    # sns.lineplot(data=df.T)
    # sns.lineplot(data=tf.T)

    sns.set_theme(style="whitegrid")
    plt.xlabel(xlabel)

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Metrics'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/Summary.png')
    if displayParams['showPlot']:
        plt.show()

    plt.close()

def averageMetric(models, metricLabels = ['bModelTrScore','bModelTeScore','bModelAcc','bModelMSE','bModelr2']):
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