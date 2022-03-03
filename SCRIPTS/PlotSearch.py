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
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Metrics'
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


def paramResiduals(modelWithParam, xTrain, yTrain, xTest, yTest, displayParams, bestParam = None, yLim = None , xLim = None, fontsize = None):

    from yellowbrick.regressor import ResidualsPlot
    title = 'Residuals for ' + str(modelWithParam)
    if bestParam:
        title += '- BEST PARAM (%s) ' % bestParam
    fig = plt.figure(figsize=(10,5))#
    if fontsize:
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Predicted Value ', fontsize=14)
        plt.ylabel('Residuals', fontsize=14)
    ax = plt.gca()
    if yLim:
        plt.ylim(yLim[0], yLim[1])
    if xLim:
        plt.xlim(xLim[0], xLim[1])
    visualizer = ResidualsPlot(modelWithParam, title = title, fig=fig, hist =True)#"frequency" qqplot = True
    visualizer.fit(xTrain, yTrain.ravel())  # Fit the training data to the visualizer
    visualizer.score(xTest, yTest.ravel())  # Evaluate the model on the test data
    # visualizer.hax.grid(False)

    resDict = {'bModelResTrR2': round(visualizer.train_score_, displayParams['roundNumber']),
               'bModelResTeR2': round(visualizer.test_score_, displayParams['roundNumber'])}

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Residuals'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        visualizer.show(outpath=outputFigPath + '/' + str(modelWithParam) + '.png')

    if displayParams['showPlot']:
        visualizer.show()

    visualizer.finalize()

    return resDict