
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

def unpackResPts(results):
    xli = []
    yli = []
    zli = []
    for r in range(len(results)):
        for c in range(len(results[0])):
            # for d in range(len(results[0][0])):
            xli.append(results[r][c][0])
            yli.append(results[r][c][1])
            zli.append(results[r][c][2])
    xl, yl, zl = np.array(xli), np.array(yli), np.array(zli)
    return xl, yl, zl

def unpackResLines(results):
    rlist = []
    for r in range(len(results)):
        row = []
        x, y, z = [], [], []
        for c in range(len(results[0])):
            x.append(results[r][c][0])
            y.append(results[r][c][1])
            z.append(results[r][c][2])
            row = [x, y, z]
        rlist.append(row)
    return rlist

def removeLinReg(dc):

    update = []
    for m in dc:
        if m['param']:
            update.append(m)
    return update

def regulPoints(dc, metric = 'paramMeanMSETest'): #'paramMeanR2Test
    allModels = []
    # for m in models:
    labels =[]

    models = removeLinReg(dc)

    for j in range(len(models)):
        labels.append(models[j]['model'])
        modelRes = []
        for i in range(len(models[j]['paramValues'])):
            paramRes = [j, models[j]['paramValues'][i], models[j][metric][i]]

            modelRes.append(paramRes)
        allModels.append(modelRes)
    return allModels, labels

def regulBestPt(points, max = True):
    best = points[0][0]
    for i in range(len(points)):
        for j in range(len(points[0])):
            if max:
                if abs(points[i][j][2])>abs(best[2]):
                    best = points[i][j]
            else :
                if abs(points[i][j][2])<abs(best[2]):
                    best = points[i][j]
    return best

def plotRegul3D(modelWithParams, displayParams, modelingParams, colorsPtsLsBest = ['b', 'g', 'c'],
                title = 'Influence of Regularization on Model Performance', xlabel = 'Model', ylabel = 'Regularization', size = [6,6],
                showgrid = False, log = False, max=False, ticks = False, lims = False):

    #todo : remove linear reg from models
    zlabel = 'Score'
    metric = modelingParams['plotregulAccordingTo']
    if metric == 'paramMeanMSETest':
        zlabel ='MSE'
    if metric == 'paramMeanR2Test':
        zlabel ='R2'
    # Create figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    regPts, labels = regulPoints(modelWithParams, metric)
    print(labels)
    figTitle = '/RegulPlot3d'
    xl, yl, zl = unpackResPts(regPts)
    lines = unpackResLines(regPts)
    if log:
        yl = list(np.log10(yl)) #[log(yl[i]) for i in range(len(yl))]
        figTitle = '/RegulPlot3d-logy'
        ylabel = 'log10(Regularization)'
        for i in range(len(lines)):
            lines[i][1] = list(np.log10(lines[i][1]))
    best = regulBestPt(regPts, max=False)
    xlim, ylim, zlim = [np.amin(xl), np.amax(xl)], [np.amin(yl), np.amax(yl)], [np.amin(zl), np.amax(zl)]
    lim = xlim, ylim, zlim
    tick = np.arange(0, xlim[1]+1, 1).tolist(), np.arange(round(ylim[0], 0), round(ylim[1], 0), 100).tolist(), np.arange(round(zlim[0]-1, 0), round(zlim[1]+1, 0), 10).tolist()

    ax.scatter(xl, yl, zl, color=colorsPtsLsBest[0])
    ax.scatter(best[0], best[1], best[2], s = 50, c=colorsPtsLsBest[2])

    for i in range(len(lines)):
        xls, yls, zls = lines[i][0], lines[i][1], lines[i][2]
        plt.plot(xls, yls, zls, color=colorsPtsLsBest[1])
    ax.set_title(title)
    if lims:
        ax.set_xlim(lim[0][0], lim[0][1])
        ax.set_ylim(lim[1][0], lim[1][1])
        ax.set_zlim(lim[2][0], lim[2][1])
    if ticks:
        ax.set_xticks(tick[0])
        ax.set_yticks(tick[1])
        ax.set_zticks(tick[2])
    ax.set_xticklabels(labels)
    # ax.set_xlabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
             rotation_mode="anchor", size = 8)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.set_size_inches(size[0], size[1])
    ax.grid(showgrid)

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Regul'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + figTitle + '.png')
    if displayParams['showPlot']:
        plt.show()
    plt.close()

def plotRegul2D(modelWithParams,  displayParams, modelingParams, yLim = None, title ='Influence of Regularization on Model Performance',
                xlabel='Regularization', log = False):
    import seaborn as sns

    # todo : remove linear reg from models
    ylabel = 'score'
    metric = modelingParams['plotregulAccordingTo']
    if metric == 'paramMeanMSETest':
        ylabel ='MSE'
    if metric == 'paramMeanR2Test':
        ylabel ='R2'

    regPts, labels = regulPoints(modelWithParams, metric)
    metric = [[regPts[i][j][2] for j in range(len(regPts[i]))] for i in range(len(regPts))]
    param = [regPts[0][i][1] for i in range(len(regPts[0]))]
    figTitle = '/RegulPlot2d'

    if log:
        param = list(np.log10(param))
        figTitle = '/RegulPlot2d-logy'
        xlabel = 'log10(Regularization)'

    df = pd.DataFrame(metric, index=labels, columns=param).T
    fig = plt.figure(figsize=(20, 10))
    plt.title(title)

    sns.lineplot(data=df)
    sns.set_theme(style="whitegrid")

    if yLim:
        plt.ylim(-yLim, yLim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Regul'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + figTitle + '.png')
    if displayParams['showPlot']:
        plt.show()

    plt.close()
