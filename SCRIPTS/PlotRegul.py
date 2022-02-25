
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

def regulPoints(models, metric = 'paramMeanScore'):
    allModels = []
    # for m in models:
    labels =[]
    for j in range(len(models)):
        labels.append(models[j]['model'])
        modelRes = []
        for i in range(len(models[j]['paramValues'])):
            paramRes = [j, models[j]['paramValues'][i], models[j][metric][i]]

            # for i in range(len(m['paramValues'])):
        #     paramRes = [m['model'],m['paramValues'][i],m['paramMeanScore'][i]]
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



def plotRegul3d(models, displayParams, metric ='paramMeanScore', colorsPtsLsBest = ['b', 'g', 'c'],
                title = 'Influence of Regularization on Model MSE', xlabel = 'Model', ylabel = 'Regularization', zlabel ='MSE', size = [6,6],
                showgrid = False, max=False, ticks = False, lims = False):

    # Create figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    regPts, labels = regulPoints(models)

    xl, yl, zl = unpackResPts(regPts)
    lines = unpackResLines(regPts)
    best = regulBestPt(regPts, max=False)
    xlim, ylim, zlim = [np.amin(xl), np.amax(xl)], [np.amin(yl), np.amax(yl)], [np.amin(zl), np.amax(zl)]
    lim = xlim, ylim, zlim
    ticks = np.arange(0, xlim[1]+1, 1).tolist(), np.arange(round(ylim[0], 0), round(ylim[1], 0), 50).tolist(), np.arange(round(zlim[0]-1, 0), round(zlim[1]+1, 0), 0.5).tolist()

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
        print('ticks')
        ax.set_xticks(ticks[0])
        ax.set_yticks(ticks[1])
        ax.set_zticks(ticks[2])
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
             rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.set_size_inches(size[0], size[1])
    ax.grid(showgrid)

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Regul'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/RegulPlot' + '.png')
    if displayParams['showPlot']:
        plt.show()
    plt.close()

def plotReguls2D(models):
    import seaborn as sns
    regPts, labels = regulPoints(models)
    metric = [[regPts[i][j][2] for j in range(len(regPts[i]))] for i in range(len(regPts))]
    df = pd.DataFrame(metric, index=labels, columns=[regPts[0][i][1] for i in range(len(regPts[0]))]).T
    sns.lineplot(data=df)
    sns.set_theme(style="whitegrid")
    plt.show()
    plt.close()

