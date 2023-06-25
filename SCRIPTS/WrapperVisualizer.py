from HelpersVisualizer import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def Construct3DPoints(ResultsList): #x : featureCount, y : valScore

    pts = []
    labels =[]

    for j in range(len(ResultsList)):
        labels.append(ResultsList[j].method)
        modelRes = []
        for i in range(len(ResultsList[j].rfeHyp_featureCount)): #x : featureCount
            paramRes = [j, ResultsList[j].rfeHyp_featureCount[i], ResultsList[j].rfeHyp_checkScore[i]] #y : valScore

            modelRes.append(paramRes)
        pts.append(modelRes)
    return pts, labels

def RFEHyperparameterPlot2D(RFEs,  displayParams, DBpath, yLim = None, figTitle = 'RFEPlot2d',
                          title ='Influence of Feature Count on Model Performance', xlabel='Feature Count',
                            ylabel = 'R2 Test score', log = False, combined=False):

    import seaborn as sns

    pts, labels = Construct3DPoints(RFEs)
    metric = [[pts[i][j][2] for j in range(len(pts[i]))] for i in range(len(pts))]
    param = [pts[0][i][1] for i in range(len(pts[0]))]

    if log:
        param = list(np.log10(param))
        figTitle += '-logy'
        xlabel = 'log10 ' + xlabel

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
        if combined:
            reference = displayParams['ref_prefix'] + '_Combined/'
        else:
            reference = displayParams['reference']

        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/FS/WRAPPER'#reference + figFolder
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + figTitle + '.png')

    if displayParams['showPlot']:
        plt.show()

    plt.close()

def RFEHyperparameterPlot3D(RFEs, displayParams, DBpath, figTitle='RFEPlot3d',
                            colorsPtsLsBest=['b', 'g', 'c', 'y'],
                            title='Influence of Feature Count on Model Performance', ylabel='Feature Count',
                            zlabel='R2 Test score', size=[6, 6],
                            showgrid=False, log=False, max=True, ticks=False, lims=False, combined=False):


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pts, labels = Construct3DPoints(RFEs)
    print(labels)

    xl, yl, zl = unpackResPts(pts)
    lines = unpackResLines(pts)
    if log:
        yl = list(np.log10(yl))  # [log(yl[i]) for i in range(len(yl))]
        figTitle += '-logy'
        ylabel = 'log10 ' + ylabel
        for i in range(len(lines)):
            lines[i][1] = list(np.log10(lines[i][1]))
    best = Highest3DPoints(pts, max=max, absVal = False)
    if log:
        best = best[0], np.log10(best[1]), best[2]
    xlim, ylim, zlim = [np.amin(xl), np.amax(xl)], [np.amin(yl), np.amax(yl)], [np.amin(zl), np.amax(zl)]
    lim = xlim, ylim, zlim
    tick = np.arange(0, xlim[1] + 1, 1).tolist(), np.arange(round(ylim[0], 0), round(ylim[1], 0),
                                                            100).tolist(), np.arange(round(zlim[0] - 1, 0),
                                                                                     round(zlim[1] + 1, 0), 10).tolist()

    ax.scatter(xl, yl, zl, color=colorsPtsLsBest[0])
    ax.scatter(best[0], best[1], best[2], s=50, c=colorsPtsLsBest[3])

    for i in range(len(lines)):
        xls, yls, zls = lines[i][0], lines[i][1], lines[i][2]
        plt.plot(xls, yls, zls, color=colorsPtsLsBest[1])
    ax.set_title(title)
    if lims:
        ax.set_xlim(lim[0][0], lim[0][1])
        ax.set_ylim(lim[1][0], lim[1][1])
        ax.set_zlim(lim[2][0], lim[2][1])
    if ticks:
        ax.set_yticks(tick[1])
        ax.set_zticks(tick[2])
    ax.set_xticks(tick[0])
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
             rotation_mode="anchor", size=8)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.set_size_inches(size[0], size[1])
    ax.grid(showgrid)
    # reference = displayParams['reference']
    if displayParams['archive']:
        if combined:
            reference = displayParams['ref_prefix'] + '_Combined/'
        else:
            reference = displayParams['reference']

        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/FS/WRAPPER' #DBpath, "RESULTS/", reference + figFolder
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + figTitle + '.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()