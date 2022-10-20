from VisualizerHelpers import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def GSConstruct3DPoints(ResultsList, key = 'gamma', score = 'mean_test_r2'): #x : featureCount, y : valScore

    pts = []
    labels =[]

    for j in range(len(ResultsList)):


        labels.append(ResultsList[j].predictorName)
        modelRes = []
        for i in range(len(ResultsList[j].param_dict[key])): #x : gamma value
            paramRes = [j, ResultsList[j].param_dict[key][i], ResultsList[j].Grid.cv_results_[score][i]] #y : Score

            modelRes.append(paramRes)
        pts.append(modelRes)
    return pts, labels


def GSParameterPlot2D(GSs,  displayParams, DBpath, yLim = None,
                      paramKey ='gamma', score ='mean_test_r2', log = False):

    "to be done with single parameter"
    import seaborn as sns
    figFolder = 'Hyperparam'
    figTitle = paramKey + 'Plot2d'

    title = 'Influence of ' + paramKey + ' on Model Performance'

    xlabel = paramKey
    ylabel = score

    pts, labels = GSConstruct3DPoints(GSs)

    metric = [[pts[i][j][2] for j in range(len(pts[i]))] for i in range(len(pts))]
    param = [pts[0][i][1] for i in range(len(pts[0]))]
    # figTitle = '/RFEPlot2d'

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
    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/GS/' + figFolder
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + figTitle + '.png')

    if displayParams['showPlot']:
        plt.show()

    plt.close()

def GSParameterPlot3D(GSs, displayParams, DBpath,
                      colorsPtsLsBest=['b', 'g', 'c', 'y'], paramKey='gamma', score='mean_test_r2',
                      size=[6, 6], showgrid=False, log=False, maxScore=True, absVal = False,  ticks=False, lims=False):

    figFolder = 'Hyperparam'
    figTitle = 'RFEPlot3d'
    ylabel = paramKey
    zlabel = score
    title = 'Influence of ' + paramKey + ' on Model Performance'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pts, labels = GSConstruct3DPoints(GSs)

    xl, yl, zl = unpackResPts(pts)
    lines = unpackResLines(pts)
    if log:
        yl = list(np.log10(yl))  # [log(yl[i]) for i in range(len(yl))]
        figTitle += '-logy'
        ylabel = 'log10 ' + ylabel
        for i in range(len(lines)):
            lines[i][1] = list(np.log10(lines[i][1]))
    best = Highest3DPoints(pts, max=maxScore, absVal=absVal)
    if log:
        best = best[0], np.log10(best[1]), best[2]
    xlim, ylim, zlim = [np.amin(xl), np.amax(xl)], [np.amin(yl), np.amax(yl)], [np.amin(zl), np.amax(zl)]
    lim = xlim, ylim, zlim
    tick = np.arange(0, xlim[1] + 1, 1).tolist(), np.arange(round(ylim[0], 0), round(ylim[1], 0),
                                                            100).tolist(), np.arange(round(zlim[0] - 1, 0),
                                                                                     round(zlim[1] + 1, 0),
                                                                                     10).tolist()

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
    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/GS/' + figFolder
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + figTitle + '.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()