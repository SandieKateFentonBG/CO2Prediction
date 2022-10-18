import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def RFEHyperparameterPlot3D(RFEs, displayParams, DBpath, reference, figFolder = 'RFE', figTitle = 'RFEPlot3d',
                            colorsPtsLsBest = ['b', 'g', 'c', 'y'],
                title = 'Influence of Feature Count on Model Performance',
                            xlabel = 'Feature Count', ylabel = 'Feature Count', zlabel = 'R2 Test score', size = [6,6],
                showgrid = False, log = False, max=True, ticks = False, lims = False):

    #todo : remove linear reg from models
    zlabel = 'R2 Test score'
    # metric = modelingParams['plotregulAccordingTo']
    # if metric == 'paramMeanMSETest':
    #     zlabel ='MSE'
    # if metric == 'paramMeanR2Test':
    #     zlabel ='R2'
    # Create figure and axes

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pts, labels = Construct3DPoints(RFEs)
    print(labels)
    figTitle = '/RegulPlot3d'
    xl, yl, zl = unpackResPts(pts)
    lines = unpackResLines(pts)
    if log:
        yl = list(np.log10(yl)) #[log(yl[i]) for i in range(len(yl))]
        figTitle = '/RegulPlot3d-logy'
        ylabel = 'log10(Regularization)'
        for i in range(len(lines)):
            lines[i][1] = list(np.log10(lines[i][1]))
    best = Highest3DPoints(pts, max = max)
    xlim, ylim, zlim = [np.amin(xl), np.amax(xl)], [np.amin(yl), np.amax(yl)], [np.amin(zl), np.amax(zl)]
    lim = xlim, ylim, zlim
    tick = np.arange(0, xlim[1]+1, 1).tolist(), np.arange(round(ylim[0], 0), round(ylim[1], 0), 100).tolist(), np.arange(round(zlim[0]-1, 0), round(zlim[1]+1, 0), 10).tolist()

    ax.scatter(xl, yl, zl, color=colorsPtsLsBest[0])
    ax.scatter(best[0], best[1], best[2], s = 50, c=colorsPtsLsBest[3])

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
    # ax.set_xlabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
             rotation_mode="anchor", size = 8)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.set_size_inches(size[0], size[1])
    ax.grid(showgrid)


    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference + figFolder
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + figTitle + '.png')


    if displayParams['showPlot']:
        plt.show()
    plt.close()