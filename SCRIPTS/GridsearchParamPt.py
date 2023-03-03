from HelpersVisualizer import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




def GS_ParameterPlot2D(GS_FSs, displayParams, DBpath, content = 'GS_FS', yLim = None,
                    score ='TestAcc', studyFolder = 'GS_FS/', combined = False):

    "to be done with single parameter"
    import seaborn as sns
    figFolder = 'GRID'
    figTitle = content + '_' + score + '_Plot2d'

    title = 'Influence of Feature Selection on Model Performance'

    xlabel = 'Feature Selection'
    xLabels = GS_FSs[0].learningDfsList
    ylabel = score

    pts, labels = GS_Construct3DPoints(GS_FSs, score)

    metric = [[pts[i][j][2] for j in range(len(pts[i]))] for i in range(len(pts))]
    param = [pts[0][i][1] for i in range(len(pts[0]))]


    df = pd.DataFrame(metric, index=labels, columns=xLabels).T
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
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + figFolder
        outputFigPath = path + folder + subFolder

        import os
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/' + figTitle + '.png')
        print("Image saved here :", outputFigPath + '/' + figTitle + '.png')

    if displayParams['showPlot']:
        plt.show()

    plt.close()


def GS_ParameterPlot3D(GS_FSs, displayParams, DBpath, content = 'GS_FS', yLim = None,
                    score ='TestAcc', colorsPtsLsBest=['b', 'g', 'c', 'y', 'r'], size=[6, 6],
                    showgrid=False,  maxScore=True, absVal = False, ticks=False, lims=False,
                    studyFolder = 'GS_FS/', combined = False):

    figFolder = 'GRID'
    figTitle = content + '_' + score + '_Plot3d'

    ylabel = 'Feature Selection'
    yLabels = GS_FSs[0].learningDfsList

    zlabel = score
    title = 'Influence of Feature Selection on Model Performance'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pts, labels = GS_Construct3DPoints(GS_FSs, score)
    xl, yl, zl = unpackResPts(pts)
    vl = np.reshape(zl, (len(labels), len(yLabels)))
    lines = unpackResLines(pts)


    best = Highest3DPoints(pts, max=maxScore, absVal=absVal)

    xlim, ylim, zlim = [np.amin(xl), np.amax(xl)], [np.amin(yl), np.amax(yl)], [np.amin(zl), np.amax(zl)]
    lim = xlim, ylim, zlim


    tick = np.arange(0, xlim[1] + 1, 1).tolist(), \
           np.arange(0, ylim[1] + 1, 1).tolist(), \
           np.arange(round(zlim[0] - 1, 0),round(zlim[1] + 1, 0),10).tolist()

    ax.scatter(xl, yl, zl, color=colorsPtsLsBest[1])
    ax.scatter(best[0], best[1], best[2], s=50, c=colorsPtsLsBest[4])

    for i in range(len(lines)):
        xls, yls, zls = lines[i][0], lines[i][1], lines[i][2]
        plt.plot(xls, yls, zls, color=colorsPtsLsBest[3])
    ax.set_title(title)

    if lims:
        ax.set_xlim(lim[0][0], lim[0][1])
        ax.set_ylim(lim[1][0], lim[1][1])
        ax.set_zlim(lim[2][0], lim[2][1])
    if ticks:
        ax.set_zticks(tick[2])

    ax.set_xticks(tick[0])
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
             rotation_mode="anchor", size=8)

    ax.set_yticks(tick[1])
    ax.set_yticklabels(yLabels)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="left", va = 'bottom',
             rotation_mode="default", size=8)

    # ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.setp(ax.get_zticklabels(), size=8)

    fig.set_size_inches(size[0], size[1])
    ax.grid(showgrid)#color='.25', linestyle='-', linewidth=0.5

    if displayParams['archive']:
        if combined:
            reference = displayParams['ref_prefix'] + '_Combined/'
        else:
            reference = displayParams['reference']
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + figFolder
        outputFigPath = path + folder + subFolder

        import os
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/' + figTitle + '.png')
        print("Image saved here :", outputFigPath + '/' + figTitle + '.png')


    # reference = displayParams['reference']
    # if displayParams['archive']:
    #     path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + figFolder
    #     import os
    #     outputFigPath = path + folder + subFolder
    #     if not os.path.isdir(outputFigPath):
    #         os.makedirs(outputFigPath)
    #
    #     plt.savefig(outputFigPath + '/' + figTitle + '.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()


def heatmap(GS_FSs, displayParams, DBpath, content='GS_FS', score='TestAcc', studyFolder='GS_FS/', combined = False):

    figFolder = 'GRID'
    figTitle = content + '_' + score + '_heatmap'

    ylabel = 'Feature Selection'
    yLabels = GS_FSs[0].learningDfsList

    zlabel = score
    title = 'Influence of Feature Selection on Model Performance - (%s)' % score

    xLabel = 'Predictor'
    pts, xLabels = GS_Construct3DPoints(GS_FSs, score)
    xl, yl, zl = unpackResPts(pts)


    zl = [np.round(elem, 2) for elem in zl]

    results = np.reshape(zl, (len(xLabels), len(yLabels)))
    fig = plt.figure()
    fig, ax = plt.subplots()
    im = ax.imshow(results)


    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(yLabels)))
    ax.set_xticklabels(yLabels)
    ax.set_yticks(np.arange(len(xLabels)))
    ax.set_yticklabels(xLabels)


    df = pd.DataFrame(results, index=xLabels, columns=yLabels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(xLabels)):
        for j in range(len(yLabels)):
            text = ax.text(j, i, results[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()

    if displayParams['archive']:
        if combined:
            reference = displayParams['ref_prefix'] + '_Combined/'
        else:
            reference = displayParams['reference']
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + figFolder
        outputFigPath = path + folder + subFolder

        import os
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/' + figTitle + '.png')
        print("Image saved here :", outputFigPath + '/' + figTitle + '.png')

    # reference = displayParams['reference']
    # if displayParams['archive']:
    #     path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + figFolder
    #     import os
    #     outputFigPath = path + folder + subFolder
    #     if not os.path.isdir(outputFigPath):
    #         os.makedirs(outputFigPath)
    #
    #     plt.savefig(outputFigPath + '/' + figTitle + '.png')
    #     print(outputFigPath)
    if displayParams['showPlot']:
        plt.show()
    plt.close()