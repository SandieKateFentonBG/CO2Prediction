def WeightsBarplotMean(models, displayParams, sorted = True, yLim = None):
    import numpy
    linModels = [m for m in models if m['Linear']==True] #only works/makes sense for linear models
    means, stdvs = averageWeight(linModels)
    labels = linModels[0]['features']
    if sorted:
        means, stdvs, labels = sortedListAccordingToGuide(means, stdvs, labels)

    fig = plt.figure(figsize=(20, 10))
    # plt.grid()
    plt.bar(numpy.arange(len(means)), means, align='center', color='red', yerr = stdvs, palette="Blues_d")
    plt.xticks(numpy.arange(len(labels)), labels, rotation=25, ha="right",rotation_mode="anchor", size = 8)
    if yLim:
        plt.ylim(-yLim, yLim)
    plt.ylabel('Weights')

    title = 'Feature Importance'
    fig.suptitle(title, fontsize="x-large")

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Coef'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/CoefImportance.png')
    if displayParams['showPlot']:
        plt.show()
    plt.close()

def WeightsSummaryPlot2(models, displayParams, sorted = True, yLim = None):

    import pandas as pd
    import numpy
    import seaborn as sns

    linModels = [m for m in models if m['Linear']==True] #only works/makes sense for linear models
    weights, modelLabels = listWeight(linModels)
    meanWeights, stdvs = averageWeight(linModels)
    features = linModels[0]['features']
    if sorted:
        meanWeights, weights, features = sortedListAccordingToGuide(meanWeights, weights, features)
    lineTable = pd.DataFrame(weights, columns=modelLabels, index=features)
    barTable = lineTable.T

    fig = plt.figure(figsize=(15, 10))
    # plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
    # plt.rcParams['ylabel.right'] = plt.rcParams['ylabel.labelright'] = True
    # plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
    plt.title("Feature Importance sorted according to mean value")
    plt.grid()
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(data=lineTable)
    sns.barplot(data=barTable,  palette="Blues_d")
    plt.xticks(numpy.arange(len(features)), features, rotation=90, ha="right", rotation_mode="anchor", size=8)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    if yLim:
        plt.ylim(-yLim, yLim)
    plt.ylabel('Weights')
    plt.xlabel('Features')
    plt.legend(loc='lower right')
    fig.tight_layout()
    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Coef'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/meanCoefImportance.png')
    if displayParams['showPlot']:
        plt.show()

    plt.close()


def WeightsSummaryPlotH(models, displayParams, sorted=True, yLim=None):
    import pandas as pd
    import numpy
    import seaborn as sns

    linModels = [m for m in models if m['Linear']==True] #only works/makes sense for linear models
    weights, modelLabels = listWeight(linModels)
    meanWeights, stdvs = averageWeight(linModels)
    features = ['a']+linModels[0]['features']

    if sorted:
        meanWeights, weights, features = sortedListAccordingToGuide(meanWeights, weights, features)
    lineTable = pd.DataFrame(weights, columns=modelLabels, index=features )
    print(lineTable)
    barTable = lineTable.T
    print(barTable)
    sns.set_theme(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))
    sns.barplot(data=lineTable,  palette="Blues_d")
    # ax = sns.lineplot(data=barTable)

    #
    # # Plot the total crashes
    # sns.set_color_codes("pastel")
    # sns.barplot(x="total", y="abbrev", data=barTable,
    #             label="Total", color="b")

    # # Plot the crashes where alcohol was involved
    # sns.set_color_codes("muted")
    # sns.barplot(x="alcohol", y="abbrev", data=crashes,
    #             label="Alcohol-involved", color="b")
    #
    # # Add a legend and informative axis label
    # ax.legend(ncol=2, loc="lower right", frameon=True)
    # ax.set(xlim=(0, 24), ylabel="",
    #        xlabel="Automobile collisions per billion miles")
    sns.despine(left=True, bottom=True)

    plt.show()

    plt.close()