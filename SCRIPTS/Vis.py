def Weights(models, displayParams, sorted = True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    import seaborn as sns

    import numpy
    linModels = [m for m in models if m['Linear']==True] #only works/makes sense for linear models
    weights, labels = listWeight(linModels)

    table = pd.DataFrame(weights, columns=["Coefficients"], index=labels)
    print(table)
    sns.set_theme(style="whitegrid")
    tips = sns.load_dataset("tips")
    ax = sns.barplot(data=table, x="Feature", y="Weight", hue="Weight")


    xList = self.xQuanti[xLabel]
    yList = self.y[yLabel]
    df = pd.DataFrame(list(zip(xList, yList)), columns=[xLabel, yLabel])
    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_title(title + " " + reference)
    if xLabel in self.xQuali.keys():
        x = np.arange(len(labels))
        ax.set_ylabel(yLabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
                 rotation_mode="anchor")
    sns.scatterplot(data=df, x=xLabel, y=yLabel, hue=yLabel, ax=ax)

    # if displayParams['archive']:
    #     import os
    #     outputFigPath = displayParams["outputPath"] + '/' + folder
    #
    #     if not os.path.isdir(outputFigPath):
    #         os.makedirs(outputFigPath)
    #
    #     plt.savefig(outputFigPath + '/' + xLabel + '-' + yLabel + '.png')
    #
    # if displayParams['showPlot']:
    #     plt.show()

    plt.close()  # todo : check this