import numpy as np
import matplotlib.pyplot as plt

def assembleResid(studies):
    residualsSplit = dict()
    residualsMerge = dict()

    for i in range(len(studies[0])):
        residualsSplit[str(studies[0][i]['model'])] = []
        residualsMerge[str(studies[0][i]['model'])] = []
    for h in range(len(studies)):
        for i in range(len(studies[h])):

            residualsSplit[str(studies[h][i]['model'])].append((studies[h][i]['bModelResid']).reshape(1, -1).tolist()[0])
    for k in residualsMerge.keys():
        residualsMerge[k] = mergeList(residualsSplit[k])

    return residualsSplit, residualsMerge

def mergeList(list):

    return [j for i in list for j in i]

def plotResiduals(m, displayParams, bestParam = None):
    import seaborn as sns
    modelWithParam = m['bModel']
    title = 'Residuals districbution for ' + str(modelWithParam)
    x= "Residuals [%s]" % displayParams['Target']
    if bestParam:
        title += '- BEST PARAM (%s) ' % bestParam

    fig, ax = plt.subplots()
    ax = sns.histplot(m['bModelResid'], kde=True, bins=14, binrange = (-100, 100), legend = False)
    plt.setp(ax.patches, linewidth=0)

    plt.title(title, fontsize=14)
    plt.xlabel("Residuals [%s]" % displayParams['Target'], fontsize=14)

    # sns.displot(m['bModelResid'], x = x, discrete = True, kde=True, kind="kde", bw_adjust=.25)
    # plt.figure(figsize=(10, 10))
    # sns.kdeplot(m['bModelResid'], color='orange')
    #sns.barplot(data=data, x='var1', color='#007b7f'),line_kws={"c":"black", "linewidth":2}, kde_kws={"c":"white", "linewidth":2}
    # , color = 'white'
    # , line_kws = {'lw': 3},
    # color = 'deepskyblue', facecolor = 'lime', edgecolor = 'black'
     #bins = 10, discrete = True
    # sns.distplot(m['bModelResid'], kde = True, norm_hist = True)  # you may select the no. of bins

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + str(displayParams['random_state']) +'/Residuals'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + str(modelWithParam) + '-histplot.png')
    if displayParams['showPlot']:
        plt.show()
    # fig.tight_layout()
    plt.close()

def plotAllResiduals(studies, displayParams):


    import seaborn as sns

    residualsSplit, residuals = assembleResid(studies)

    for k, v in residuals.items():
        title = 'Residuals districbution for ' + k
        x = "Residuals [%s]" % displayParams['Target']
        fig, ax = plt.subplots()
        ax = sns.histplot(v, kde=True, bins=14, binrange = (-100, 100), legend = False)
        plt.setp(ax.patches, linewidth=0)

        plt.title(title, fontsize=14)
        plt.xlabel("Residuals [%s]" % displayParams['Target'], fontsize=14)

        if displayParams['archive']:
            import os
            outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Residuals'
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)
            plt.savefig(outputFigPath + '/' + k + '-JointHistplot.png')
        if displayParams['showPlot']:
            plt.show()
        # fig.tight_layout()
        plt.close()

def plotResHistGauss(studies, displayParams, binwidth = 25, setxLim = (-150, 150), fontsize = 14):
    from scipy.stats import norm

    import seaborn as sns

    residualsSplit, residuals = assembleResid(studies)

    for k, v in residuals.items():
        title = 'Residuals distribution for ' + k
        x = "Residuals [%s]" % displayParams['Target']
        fig, ax = plt.subplots()
        ax = sns.histplot(v, kde=True, legend = False, binwidth=binwidth, label ="Residuals kde curve")
        plt.setp(ax.patches, linewidth=0)
        plt.title(title, fontsize=fontsize)
        plt.xlabel("Residuals [%s]" % displayParams['Target'], fontsize=fontsize)
        plt.ylabel("Count", fontsize=fontsize)
        arr = np.array(v)
        plt.figure(1)
        if setxLim:
            xLim = setxLim
        else :
            xLim = (min(arr), max(arr))
        plt.xlim(xLim)
        mean = np.mean(arr)
        variance = np.var(arr)
        sigma = np.sqrt(variance)
        x = np.linspace(min(arr), max(arr), 100)
        t = np.linspace(-300, 300, 100)
        dx = binwidth
        scale = len(arr) * dx
        # plt.plot(x, norm.pdf(x, mean, sigma) * scale, color='red', linestyle='dashed', label = "Gaussian curve")
        plt.plot(t, norm.pdf(t, mean, sigma) * scale, color='red', linestyle='dashed', label = "Gaussian curve")

        plt.legend()

        if displayParams['archive']:
            import os
            outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Residuals'
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)
            plt.savefig(outputFigPath + '/' + k + '-HistGaussPlot.png')
        if displayParams['showPlot']:
            plt.show()
        plt.close()

def plotNormResDistribution(studies, displayParams):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab
    from scipy.stats import norm
    import seaborn as sns

    residualsSplit, residuals = assembleResid(studies)

    for k, v in residuals.items():

        arr = np.array(v)

        plt.figure(1)
        plt.hist(arr, density=True)
        plt.xlim((min(arr), max(arr)))

        mean = np.mean(arr)
        variance = np.var(arr)
        sigma = np.sqrt(variance)
        x = np.linspace(min(arr), max(arr), 100)
        plt.plot(x, norm.pdf(x, mean, sigma))

        plt.show()

def plotScaleResDistribution(studies, displayParams):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab
    from scipy.stats import norm
    import seaborn as sns

    residualsSplit, residuals = assembleResid(studies)

    for k, v in residuals.items():

        arr = np.array(v)


        plt.figure(1)
        result = plt.hist(arr)
        plt.xlim((min(arr), max(arr)))

        mean = np.mean(arr)
        variance = np.var(arr)
        sigma = np.sqrt(variance)
        x = np.linspace(min(arr), max(arr), 100)
        dx = result[1][1] - result[1][0]
        scale = len(arr)*dx
        plt.plot(x, norm.pdf(x, mean, sigma)*scale)

        plt.show()



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
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + str(displayParams['random_state']) + '/Residuals'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        visualizer.show(outpath=outputFigPath + '/' + str(modelWithParam) + '.png')

    if displayParams['showPlot']:
        visualizer.show()

    visualizer.finalize()

    return resDict