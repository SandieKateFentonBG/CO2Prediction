import numpy as np
import matplotlib.pyplot as plt

def paramResiduals(modelGridsearch, df, displayParams, reference, DBpath, yLim = None , xLim = None, fontsize = None):

    if displayParams['showPlot'] or displayParams['archive']:

        from yellowbrick.regressor import ResidualsPlot

        xTrain, yTrain, xTest, yTest = df.XTrain.to_numpy(), df.yTrain.to_numpy().ravel(), df.XTest.to_numpy(), df.yTest.to_numpy().ravel()

        title = 'Residuals for ' + str(modelGridsearch.estimator) + '\n' + '- BEST PARAM (%s) ' % modelGridsearch.bModelParam

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
        visualizer = ResidualsPlot(modelGridsearch.paramGrid, title = title, fig=fig, hist =True)#"frequency" qqplot = True #todo : am i sure the fitting is identical?
        visualizer.fit(xTrain, yTrain.ravel())  # Fit the training data to the visualizer
        visualizer.score(xTest, yTest.ravel())  # Evaluate the model on the test data

        bModelResTrR2 = round(visualizer.train_score_, modelGridsearch.rounding)
        bModelResTeR2 = round(visualizer.test_score_,  modelGridsearch.rounding)

        # print("check this fitting returns identical results to the model gridsearch :")
        # print("Train score from visualizer / from gridsearch:", bModelResTrR2, modelGridsearch.bModelTrainScore)
        # print("Test score from visualizer / from gridsearch::", bModelResTeR2, modelGridsearch.bModelTestScore)

        if displayParams['archive']:

            path, folder, subFolder = DBpath, "RESULTS/", reference + 'Residuals'
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + str(modelGridsearch.name) + '.png')

        if displayParams['showPlot']:
            visualizer.show()

def plotResiduals(modelGridsearch, displayParams, reference, DBpath, processingParams, bins=None, binrange = None):

    #todo : adapt bin count / bin range

    if displayParams['showPlot'] or displayParams['archive']:

        import seaborn as sns

        title = 'Residuals distribution for ' + str(modelGridsearch.estimator) \
                + '\n' + '- BEST PARAM (%s) ' % modelGridsearch.bModelParam

        print("modelGridsearch.bModelResid", modelGridsearch.bModelResid)
        fig, ax = plt.subplots(figsize=(10, 8))
        if bins and binrange: #ex : bins=20, binrange = (-200, 200)
            resmin, resmax = min(modelGridsearch.bModelResid), max(modelGridsearch.bModelResid)
            if resmax > binrange[1]:
                import math
                binrange[1] = math.ceil(resmax / 100) * 100
                print("residuals out of binrange - max values should exceed : (%s)" % max(modelGridsearch.bModelResid))
                print("bin max changed to :", binrange[1])
            if resmin < binrange[0]:
                import math
                binrange[0] = math.floor(resmin / 100) * 100
                print("residuals out of binrange - min values should exceed : (%s)" % min(modelGridsearch.bModelResid))
                print("bin min changed to :", binrange[0])
            ax = sns.histplot(modelGridsearch.bModelResid, kde=True, bins=bins, binrange = binrange, legend = False)
        else:
            ax = sns.histplot(modelGridsearch.bModelResid, kde=True, legend = False)

        plt.setp(ax.patches, linewidth=0)

        plt.title(title, fontsize=14)
        plt.xlabel("Residuals [%s]" % processingParams['targetLabels'][0], fontsize=14)

        if displayParams['archive']:

            path, folder, subFolder = DBpath, "RESULTS/", reference + 'Residuals'
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + str(modelGridsearch.name) + '-histplot.png')

        if displayParams['showPlot']:
            plt.show()
        # fig.tight_layout()
        plt.close()