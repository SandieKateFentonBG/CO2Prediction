import numpy as np
import matplotlib.pyplot as plt

def plotModelYellowResiduals(modelGridsearch, displayParams, DBpath, yLim = None, xLim = None, fontsize = None, studyFolder ='GS/'):

    df = modelGridsearch.learningDf
    if displayParams['showPlot'] or displayParams['archive']:

        from yellowbrick.regressor import ResidualsPlot

        xTrain, yTrain, xTest, yTest = df.XTrain.to_numpy(), df.yTrain.to_numpy().ravel(), df.XTest.to_numpy(), df.yTest.to_numpy().ravel()

        title = 'Residuals for ' + str(modelGridsearch.modelPredictor) + ' with ' + str(modelGridsearch.selectorName) \
                + '\n' + '- BEST PARAM (%s) ' % modelGridsearch.Param

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
        visualizer = ResidualsPlot(modelGridsearch.Grid, title = title, fig=fig, hist =True)#"frequency" qqplot = True
        visualizer.fit(xTrain, yTrain.ravel())  # Fit the training data to the visualizer
        visualizer.score(xTest, yTest.ravel())  # Evaluate the model on the test data

        bModelResTrR2 = round(visualizer.train_score_, modelGridsearch.rounding) #todo :remove this?
        bModelResTeR2 = round(visualizer.test_score_,  modelGridsearch.rounding) #todo :remove this?

        reference = displayParams['reference']

        if displayParams['archive']:

            path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'Residuals'
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            visualizer.show(outpath=outputFigPath + '/' + str(modelGridsearch.predictorName) + '_'
                        + str(modelGridsearch.selectorName) + '.png')

            # plt.savefig(outputFigPath + '/' + str(modelGridsearch.predictorName) + '_'
            #             + str(modelGridsearch.selectorName) + '.png')

        if displayParams['showPlot']:
            visualizer.show()

        #todo : this might be a problem
        plt.close()


def plotBlenderYellowResiduals(blendModel, displayParams, DBpath, NBestScore, NCount, yLim = None, xLim = None, fontsize = None,
                               studyFolder ='BLENDER/'):

    # df = blendModel.learningDf
    if displayParams['showPlot'] or displayParams['archive']:

        from yellowbrick.regressor import ResidualsPlot

        # xTrain, yTrain, xTest, yTest = df.XTrain.to_numpy(), df.yTrain.to_numpy().ravel(), df.XTest.to_numpy(), df.yTest.to_numpy().ravel()

        title = 'Residuals for ' + str(blendModel.GSName) + '- BEST PARAM (%s) ' % blendModel.Param

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
        visualizer = ResidualsPlot(blendModel.Estimator, title = title, fig=fig, hist =True)#"frequency" qqplot = True
        visualizer.fit(blendModel.blendXtrain, blendModel.yTrain.ravel())  # Fit the training data to the visualizer
        visualizer.score(blendModel.blendXtest, blendModel.yTest.ravel())  # Evaluate the model on the test data

        reference = displayParams['reference']

        if displayParams['archive']:

            path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'Residuals'
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            visualizer.show(outpath=outputFigPath + '/' + str(blendModel.GSName) + '_' + NBestScore + '_' + str(NCount) + '.png')

            # plt.savefig(outputFigPath + '/' + str(modelGridsearch.predictorName) + '_'
            #             + str(modelGridsearch.selectorName) + '.png')

        if displayParams['showPlot']:
            visualizer.show()


        plt.close()


def plotModelHistResiduals(modelGridsearch, displayParams, DBpath, bins=None, binrange = None, studyFolder ='GS/'):

    #todo : adapt bin count / bin range

    if displayParams['showPlot'] or displayParams['archive']:

        import seaborn as sns

        title = 'Residuals distribution for ' + str(modelGridsearch.modelPredictor) \
                + ' with ' + str(modelGridsearch.selectorName) \
                + '\n' + '- BEST PARAM (%s) ' % modelGridsearch.Param

        # print("modelGridsearch.bModelResid", modelGridsearch.Resid)
        fig, ax = plt.subplots(figsize=(10, 8))
        if bins and binrange: #ex : bins=20, binrange = (-200, 200)
            resmin, resmax = min(modelGridsearch.Resid), max(modelGridsearch.Resid)
            if resmax > binrange[1]:
                import math
                binrange[1] = math.ceil(resmax / 100) * 100
                print("residuals out of binrange  : (%s)" % max(modelGridsearch.Resid))
                print("bin max changed to :", binrange[1])
            if resmin < binrange[0]:
                import math
                binrange[0] = math.floor(resmin / 100) * 100
                print("residuals out of binrange  : (%s)" % min(modelGridsearch.Resid))
                print("bin min changed to :", binrange[0])
            ax = sns.histplot(modelGridsearch.Resid, kde=True, bins=bins, binrange = binrange, legend = False)
        else:
            ax = sns.histplot(modelGridsearch.Resid, kde=True, legend = False)

        plt.setp(ax.patches, linewidth=0)

        plt.title(title, fontsize=14)
        plt.xlabel("Residuals [%s]" % modelGridsearch.learningDf.yLabel, fontsize=14)
        reference = displayParams['reference']
        if displayParams['archive']:

            path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'Residuals'
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + str(modelGridsearch.predictorName) + '_'
                        + str(modelGridsearch.selectorName) + '-histplot.png')

        if displayParams['showPlot']:
            plt.show()
        # fig.tight_layout()
        plt.close()