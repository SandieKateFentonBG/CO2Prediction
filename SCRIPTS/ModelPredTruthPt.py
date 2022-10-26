import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plotPredTruth(displayParams, modelGridsearch, DBpath, TargetMinMaxVal, fontsize = 14, studyFolder = 'GS/' ):

    if displayParams['showPlot'] or displayParams['archive']:

        plt.figure(figsize=(10, 8))
        plt.grid(False)

        yTest = modelGridsearch.learningDf.yTest.to_numpy().ravel()
        yPred = modelGridsearch.yPred

        l1, = plt.plot(yTest, 'g')
        l2, = plt.plot(yPred, 'r', alpha=0.7)
        plt.legend(['Ground truth', 'Predicted'], fontsize=fontsize)
        title = str(modelGridsearch.modelPredictor) + ' with ' + str(modelGridsearch.selectorName) + '\n' + '- BEST PARAM (%s) ' % modelGridsearch.Param \
                + '\n' + '- SCORE : ACC(%s) ' % modelGridsearch.TestAcc + 'MSE(%s) ' % modelGridsearch.TestMSE + \
                'R2(%s)' % modelGridsearch.TestR2

        plt.title(title, fontdict = {'fontsize' : fontsize})
        plt.xticks(fontsize=fontsize+2)
        plt.xlabel('Test Building', fontsize=fontsize)
        plt.ylim(ymin=TargetMinMaxVal[0], ymax=TargetMinMaxVal[1])
        plt.yticks(fontsize=fontsize)
        plt.ylabel(modelGridsearch.learningDf.yLabel, fontsize=fontsize)

        reference = displayParams['reference']
        if displayParams['archive']:
            path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'Pred_Truth'
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + str(modelGridsearch.predictorName) + '_' + str(modelGridsearch.selectorName) + '.png')
        if displayParams['showPlot']:
            plt.show()
        plt.close()

def predTruthCombined(displayParams, GSs, DBpath, scatter=False, fontsize=14, studyFolder = 'GS/' ):

    if displayParams['showPlot'] or displayParams['archive']:
        yTest = GSs[0].learningDf.yTest.to_numpy().ravel()
        yLabel = GSs[0].learningDf.yLabel

        plt.clf()

        fig = plt.figure(figsize=(18,18))
        test = list(yTest.T)

        yPreds = [test]
        labels = ['Groundtruth']
        y = []
        label = []
        groundtruthDf = pd.DataFrame(yPreds, index=labels)

        for GS in GSs:
            lab = str(GS.modelPredictor) + '-' + GS.selectorName
            labels.append(lab)
            yPreds.append(GS.yPred)
            label.append(lab)
            y.append(GS.yPred)

        predDf = pd.DataFrame(y, index=label)
        combinedDf = pd.DataFrame(yPreds, index=labels)

        if scatter:
            sns.lineplot(data=predDf.T)
            # sns.scatterplot(data=groundtruthDf.T, marker="$\circ$", ec="blue", fc='none', s=100, facecolors="none" )#markers='o'

            sns.scatterplot(data=groundtruthDf.T, marker="$\circ$", ec="face", s=100, facecolors="none" )#markers='o'
            sns.scatterplot(data=groundtruthDf.T, marker="$\circ$", ec="face", s=40, facecolors="none", palette = ['white'], legend=None )#markers='o'

        else:
            sns.lineplot(data=combinedDf.T)

        title = 'Predicted values vs Groundtruth'

        plt.title(label = title, fontdict = {'fontsize' : fontsize})
        plt.xlabel('Test Building', fontsize=fontsize)

        plt.ylabel(yLabel, fontsize=fontsize) #change displayParams['Target'] for : df.yLabels[0]

        reference = displayParams['reference']
        if displayParams['archive']:
            path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'Pred_Truth'
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + 'Combined.png')
        if displayParams['showPlot']:
            plt.show()
        plt.close()
        plt.clf()



