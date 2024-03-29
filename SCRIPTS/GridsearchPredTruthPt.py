import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def GS_predTruthCombined(displayParams, GS_FSs, DBpath, content = '', scatter=False, fontsize=14,
                      studyFolder='GS_FS/'):

    if displayParams['showPlot'] or displayParams['archive']:

        learningDflabel = GS_FSs[0].learningDfsList[0] #spearman_fl

        GS = GS_FSs[0].__getattribute__(learningDflabel) #GS_FS.spearman_fl

        yTest = GS.learningDf.yTest.to_numpy().ravel()
        yLabel = GS.learningDf.yLabel

        plt.clf()

        fig = plt.figure(figsize=(18, 18))
        test = list(yTest.T)

        yPreds = [test]
        labels = ['Groundtruth']
        y = []
        label = []
        groundtruthDf = pd.DataFrame(yPreds, index=labels)

        for GS_FS in GS_FSs:  # ,LR_LASSO_FS_GS, LR_RIDGE_FS_GS, LR_ELAST_FS_GS
            for learningDflabel in GS_FS.learningDfsList:
                GS = GS_FS.__getattribute__(learningDflabel)
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

            sns.scatterplot(data=groundtruthDf.T, marker="$\circ$", ec="face", s=100,
                            facecolors="none")  # markers='o'
            sns.scatterplot(data=groundtruthDf.T, marker="$\circ$", ec="face", s=40, facecolors="none",
                            palette=['white'], legend=None)  # markers='o'

        else:
            sns.lineplot(data=combinedDf.T)

        title = 'Predicted values vs Groundtruth'

        plt.title(label=title, fontdict={'fontsize': fontsize})
        plt.xlabel('Test Building', fontsize=fontsize)

        plt.ylabel(yLabel, fontsize=fontsize)  # change displayParams['Target'] for : df.yLabels[0]

        reference = displayParams['reference']
        if displayParams['archive']:
            path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'Pred_Truth'
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + 'Combined_' + content +'.png')
        if displayParams['showPlot']:
            plt.show()
        plt.close()
        plt.clf()
