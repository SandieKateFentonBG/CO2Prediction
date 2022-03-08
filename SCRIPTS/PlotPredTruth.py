
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plotPredTruth(yTest, yPred, displayParams, modeldict, fontsize = 14):

    # plt.clf()
    # plt.cla()
    # plt.rcParams['figure.figsize'] = [18, 18]
    plt.figure(figsize=(10, 8))
    plt.grid(False)
    l1, = plt.plot(yTest, 'g')
    l2, = plt.plot(yPred, 'r', alpha=0.7)
    plt.legend(['Ground truth', 'Predicted'], fontsize=fontsize)
    title = str(modeldict['bModel']) + '- BEST PARAM (%s) ' % modeldict['bModelParam'] \
            + '- SCORE : ACC(%s) ' % modeldict['bModelAcc'] + 'MSE(%s) ' % modeldict['bModelMSE'] + 'R2(%s)' % modeldict['bModelr2']
    plt.title(title, fontdict = {'fontsize' : fontsize})
    plt.xticks(fontsize=fontsize+2)
    plt.xlabel('Test Building', fontsize=fontsize)
    plt.ylim(ymin=displayParams['TargetMinMaxVal'][0], ymax=displayParams['TargetMinMaxVal'][1])
    plt.yticks(fontsize=fontsize)
    plt.ylabel(displayParams['Target'], fontsize=fontsize)
    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + str(displayParams['random_state']) + '/Pred_Truth'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + str(modeldict['bModel']) + '.png')
    # fig.tight_layout()
    if displayParams['showPlot']:
        plt.show()
    plt.close()

def predTruthCombined(displayParams, models, x, y, Train = False):

    plt.clf()
    # plt.rcParams['figure.figsize'] = [18, 18]
    fig = plt.figure(figsize=(18,18))
    yPreds = [list(y.T[0])]
    labels = ['Groundtruth']
    for m in models:
        labels.append(m['bModel'])
        yPreds.append(m['bModel'].predict(x))

    df = pd.DataFrame(yPreds, index=labels)  # , columns=yTePreds

    sns.lineplot(data=df.T)
    if Train:
        title = 'Predicted values for various models compared to groundtruth on Training Set'
    else:
        title = 'Predicted values for various models compared to groundtruth on Testing Set'

    plt.title(label = title, fontdict = {'fontsize' : 20})
    plt.xlabel('Test Building', fontsize=18)

    plt.ylabel(displayParams['Target'], fontsize=18)
    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + str(displayParams['random_state']) + '/Pred_Truth'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + 'Combined.png')
    if displayParams['showPlot']:
        plt.show()
    plt.close()
    plt.clf()