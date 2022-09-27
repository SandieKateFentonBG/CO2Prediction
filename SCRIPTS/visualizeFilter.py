from Filter import *

def plotCorrelation(correlationMatrix, DBpath, displayParams, filteringName):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    mask = np.zeros_like(correlationMatrix)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(18,15))
    xticklabels = list(range(len(correlationMatrix)))
    if filteringName == 'checkup':
        xticklabels = "auto"
        sub = '- Check FEATURES - '
        cbar= True
        annot = True

    if filteringName == 'nofilter':
        title = '%s correlation coefficient heatmap' % method
        sub = ' - unfiltered features - '
        cbar= True
        annot = True
    if filteringName == 'dropuncorr' :
        title ='%s correlation coefficient heatmap' % method
        sub = ' - uncorrelated features removed - (CORR > %s)' % lowThreshhold
        cbar= True
        annot = True
    if filteringName == 'dropcolinear':
        title ='%s correlation coefficient heatmap' % method
        sub = ' - redundant features removed (CORR < %s) - ' % highThreshhold
        cbar = True
        annot = True
    plt.title(label = title + sub, fontsize = 18, loc='left', va='bottom' )
    # plt.suptitle(t = sub, fontsize = 14, horizontalalignment='left', verticalalignment = 'bottom')
    sns.heatmap(correlationMatrix, annot=annot, mask = mask, cbar = cbar, cbar_kws={"shrink": .80},
                xticklabels = xticklabels, fmt=".001f",ax=ax, cmap="bwr", center = 0, vmin=-1, vmax=1, square = True)

    # sns.set(font_scale=0.5)
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", 'correlation'+ filteringName
        import os
        outputFigPath = path + folder + subFolder # displayParams["outputPath"] + displayParams["reference"] + str(displayParams['random_state']) +'/correlation'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + filteringName + '.png')
    if displayParams['showCorr']:
        plt.show()
    plt.close()

    #todo : add archive
