

def reportCV_Filter(CV_AllModels, CV_Filters_Spearman, CV_Filters_Pearson, seeds, displayParams, DBpath):

    import pandas as pd

    AllDfs = []

    for CV_Filters in [CV_Filters_Spearman, CV_Filters_Pearson]:
        #dataframe labels
        horizTitles = []
        horizLabels_selected = []
        horizLabels_redundant = []
        horizLabels_uncorrelated = []
        horizLabels_dropped = []

        #vertilabels
        vertiLabels = CV_AllModels[0][0].__getattribute__('NoSelector').selectedLabels

        #horizlabels
        for filter, seed in zip(CV_Filters, seeds):  # 10studies
            horizTitle = seed  # ex : 38
            horizLabel_selected = filter.selectedLabels
            horizLabel_redundant = filter.redundantLabels
            horizLabel_uncorrelated = filter.uncorrelatedLabels
            horizLabel_dropped = filter.droppedLabels

            horizTitles.append(horizTitle)  # ex : [38 38 38]
            horizLabels_selected.append(horizLabel_selected) #0
            horizLabels_redundant.append(horizLabel_redundant) #11
            horizLabels_uncorrelated.append(horizLabel_uncorrelated) #1
            horizLabels_dropped.append(horizLabel_dropped)

        # create empty dfs
        ScoresDf = pd.DataFrame(columns=horizTitles, index=vertiLabels)
        ScoresDf2 = pd.DataFrame(columns=horizTitles, index=vertiLabels)
        # fill in values
        for name, num, horizlabels in zip(["sel", "red", "unc"], [10, 8, 0], [horizLabels_selected, horizLabels_redundant, horizLabels_uncorrelated]):
            for i in range(len(horizlabels)): #col par col #ex i = 4 : [[Gifa, floors_bg, structure][Gifa, floors_bg, structure][Gifa, floors_bg, structure][Gifa, floors_bg, structure]]
                for j in range(len(horizlabels[i])): #ex : j = 3
                    ScoresDf.loc[[horizlabels[i][j]], [horizTitles[i]]] = name #ex : HAPDf.loc[[Gifa], [40]] = selected
                    ScoresDf2.loc[[horizlabels[i][j]], [horizTitles[i]]] = num

        slice = ScoresDf2.iloc[:, 0:len(horizTitles)]
        ScoresDf2.loc[:, 'Total'] = slice.sum(axis=1)
        ScoresDf2.loc[:, 'Mean'] = slice.mean(axis=1)
        ScoresDf.loc[:, 'Mean'] = ScoresDf2.loc[:, 'Mean']
        ScoresDf.loc[:, 'Total'] = ScoresDf2.loc[:, 'Total']

        AllDfs.append(ScoresDf)
        AllDfs.append(ScoresDf2)

    sortedDfs =[]
    for df in AllDfs:
        sortedDf = df.sort_values('Total', ascending=False)
        sortedDfs.append(sortedDf)
    AllDfs += sortedDfs

    sheetNames = ["Spearman", "Spearman-rank", "Pearson", "Pearson-rank",
                  "Spearman-sorted", "Spearman-rank-sorted", "Pearson-sorted", "Pearson-rank-sorted"]
    # export
    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference[:-6] + '_Combined/' + 'RECORDS/'
        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)
        with pd.ExcelWriter(outputPathStudy + reference[:-6] + "_CV_Filter" + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name, freeze_panes=(0, 1))
    for n in AllDfs:
        print(n)


