

def split_list(main_list):
    n = len(main_list[0])
    sub_lists = [[] for _ in range(n)]

    for sublist in main_list:
        for i, item in enumerate(sublist):
            sub_lists[i].append(item)

    return sub_lists


def reportCV_Filter(CV_AllModels, filterList, seeds, displayParams, studyParams, DBpath):

    import pandas as pd

    AllDfs = []

    sub_lists = split_list(filterList)

    for CV_Filters in sub_lists: # 2 filters
        #dataframe labels
        horizTitles = []
        horizLabels_selected = []
        horizLabels_redundant = []
        horizLabels_uncorrelated = []
        horizLabels_dropped = []

        #vertilabels
        vertiLabels = CV_AllModels[0][0].__getattribute__('NoSelector').selectedDict

        #horizlabels
        for filter, seed in zip(CV_Filters, seeds):  # 10studies
            horizTitle = seed  # ex : 38
            horizLabel_selected = filter.selectedDict
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

    sheetNames = []
    for name in studyParams['fl_selectors']:
        sheetNames += [name, name + '-rank']
    for name in studyParams['fl_selectors']:
        sheetNames += [name + '-sorted', name + '-rank-sorted']

    # export
    if displayParams['archive']:
        import os
        reference = displayParams['ref_prefix'] + '_Combined/'
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'
        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)
        with pd.ExcelWriter(outputPathStudy + displayParams['ref_prefix'] + "_CV_Filter" + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name, freeze_panes=(0, 1))



