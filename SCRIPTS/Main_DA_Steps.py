#DASHBOARD IMPORT
from Dashboard_Current import *

#SCRIPT IMPORTS
from Main_GS_FS_Steps import *
from Main_Combine_Steps import *


 #INPUT
path=DB_Values['DBpath']
dbName=DB_Values['DBname']
delimiter=DB_Values['DBdelimiter']
firstLine=DB_Values['DBfirstLine']
displayParams["ref_prefix"] = DB_Values['acronym'] + '_' + studyParams['sets'][0][1]



class DataAnalysis:
    def __init__(self, path, dbName, delimiter, firstLine, xQualLabels, xQuantLabels, yLabels,
                 Summed_Labels, Divided_Labels):

        """
        # >> 1 BASE DATAFRAME
        # >> 2 NO OUTLIER DATAFRAME
        # >> 3 SCALE DATAFRAME
        # >> 4 SUBSET DATAFRAME
        # >> 5 ANALYZE DATAFRAME
        """
        # >> 1 BASE DATAFRAME

        #open file
        header, reader = open_csv_at_given_line(path, dbName, delimiter, firstLine)

        #as dict
        self.xQuali = {k: [] for k in xQualLabels}
        self.xQuanti = {k: [] for k in xQuantLabels}
        self.y = {k: [] for k in yLabels}
        self.AddedLabels = [k for k in Summed_Labels.keys()] + [k for k in Divided_Labels.keys()]
        for line in reader:
            for (labels, attribute) in [(xQuantLabels, self.xQuanti), (yLabels, self.y)]:
                for label in labels:
                    attribute[label].append(float(line[header.index(label)].replace(',', '.')))
            for label in xQualLabels:
                self.xQuali[label].append(line[header.index(label)])

        keys = [k for k in self.xQuanti.keys()] + [l for l in self.xQuali.keys()] + [m for m in self.y.keys()]
        vals = [k for k in self.xQuanti.values()] + [l for l in self.xQuali.values()] + [m for m in self.y.values()]

        # Feature analysis
        self.possibleQualities = dict()
        for label, column in self.xQuali.items():
            self.possibleQualities[label] = []
            for value in column:
                if value not in self.possibleQualities[label]:
                    self.possibleQualities[label].append(value)
        self.rawDfsorted = None
        self.workingDfsorted = None
        self.scaleDfsorted = None
        self.normalizeDfsorted = None
        self.splittingFt = None

        #as dataframe
        self.rawDf = pd.DataFrame(columns=keys) #, index=list(range(vals[0]))

        for k, v in zip(keys, vals):
            self.rawDf.loc[:, k] = v

        #ADD
        for k,v in Summed_Labels.items():
            self.rawDf.loc[:, k] = sum([self.rawDf[v[i]] for i in range(len(v))])
        for k,v in Divided_Labels.items():
            self.rawDf.loc[:, k] = self.rawDf[v[0]] / self.rawDf[v[1]]

        # >> 2 NO OUTLIER DATAFRAME

        workingDf, self.removedDict = removeUnderrepresenteds(self.rawDf, labels=PROCESS_VALUES['removeUnderrepresentedsFrom'],
                                             cutOffThreshhold=PROCESS_VALUES['UnderrepresentedCutOffThreshhold'])
        self.workingDf = removeOutliers(workingDf, labels=PROCESS_VALUES['RemoveOutliersFrom'] + DAyLabels,
                                   cutOffThreshhold=PROCESS_VALUES['OutlierCutOffThreshhold'])

        # >> 3 SCALE DATAFRAME

        #SCALE
        self.scaleDf = self.workingDf.copy()
        self.normalizeDf = self.workingDf.copy()
        for k in xQuantLabels + yLabels + self.AddedLabels:
            colMean, colStd = self.workingDf[k].mean(axis=0), self.workingDf[k].std(axis=0)
            colMax, colMin = self.workingDf[k].max(axis=0), self.workingDf[k].min(axis=0)
            self.scaleDf[k] = (self.workingDf[k] - colMean) / colStd
            self.normalizeDf[k] = (self.workingDf[k] - colMin) / (colMax - colMin)

        self.workingDfFull = None # workingDf + scaleDf + normalizeDf
        self.sortingDfFull = None # workingDfsorted + scaleDfsorted + normalizeDfsorted

        # store into big df for plotters
        new_scaleDf = self.scaleDf.copy().add_suffix("_scale")
        new_normalizeDf = self.normalizeDf.copy().add_suffix("_normalize")
        self.workingDfFull = pd.concat([self.workingDf, new_scaleDf, new_normalizeDf], axis=1)

    def DataAnalysis_Scatterplots(self, DBpath, ref_prefix, dataname, ylabel):

        for xLabel in [l for l in self.xQuali.keys()] + [l for l in self.xQuanti.keys()]:
            self.DataAnalysis_Scatterplot(DBpath, ref_prefix, dataname, ylabel, xLabel)

    def DataAnalysis_Scatterplot(self, DBpath, ref_prefix, dataname, yLabel, xLabel):

        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np


        fontsize = 10
        removed = ''
        if xLabel in self.removedDict.keys():
            removed = '\n Removed :' + str(self.removedDict[xLabel])
        data = self.__getattribute__(dataname)
        fig, ax = plt.subplots(figsize=(8, 8))
        # ax.set_title(title)

        if xLabel in self.xQuali.keys():
            labels = self.possibleQualities[xLabel]
            x = np.arange(len(labels))
            ax.set_ylabel(yLabel, fontsize = fontsize)
            ax.set_xlabel(xLabel + removed, fontsize = fontsize)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize = fontsize)
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
                     rotation_mode="anchor")

        sns.scatterplot(data=data, x=xLabel, y=yLabel, hue=yLabel, ax=ax)

        # fig.tight_layout(pad=1.08)

        if displayParams['archive']:
            path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'VISU/ANALYSIS/' + dataname
            import os

            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)
            plt.savefig(outputFigPath + '/' + xLabel + '-' + yLabel + '.png')

        if displayParams['showPlot']:
            plt.show()
        plt.close()

    def createSubsets(self, splittingFt, df, order = None):

        # >> 4 SUBSET DATAFRAME
        subsets = []
        feature_dict = dict()
        for quality in self.possibleQualities[splittingFt]:
            newsubset = df.loc[df[splittingFt] == quality]
            subsets.append(newsubset)
            feature_dict[quality] = newsubset
        self.splittingFt = feature_dict
        self.__setattr__(splittingFt, feature_dict)

        return subsets

    def analyzeDataframe(self, df, index = ['']):

        # >> 5 ANALYZE DATAFRAME
        # ANALYZE
        if df.empty:
            cols = df.columns.values.tolist()
            index += ['Mean','Stdv','Min','Max','Mode']
            statDf = pd.DataFrame(columns=cols, index=index)  # done to leave empty space

        else :
            cols = df.columns.values.tolist()
            statDf = pd.DataFrame(columns=cols, index = index) #done to leave empty space
            statDf.loc['Mean', :] = df.mean(axis=0)
            statDf.loc['Stdv', :] = df.std(axis=0)
            statDf.loc['Min', :] = df.min(axis=0)
            statDf.loc['Max', :] = df.max(axis=0)
            mode_df = pd.DataFrame([df.mode().iloc[0]], columns=cols)
            statDf.loc['Mode'] = mode_df.loc[0]

        return statDf

    def studyDatabase(self, path, splittingFt, labels= ['rawDf', 'workingDf', 'scaleDf', 'normalizeDf'], orderFt = None):

        # ANALYZE
        singleDf_List = [self.__getattribute__(l) for l in labels]
        singleDA_List = []

        for df, l in zip(singleDf_List,labels) :
            dfDA = self.analyzeDataframe(df)
            singleDA_List.append(dfDA)
        # STORE
        all_labels = []
        all_tables = []
        name = 'DataAnalysis'
        for df, DA, lab in zip(singleDf_List, singleDA_List, labels):
            all_tables.append(df)
            all_tables.append(DA)
            all_labels.append(lab)
            all_labels.append(lab + 'DA')

        if splittingFt:
            # ANALYZE
            if orderFt:
                self.possibleQualities[splittingFt] = orderFt
            qualities = self.possibleQualities[splittingFt]
            subsetDf_list_unmerged = [self.createSubsets(splittingFt, self.__getattribute__(l)) for l in labels]
            subsetDf_list = []
            subsetDA_list = []
            for df_list, la in zip(subsetDf_list_unmerged, labels): #df_list ex 5 qualities
                DA_list = []
                df_list_merged = pd.concat(df_list, axis=0)
                for df, qual in zip(df_list, qualities):
                    if df.empty:
                        print(la, qual, 'DataFrame is empty!')

                    dfDA = self.analyzeDataframe(df, index=[qual])
                    DA_list.append(dfDA)
                DA_list_merged = pd.concat(DA_list, axis=0)
                subsetDf_list.append(df_list_merged)
                subsetDA_list.append(DA_list_merged)
            # STORE
            name += splittingFt
            for df, DA, lab in zip(subsetDf_list, subsetDA_list, labels):
                self.__setattr__(lab + 'sorted', df)
                all_tables.append(df)
                all_tables.append(DA)
                all_labels.append(lab + splittingFt)
                all_labels.append(lab + 'DA' + splittingFt)

        # store into big df for plotters
        new_scaleDf = self.scaleDfsorted.copy().add_suffix("_scale")
        new_normalizeDf = self.normalizeDfsorted.copy().add_suffix("_normalize")
        self.sortingDfFull = pd.concat([self.workingDfsorted,new_scaleDf, new_normalizeDf], axis=1) #todo

        if displayParams['archive']:

            import os

            reference = displayParams['ref_prefix'] + '_Combined/'
            outputPathStudy = path + "RESULTS/" + reference + 'RECORDS/' + 'ANALYSIS' + '/'
            if not os.path.isdir(outputPathStudy):
                os.makedirs(outputPathStudy)

            with pd.ExcelWriter(outputPathStudy + name + ".xlsx", mode='w') as writer:

                for df, name in zip(all_tables, all_labels):
                    df.to_excel(writer, sheet_name=name)

def DataAnalysis_boxPlot (DBpath, ref_prefix, data, x, y, legend = None, name ='', order = None) :

    fig = plt.figure() #figsize=(20, 5)
    dodge = True
    if not legend:
        hue = y
        dodge = False
    else:
        hue = legend
    ax = sns.boxplot(data=data, x=x, y=y, hue = hue, order = order, dodge=dodge, palette = "vlag") # hue=y,
    if not legend:
        plt.legend([], [], frameon=False)
    fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'VISU/ANALYSIS'
        import os

        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/' + 'DA_boxPlot' + name + '.png')
        print(outputFigPath + '/' + 'DA_boxPlot' + name + '.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()

def DataAnalysis_boxPlot_Multi_1D(DBpath, ref_prefix, data, labels, y, legend = None, name ='', order = None):
    l = len(labels)
    fig, axes = plt.subplots(1, l, figsize=(18, 5),sharey=True) #,  sharex=True,
    for i in range(len(labels)):
        dodge = True
        if not legend:
            hue = y
            dodge = False
        else:
            hue = legend
        ax = sns.boxplot(ax=axes[i], data=data, x=labels[i], y=y, hue=hue, order=order, dodge=dodge, palette = "vlag") # hue=y,, 0

    for ax in axes:
        ax.legend([], [], frameon=False)

    fig.tight_layout(pad=1.08)

    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'VISU/ANALYSIS'
        import os

        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/' + 'DA_boxPlot' + name + '.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()

def DataAnalysis_boxPlot_Multi_2D(DBpath, ref_prefix, data, labels, y, legend = None, name ='', order = None):
    cols = len(labels[0])
    rows = len(labels)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10),sharey=True) #,  sharex=True,
    for i in range(rows):
        for j in range(cols):
            dodge = True
            if not legend:
                hue = y
                dodge = False
            else:
                hue = legend
            ax = sns.boxplot(ax=axes[i, j], data=data, x=labels[i][j], y=y, hue=hue, order=order, dodge=dodge, palette = "vlag") # hue=y,, 0

    for axs in axes:
        for ax in axs:
            ax.legend([], [], frameon=False)

    fig.tight_layout(pad=1.08)

    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'VISU/ANALYSIS'
        import os

        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/' + 'DA_boxPlot' + name + '.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()

def import_DataAnalysis(ref_prefix, name):

    path = DB_Values['DBpath'] + 'RESULTS/' + ref_prefix + '_Combined/' + 'RECORDS/ANALYSIS/' + name + '.pkl'
    DA = pickleLoadMe(path=path, show=False)

    return DA

def Run_DA(path, dbName, delimiter, firstLine, xQualLabels, xQuantLabels, yLabels,
           Summed_Labels, Divided_Labels, splittingFt, order, mainTarget,
           labels_1D, labels_2D_norm, labels_2D_scale, exploded_ft, splittingFt_focus, splittingFt_2):

    # RUN
    DA = DataAnalysis(path, dbName, delimiter, firstLine, xQualLabels, xQuantLabels, yLabels,
                     Summed_Labels, Divided_Labels)
    dfAsTable(DB_Values['DBpath'], displayParams, DA.workingDf, objFolder='ANALYSIS', name = "DA.workingDf", combined = True)
    dfAsTable(DB_Values['DBpath'], displayParams, DA.rawDf, objFolder='ANALYSIS', name = "DA.rawDf", combined = True)

    DA.DataAnalysis_Scatterplots(DB_Values['DBpath'], displayParams["ref_prefix"], dataname='workingDf', ylabel=mainTarget)
    dfAsTable(DB_Values['DBpath'], displayParams, DA.workingDf, objFolder='ANALYSIS', name = "mycheck", combined = True)

    DA.studyDatabase(path, splittingFt = splittingFt, labels= ['rawDf', 'workingDf', 'scaleDf', 'normalizeDf'],
                     orderFt = order)
    pickleDumpMe(path, displayParams, DA, 'ANALYSIS', 'DataAnalysis' + splittingFt, combined=True)

    # # IMPORT
    # DA = import_DataAnalysis(displayParams["ref_prefix"], name = 'DataAnalysis' + splittingFt)

    # SINGLE DF EXPORT TO EXCEL
    dfAsTable(DB_Values['DBpath'], displayParams, DA.sortingDfFull, objFolder='ANALYSIS', name = "DAi.sortingDfFull", combined = True)


    # #plot normal
    DataAnalysis_boxPlot(DB_Values['DBpath'], displayParams["ref_prefix"],
                          data=DA.workingDf, x=mainTarget, y=splittingFt, name = focus)
    #plot exploded feature
    DataAnalysis_boxPlot(DB_Values['DBpath'], displayParams["ref_prefix"],
                          data=DA.workingDfsorted, x=mainTarget, y=splittingFt, legend = exploded_ft, name = '_explo' + focus) #DAi.__getattribute__('workingDfsorted')
    #plot single feature
    DataAnalysis_boxPlot(DB_Values['DBpath'], displayParams["ref_prefix"],
                          data=DA.splittingFt[splittingFt_focus], x=mainTarget, y=splittingFt_2, name = focus) #DAi.__getattribute__(splittingFt)
    # plot mutliple 1D
    DataAnalysis_boxPlot_Multi_1D(DB_Values['DBpath'], displayParams["ref_prefix"], data=DA.sortingDfFull,
                               labels = labels_1D, y =splittingFt, legend = None, name ='Multi_1D'+ focus)
    #plot mutliple 2D
    DataAnalysis_boxPlot_Multi_2D(DB_Values['DBpath'], displayParams["ref_prefix"], data=DA.sortingDfFull,
                               labels = labels_2D_norm, y =splittingFt, legend = None, name ='Multi_2D_norm'+ focus)
    #plot mutliple 2D
    DataAnalysis_boxPlot_Multi_2D(DB_Values['DBpath'], displayParams["ref_prefix"], data=DA.sortingDfFull,
                               labels = labels_2D_scale, y =splittingFt, legend = None, name ='Multi_2D_scale'+ focus)



# run_DataAnalysis(path, dbName, delimiter, firstLine, xQualLabels, xQuantLabels, DAyLabels,
#                  Summed_Labels, Divided_Labels, splittingFt = splittingFt, order = order, mainTarget = mainTarget,
#                  labels_1D = labels_1D, labels_2D_norm = labels_2D_norm, labels_2D_scale = labels_2D_scale,
#                  exploded_ft = exploded_ft, splittingFt_focus = splittingFt_focus, splittingFt_2 = splittingFt_2)