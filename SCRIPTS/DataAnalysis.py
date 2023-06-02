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
# ['A123-C34 Rate (kgCO2e/m2)', 'A1-A3 Rate (kgCO2e/m2)',	'A1-A5 Rate (kgCO2e/m2)',	'A1-C4 Rate (kgCO2e/m2)',	'A1-D Rate (kgCO2e/m2)']
yLabels = ['Carbon A1-A3 (kgCO2e)', 'A1-A3 Rate (kgCO2e/m2)', 'Carbon C3-C4 (kgCO2e)']
xQualLabels = ['Calculation Design Stage','Location','Value Type','Project Sector', 'Type', 'Passivhaus', 'Basement',
               'Foundation Type', 'Ground Floor Type', 'Superstructure Type', 'Cladding Type', 'Fire Rating']#
xQuantLabels = ['GIFA (m2)', 'Calculation Year', 'Project Value (poundm)','Storeys (#)']
RemoveOutliersFrom = xQuantLabels + yLabels

#CHANGES   !! LABELS MUST BE IN INITIAL IMPORT!
Summed_Labels = {'Carbon A123C34 (kgCO2e)' : ['Carbon A1-A3 (kgCO2e)', 'Carbon C3-C4 (kgCO2e)']} #SUMMED
Divided_Labels = {'A123C34 Rate (kgCO2e/m2)' : ['Carbon A123C34 (kgCO2e)', 'GIFA (m2)']} #SUMMED LABELS MUST BE IN INITIAL IMPORT!
# AddedLabels = [k for k in Summed_Labels.keys()] + [k for k in Divided_Labels.keys()]
splittingFt = 'Superstructure Type'
order = ['Concrete (In-Situ)', 'Concrete (Precast)','Concrete (PT)','Timber Frame (Glulam/CLT)',
         'Timber Frame (Softwood)','Steel Frame/Precast', 'Steel Frame/Composite','Steel Frame/Timber',
         'Steel Frame/Other', 'Masonry/Concrete','Masonry/Timber', 'Masonry & Timber','Other']
mainTarget = 'A123C34 Rate (kgCO2e/m2)'
labels_1D = ['Carbon A1-A3 (kgCO2e)', 'A1-A3 Rate (kgCO2e/m2)', 'Carbon A123C34 (kgCO2e)', 'A123C34 Rate (kgCO2e/m2)']
labels_2D_norm = [['Carbon A1-A3 (kgCO2e)', 'A1-A3 Rate (kgCO2e/m2)', 'Carbon A1-A3 (kgCO2e)_normalize', 'A1-A3 Rate (kgCO2e/m2)_normalize'],
                  ['Carbon A123C34 (kgCO2e)', 'A123C34 Rate (kgCO2e/m2)', 'Carbon A123C34 (kgCO2e)_normalize', 'A123C34 Rate (kgCO2e/m2)_normalize']]
labels_2D_scale = [['Carbon A1-A3 (kgCO2e)', 'A1-A3 Rate (kgCO2e/m2)', 'Carbon A1-A3 (kgCO2e)_scale', 'A1-A3 Rate (kgCO2e/m2)_scale'],
    ['Carbon A123C34 (kgCO2e)', 'A123C34 Rate (kgCO2e/m2)', 'Carbon A123C34 (kgCO2e)_scale','A123C34 Rate (kgCO2e/m2)_scale']]
exploded_ft = 'Calculation Year' #qual feature with few different values
splittingFt_focus = 'Concrete (In-Situ)' #order[0]
splittingFt_2 = 'Cladding Type'
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

        self.workingDf = removeOutliers(self.rawDf, labels=RemoveOutliersFrom,
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
        for df in singleDf_List:
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
            for df_list in subsetDf_list_unmerged: #df_list ex 5 qualities
                DA_list = []
                df_list_merged = pd.concat(df_list, axis=0)

                for df,qual in zip(df_list, qualities):
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
            outputPathStudy = path + "RESULTS/" + reference + 'RECORDS/' + 'DATA' + '/'
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
        path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'VISU/DATA'
        import os

        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/' + 'DA_boxPlot' + name + '.png')

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
        path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'VISU/DATA'
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
        path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'VISU/DATA'
        import os

        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/' + 'DA_boxPlot' + name + '.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()

def import_DataAnalysis(ref_prefix, name):

    path = DB_Values['DBpath'] + 'RESULTS/' + ref_prefix + '_Combined/' + 'RECORDS/DATA/' + name + '.pkl'
    DA = pickleLoadMe(path=path, show=False)

    return DA

def run_DataAnalysis(path, dbName, delimiter, firstLine, xQualLabels, xQuantLabels, yLabels,
                     Summed_Labels, Divided_Labels, splittingFt, order, mainTarget,
                     labels_1D, labels_2D_norm, labels_2D_scale, exploded_ft, splittingFt_focus, splittingFt_2):

    # RUN
    DA = DataAnalysis(path, dbName, delimiter, firstLine, xQualLabels, xQuantLabels, yLabels,
                     Summed_Labels, Divided_Labels)
    DA.studyDatabase(path, splittingFt, labels= ['rawDf', 'workingDf', 'scaleDf', 'normalizeDf'], orderFt = order)
    pickleDumpMe(path, displayParams, DA, 'DATA', 'DataAnalysis' + splittingFt, combined=True)

    # IMPORT
    DA = import_DataAnalysis(displayParams["ref_prefix"], name = 'DataAnalysis' + splittingFt)

    # SINGLE DF EXPORT TO EXCEL
    dfAsTable(DB_Values['DBpath'], displayParams, DA.sortingDfFull, objFolder='DATA', name = "DAi.sortingDfFull", combined = True)


    #plot normal
    DataAnalysis_boxPlot(DB_Values['DBpath'], displayParams["ref_prefix"],
                          data=DA.workingDf, x=mainTarget, y=splittingFt)
    #plot exploded feature
    DataAnalysis_boxPlot(DB_Values['DBpath'], displayParams["ref_prefix"],
                          data=DA.workingDfsorted, x=mainTarget, y=splittingFt, legend = exploded_ft, name = '_explo') #DAi.__getattribute__('workingDfsorted')
    #plot single feature
    DataAnalysis_boxPlot(DB_Values['DBpath'], displayParams["ref_prefix"],
                          data=DA.splittingFt[splittingFt_focus], x=mainTarget, y=splittingFt_2, name = splittingFt_focus) #DAi.__getattribute__(splittingFt)
    #plot mutliple 1D
    DataAnalysis_boxPlot_Multi_1D(DB_Values['DBpath'], displayParams["ref_prefix"], data=DA.workingDfsorted,
                               labels = labels_1D, y =splittingFt, legend = None, name ='Multi_1D')
    #plot mutliple 2D
    DataAnalysis_boxPlot_Multi_2D(DB_Values['DBpath'], displayParams["ref_prefix"], data=DA.sortingDfFull,
                               labels = labels_2D_norm, y =splittingFt, legend = None, name ='Multi_2D_norm')
    #plot mutliple 2D
    DataAnalysis_boxPlot_Multi_2D(DB_Values['DBpath'], displayParams["ref_prefix"], data=DA.sortingDfFull,
                               labels = labels_2D_scale, y =splittingFt, legend = None, name ='Multi_2D_scale')



run_DataAnalysis(path, dbName, delimiter, firstLine, xQualLabels, xQuantLabels, yLabels,
                     Summed_Labels, Divided_Labels, splittingFt, order, mainTarget,
                     labels_1D, labels_2D_norm, labels_2D_scale, exploded_ft, splittingFt_focus, splittingFt_2)