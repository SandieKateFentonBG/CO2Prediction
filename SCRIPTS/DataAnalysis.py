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
AddedLabels = [k for k in Summed_Labels.keys()] + [k for k in Divided_Labels.keys()]
splittingFt = 'Superstructure Type'

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

        self.possibleQualities = dict()
        for label, column in self.xQuali.items():
            self.possibleQualities[label] = []
            for value in column:
                if value not in self.possibleQualities[label]:
                    self.possibleQualities[label].append(value)

        #as dataframe
        self.rawDf = pd.DataFrame(columns=keys) #, index=list(range(vals[0]))

        for k, v in zip(keys, vals):
            self.rawDf.loc[:, k] = v

        #ADD
        for k,v in Summed_Labels.items():
            print(k, v)
            self.rawDf.loc[:, k] = sum([self.rawDf[v[i]] for i in range(len(v))])
        for k,v in Divided_Labels.items():
            print(k, v)
            self.rawDf.loc[:, k] = self.rawDf[v[0]] / self.rawDf[v[1]]

        # >> 2 NO OUTLIER DATAFRAME

        self.workingDf = removeOutliers(self.rawDf, labels=RemoveOutliersFrom,
                                        cutOffThreshhold=PROCESS_VALUES['OutlierCutOffThreshhold'])
        # >> 3 SCALE DATAFRAME

        #SCALE
        self.scaleDf = self.workingDf.copy()
        self.normalizeDf = self.workingDf.copy()
        for k in xQuantLabels + yLabels + AddedLabels:
            colMean, colStd = self.workingDf[k].mean(axis=0), self.workingDf[k].std(axis=0)
            colMax, colMin = self.workingDf[k].max(axis=0), self.workingDf[k].min(axis=0)
            self.scaleDf[k] = (self.workingDf[k] - colMean) / colStd
            self.normalizeDf[k] = (self.workingDf[k] - colMin) / (colMax - colMin)


    def createSubsets(self, splittingFt, df):

        # >> 4 SUBSET DATAFRAME
        subsets = []
        feature_dict = dict()
        for quality in self.possibleQualities[splittingFt]:
            newsubset = df.loc[df[splittingFt] == quality]
            subsets.append(newsubset)
            feature_dict[quality] = newsubset
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



    def studyDatabase(self, path, splittingFt, labels= ['rawDf', 'workingDf', 'scaleDf', 'normalizeDf']):

        qualities = self.possibleQualities[splittingFt]
        singleDf_List = [self.__getattribute__(l) for l in labels]
        subsetDf_list_unmerged = [self.createSubsets(splittingFt, self.__getattribute__(l)) for l in labels]

        singleDA_List = []
        for df in singleDf_List:
            dfDA = self.analyzeDataframe(df)
            singleDA_List.append(dfDA)

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

        # tables = singleDf_List + singleDA_List + subsetDf_list + subsetDA_list
        all_labels = []
        all_tables = []
        # for pair in [[singleDf_List, singleDA_List][subsetDf_list, subsetDA_list]]:
        for df, DA, lab in zip(singleDf_List, singleDA_List, labels):
            all_tables.append(df)
            all_tables.append(DA)
            all_labels.append(lab)
            all_labels.append(lab + 'DA')
        for df, DA, lab in zip(subsetDf_list, subsetDA_list, labels):
            all_tables.append(df)
            all_tables.append(DA)
            all_labels.append(lab + splittingFt)
            all_labels.append(lab + 'DA' + splittingFt)

        if displayParams['archive']:
            import os

            reference = displayParams['ref_prefix'] + '_Combined/'
            outputPathStudy = path + "RESULTS/" + reference + 'RECORDS/' + 'DATA' + '/'
            if not os.path.isdir(outputPathStudy):
                os.makedirs(outputPathStudy)

            with pd.ExcelWriter(outputPathStudy + 'DataAnalysis' + ".xlsx", mode='w') as writer:

                for df, name in zip(all_tables, all_labels):
                    df.to_excel(writer, sheet_name=name)


        pickleDumpMe(path, displayParams, DA, 'DATA', 'DataAnalysis', combined=True)


# ggplot

DA = DataAnalysis(path, dbName, delimiter, firstLine, xQualLabels, xQuantLabels, yLabels,
                 Summed_Labels, Divided_Labels)
DA.studyDatabase(path, splittingFt, labels= ['rawDf', 'workingDf', 'scaleDf', 'normalizeDf'])




#
#
# # STOCK
# pickleDumpMe(DB_Values['DBpath'], displayParams, dataframe, 'DATA', 'rdat', combined=True)
# dfAsTable(DB_Values['DBpath'], displayParams, dataframe, objFolder='DATA', name = "DF", combined = True)