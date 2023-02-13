#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
# from Dashboard_EUCB_FR import *
from Dashboard_EUCB_FR_v2 import *

from Raw import *
from Features import *
from Main_FS_Steps import *
from Main_GS_FS_Steps import *
import shap


class Sample:
    def __init__(self, dbRefName, MyPred_Sample): #, delimiter, firstLine, xQualLabels, xQuantLabels, model):#self, path, dbName, delimiter, firstLine, xQualLabels, xQuantLabels, yLabels, updateLabels=None

        """

        """

        # IMPORT
        rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(dbRefName, show = False)
        mean = baseFormatedDf.MeanStdDf.loc["mean",:]
        std = baseFormatedDf.MeanStdDf.loc["std",:]
        self.name = MyPred_Sample["DBname"]

        # RAW DATA
        from Raw import open_csv_at_given_line
        header, reader = open_csv_at_given_line(path=MyPred_Sample['DBpath'], dbName=MyPred_Sample['DBname'],
                                                delimiter=MyPred_Sample['DBdelimiter'],
                                                firstLine=MyPred_Sample['DBfirstLine'])

        # format new input to data the ML algo was trained on
        self.xQuali = {k: [] for k in rdat.xQuali.keys()}
        self.xQuanti = {k: [] for k in rdat.xQuanti.keys()}
        self.y = {k: [] for k in rdat.y.keys()}
        self.possibleQualities = rdat.possibleQualities

        for line in reader:
            for (labels, attribute) in [(xQuantLabels, self.xQuanti), (yLabels, self.y)]:
                for label in labels:
                    attribute[label].append(float(line[header.index(label)].replace(',', '.')))
            for label in xQualLabels:
                self.xQuali[label].append(line[header.index(label)])

        input = self.xQuanti.copy()
        input.update(self.xQuali)
        self.input = pd.DataFrame.from_dict(input)

        for label in self.xQuali.keys():
            self.xQuali[label] = [self.possibleQualities[label].index(value) for value in self.xQuali[label]]

        # FEATURES
        from Features import logitize

        # UNSCALED
        XDfunsc = self.xQuanti.copy()
        XDfunsc.update(logitize(self.xQuali, self.possibleQualities))
        self.XDfunsc = pd.DataFrame.from_dict(XDfunsc) #input data digitized but unscaled

        #SCALED
        self.x = dict(self.xQuanti)
        for l in self.x.keys():
            self.x[l] = list((self.x[l] - mean[l]) / std[l])
        self.x.update(logitize(self.xQuali, self.possibleQualities))

        XDf = self.asDataframe().drop(columns=yLabels)
        yDf = np.multiply(self.asDataframe()[yLabels], FORMAT_Values['yUnitFactor'])
        self.yDf = yDf
        self.XDf = XDf
        self.yDf.rename(columns={yLabels[0]: FORMAT_Values['targetLabels'][0]})

        #todo : __new__ features object in whicjh attributes provided
        #https://stackoverflow.com/questions/47169489/how-to-create-an-object-inside-class-static-method
        #https://realpython.com/python-class-constructor/

    def asDataframes(self, batchCount=5, powers=None, mixVariables=None):
        x, y, xlabels = self.asArray(powers, mixVariables)
        cutoffIndex = batchCount if x.shape[0] % batchCount == 0 \
            else [int(x.shape[0] / batchCount * i) for i in range(1, batchCount)]
        return np.split(x, cutoffIndex), np.split(y, cutoffIndex), xlabels

    def asArray(self, powers={}, mixVariables=[]):
        numValues = len(next(iter(self.x.values())))
        x = np.zeros((numValues, len(self.x) - len(powers)))
        y = np.zeros((numValues, len(self.y)))
        xlabels = [f for f in self.x.keys() if f not in powers.keys()]
        for i in range(numValues):  # 80
            x[i, :] = np.array([self.x[f][i] for f in self.x.keys() if f not in powers.keys()])
            y[i, :] = np.array([self.y[f][i] for f in self.y.keys()])
        return x, y, xlabels

    def asDataframe(self, powers={}, mixVariables=[]):
        x, y, xlabels = self.asArray(powers, mixVariables)
        self.Dataframe = [x, y]
        return pd.DataFrame(np.hstack((x, y)), columns=xlabels + list(self.y.keys()))

    def formatDf(self, data, model):

        XDf = data
        if model.learningDf.droppedLabels != '':
            droppedLabels = model.learningDf.droppedLabels
            XDf = data.drop(columns=droppedLabels)
        return XDf

    def SamplePrediction(self, model):

        XDf = self.formatDf(self.XDf, model)
        yPred = model.Estimator.predict(XDf)
        return yPred

    def SHAP_WaterfallPlot(self, model, DBpath, content = "WaterfallPlot"):

        # todo : def doesn't work vor Kernel Explainer

        XDf = self.formatDf(self.XDf, model)
        features = self.formatDf(self.XDfunsc, model).round(3) # to indicate unscaled values on axis -  no attributre found - doesn't work
        sample = self.input.to_string(index=False)
        name = 'SHAP_' + content + '_' + model.GSName

        #SHAP
        explainer = model.SHAPexplainer

        try:
            shap_values = explainer(XDf)  # explainer.shap_values(XDf)
            shap_wf = shap.waterfall_plot(shap_values=shap_values[0], show=displayParams['showPlot'])

            # EDIT PLOT
            plt.gcf().set_size_inches(20, 6)
            plt.tight_layout()
            plt.suptitle(name, ha='center', size='small', va='top')
            # plt.suptitle(sample, ha='center', size='small', va = 'top')

            # SAVE
            reference = displayParams['reference']
            if displayParams['archive']:
                path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + 'PREDICTIONS/' + self.name
                import os
                outputFigPath = path + folder + subFolder
                if not os.path.isdir(outputFigPath):
                    os.makedirs(outputFigPath)
                plt.savefig(outputFigPath + '/' + name + '.png')
                print("file saved :", outputFigPath + '/' + name + '.png')
                if displayParams['showPlot']:
                    plt.show()

            plt.close()


        except Exception:
            pass
            # shap_values = explainer.shap_values(XDf)  # explainer.shap_values(XDf)
            # expected_value = explainer.expected_value
            # base_values = explainer.base_values
            # shap_wf = shap.waterfall_plot(shap_values[0], base_values, show=displayParams['showPlot'])






    def SHAP_ForcePlot(self, model, DBpath, content = "ForcePlot", sampleOnly = True ):

        try:

            name = 'SHAP_' + content + '_' + model.GSName

            explainer = model.SHAPexplainer
            if sampleOnly:
                XDf = self.formatDf(self.XDf, model)
                features = self.formatDf(self.XDfunsc, model).round(3)
                shap_values = explainer(XDf)[0].values # plot force for data sample only
                name +=  'Sample'
            else:
                features = model.learningDf.XTest
                shap_values = model.SHAPvalues #explainer(model.learningDf.XTest) # plot force for all testing data / exclude sample
                name +=  'Testing'
                #todo : UPDATE THIS - not working currently since matplotlib attribute doesn't work for multiple samples ..

            expected_value = model.SHAPexplainer.expected_value
            shap_wf = shap.force_plot(base_value = expected_value, shap_values = shap_values, features = features, matplotlib= True,
                                       show = displayParams['showPlot'], text_rotation=45, plot_cmap = ["#ca0020", "#92c5de"]) #, show = True

            plt.gcf().set_size_inches(20, 6)
            plt.tight_layout()
            plt.suptitle(name, ha='right', size='large')

            reference = displayParams['reference']
            if displayParams['archive']:
                path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + 'PREDICTIONS/' + self.name
                import os
                outputFigPath = path + folder + subFolder
                if not os.path.isdir(outputFigPath):
                    os.makedirs(outputFigPath)
                plt.savefig(outputFigPath + '/' + name + '.png')
                print("file saved :", outputFigPath + '/' + name + '.png')

                if displayParams['showPlot']:
                    plt.show()

            plt.close()

        except Exception:
            pass
            # shap_values = explainer.shap_values(XDf)  # explainer.shap_values(XDf)
            # expected_value = explainer.expected_value
            # base_values = explainer.base_values
            # shap_wf = shap.waterfall_plot(shap_values[0], base_values, show=displayParams['showPlot'])

    def SHAP_ScatterPlot(self, model, DBpath, feature = "Gross_Floor_Area",  content = "ScatterPlot"):

        try:

            XDf = self.formatDf(self.XDf, model)
            explainer = model.SHAPexplainer
            shap_values = explainer(model.learningDf.XTest)
            shap_wf = shap.plots.scatter(shap_values[:, feature], show = displayParams['showPlot'])

            name = 'SHAP_' + content + '_' + model.GSName

            plt.gcf().set_size_inches(20, 6)
            plt.tight_layout()
            plt.suptitle(name, ha='right', size='large')

            reference = displayParams['reference']
            if displayParams['archive']:
                path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + 'PREDICTIONS/' + self.name
                import os
                outputFigPath = path + folder + subFolder
                if not os.path.isdir(outputFigPath):
                    os.makedirs(outputFigPath)
                plt.savefig(outputFigPath + '/' + name + '.png')
                print("file saved :", outputFigPath + '/' + name + '.png')

                if displayParams['showPlot']:
                    plt.show()

            plt.close()

        except Exception:
            pass
            # shap_values = explainer.shap_values(XDf)  # explainer.shap_values(XDf)
            # expected_value = explainer.expected_value
            # base_values = explainer.base_values
            # shap_wf = shap.waterfall_plot(shap_values[0], base_values, show=displayParams['showPlot'])


#REFERENCE

for set in studyParams['sets']:
    yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set
    DBname = DB_Values['acronym'] + '_' + yLabelsAc + '_rd'

    for value in studyParams['randomvalues']:
        PROCESS_VALUES['random_state'] = value
yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = studyParams['sets'][0]
displayParams["reference"] = DB_Values['acronym'] + '_' + yLabelsAc + '_rd' + str(PROCESS_VALUES['random_state']) + '/'

# MODEL

GS_FSs = import_Main_GS_FS(displayParams["reference"], GS_FS_List_Labels = studyParams['Regressors'])
Model_List_All = unpackGS_FSs(GS_FSs, remove='')
print(Model_List_All)
Model_List = [Model_List_All[-1]]
print(Model_List)

def Run_Prediction(Model_List, MyPred_Sample, ArchPath):

    sample = Sample(displayParams["reference"], MyPred_Sample)

    pickleDumpMe(ArchPath, displayParams, sample, 'PREDICTIONS', MyPred_Sample["DBname"])
    predDict = dict()
    for model in Model_List:
        pred = sample.SamplePrediction(model)
        predDict[model.GSName] = pred

        sample.SHAP_WaterfallPlot(model, DB_Values['DBpath'])
        sample.SHAP_ScatterPlot(model, DB_Values['DBpath'])
        sample.SHAP_ForcePlot(model, DB_Values['DBpath'])

    predDf = pd.DataFrame.from_dict(predDict)


    if displayParams['archive']:

        import os
        reference = displayParams['reference']
        outputPathStudy = DB_Values['DBpath'] + "RESULTS/" + reference + 'RECORDS/' + 'PREDICTIONS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with pd.ExcelWriter(outputPathStudy + sample.name + '_Pred_Records_All' + ".xlsx", mode='w', if_sheet_exists="overlay") as writer:
            sample.input.T.to_excel(writer, sheet_name="Sheet1")
            predDf.to_excel(writer, sheet_name="Sheet1", startrow=len(sample.input.T) + 5)


Run_Prediction(Model_List_All, MyPred_Sample, DB_Values['DBpath'])




