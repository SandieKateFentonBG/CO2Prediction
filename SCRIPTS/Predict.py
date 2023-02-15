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
        self.mean = baseFormatedDf.MeanStdDf.loc["mean",:]
        self.std = baseFormatedDf.MeanStdDf.loc["std",:]
        self.yLabels = studyParams['sets'][0][0]
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
            for (labels, attribute) in [(xQuantLabels, self.xQuanti), (self.yLabels, self.y)]:
                for label in labels:
                    attribute[label].append(float(line[header.index(label)].replace(',', '.')))
            for label in xQualLabels:
                self.xQuali[label].append(line[header.index(label)])
        self.xQualiDict = self.xQuali.copy()
        self.xQuantiDict = self.xQuanti.copy()

        self.createSample(xQuali=self.xQuali, xQuanti=self.xQuanti)

    def createSample(self, xQuali, xQuanti):

        self.xQuali = xQuali
        self.xQuanti = xQuanti

        input = xQuanti.copy()
        input.update(xQuali)
        self.input = pd.DataFrame.from_dict(input)

        for label in xQuali.keys():
            xQuali[label] = [self.possibleQualities[label].index(value) for value in xQuali[label]]

        # FEATURES
        from Features import logitize

        # UNSCALED
        XDfunsc = self.xQuanti.copy()
        XDfunsc.update(logitize(xQuali, self.possibleQualities))
        self.XDfunsc = pd.DataFrame.from_dict(XDfunsc) #input data digitized but unscaled

        #SCALED
        self.x = dict(xQuanti)
        for l in self.x.keys():
            self.x[l] = list((self.x[l] - self.mean[l]) / self.std[l])
        self.x.update(logitize(xQuali, self.possibleQualities))

        XDf = self.asDataframe().drop(columns=self.yLabels)
        yDf = np.multiply(self.asDataframe()[self.yLabels], FORMAT_Values['yUnitFactor'])
        self.yDf = yDf
        self.XDf = XDf
        self.yDf.rename(columns={self.yLabels[0]: FORMAT_Values['targetLabels'][0]})

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

    def blendformatDf(self, blender):

        #create meta learning data
        blend_sample_sets = []
        for model in blender.modelList:

            XDf = self.formatDf(self.XDf, model)
            blend_train_i = model.Estimator.predict(XDf)
            blend_train_i = pd.DataFrame(blend_train_i)
            blend_sample_sets.append(blend_train_i)

        blendXDf = pd.concat(blend_sample_sets, axis=1)
        blendXDf = (blendXDf - blender.ScaleMean) / blender.ScaleStd

        return blendXDf

    def SamplePredictionBlender(self, blender):

        blendXDf = self.blendformatDf(blender)
        yPred = blender.Estimator.predict(blendXDf)
        return yPred

    def SHAP_WaterfallPlot(self, model, explainer, DBpath, content = "WaterfallPlot", Blender = False):

        # todo : def doesn't work vor Kernel Explainer

        if Blender:
            XDf = self.blendformatDf(model)

        else:
            XDf = self.formatDf(self.XDf, model)
            features = self.formatDf(self.XDfunsc, model).round(3) # to indicate unscaled values on axis -  no attributre found - doesn't work


        sample = self.input.to_string(index=False)
        name = self.name + '_' + content + '_' + model.GSName

        try :
            shap_values = explainer(XDf)  # explainer.shap_values(XDf)
            shap_wf = shap.waterfall_plot(shap_values=shap_values[0], show=displayParams['showPlot'], max_display=24)

        except Exception:
            sv = explainer.shap_values(XDf)
            bv = explainer.expected_value
            exp = shap.Explanation(sv, bv, XDf) #, feature_names=None
            idx = 0  # datapoint to explain
            shap_wf = shap.waterfall_plot(exp[idx], show=displayParams['showPlot'], max_display=24)

        # EDIT PLOT
        plt.gcf().set_size_inches(20, 10)
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


    def SHAP_ForcePlot(self, model, explainer, DBpath, content = "ForcePlot", sampleOnly = True, Blender = False ):

        name = self.name + '_' + content + '_' + model.GSName

        if sampleOnly:

            if Blender:
                XDf = self.blendformatDf(model)
                features = []
            else:
                XDf = self.formatDf(self.XDf, model)
                features = self.formatDf(self.XDfunsc, model).round(3)
            name += 'Sample'
        else:
            features = model.learningDf.XTest
            name += 'Testing'
            # todo : UPDATE THIS - not working currently since matplotlib attribute doesn't work for multiple samples ..

        try:
            if sampleOnly:
                shap_values = explainer(XDf)[0].values # plot force for data sample only

            else:
                shap_values = model.SHAPvalues #explainer(model.learningDf.XTest) # plot force for all testing data / exclude sample

            expected_value = model.SHAPexplainer.expected_value
            shap_wf = shap.force_plot(base_value = expected_value, shap_values = shap_values, features = features, matplotlib= True,
                                       show = displayParams['showPlot'], text_rotation=45, plot_cmap = ["#ca0020", "#92c5de"]) #, show = True

        except Exception:
            sv = explainer.shap_values(XDf)
            bv = explainer.expected_value
            exp = shap.Explanation(sv, bv, XDf)  # , feature_names=None
            idx = 0  # datapoint to explain

            shap_wf = shap.force_plot(base_value = bv, shap_values = sv, features = features, matplotlib= True,
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



    def SHAP_ScatterPlot(self, model, explainer, DBpath, feature = "Users_Total", content = "ScatterPlot", Blender = False): #Main_Material_Timber, wood "Gross_Floor_Area"

        name = self.name + '_' + content + '_' + model.GSName

        if Blender:
            XDf = self.blendformatDf(model)
        else :
            XDf = self.formatDf(self.XDf, model)

        try:
            shap_values = explainer(model.learningDf.XTest)
            shap_wf = shap.plots.scatter(shap_values[:, feature], show = displayParams['showPlot'])

            # sv = explainer.shap_values(model.learningDf.XTest)
            # bv = explainer.expected_value
            # id = XDf.columns.get_loc(feature)
            # fn = np.array(model.selectedLabels).reshape(1,-1)
            # exp = shap.Explanation(sv, bv, XDf.to_numpy()[0], feature_names=fn)
            #
            # shap_wf = shap.plots.scatter(exp[:, id], show = displayParams['showPlot'])

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





