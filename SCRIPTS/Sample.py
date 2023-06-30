#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
# from Dashboard_EUCB_FR import *

from Main_GS_FS_Steps import *
from HelpersFormatter import *
import shap




class Sample:
    def __init__(self, displayParams, MyPred_Sample): #, delimiter, firstLine, xQualLabels, xQuantLabels, model):#self, path, dbName, delimiter, firstLine, xQualLabels, xQuantLabels, yLabels, updateLabels=None

        """

        """

        import_combined_ref = displayParams['ref_prefix'] + '_Combined/'
        import_single_ref = displayParams['reference']

        rdat = pickleLoadMe(path=DB_Values['DBpath'] + 'RESULTS/' + import_combined_ref + 'RECORDS/DATA/rdat.pkl',
                            show=False)
        dat = pickleLoadMe(path=DB_Values['DBpath'] + 'RESULTS/' + import_combined_ref + 'RECORDS/DATA/dat.pkl')

        baseFormatedDf = pickleLoadMe(
            path=DB_Values['DBpath'] + 'RESULTS/' + import_single_ref + 'RECORDS/DATA/baseFormatedDf.pkl', show=False)

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
        self.droppedLabels = dat.droppedLabels
        self.removedDict = dat.removedDict
        self.selectedDict = dict()
        for k, v in self.possibleQualities.items():
            self.selectedDict[k] = [elem for elem in self.possibleQualities[k] if elem not in self.removedDict[k]]

        for line in reader:
            for (labels, attribute) in [(xQuantLabels, self.xQuanti), (self.yLabels, self.y)]:
                for label in labels:
                    attribute[label].append(float(line[header.index(label)].replace(',', '.')))
            for label in xQualLabels:
                self.xQuali[label].append(line[header.index(label)])
        self.xQualiDict = self.xQuali.copy()
        self.xQuantiDict = self.xQuanti.copy()

        self.createSample(xQuali=self.xQuali, xQuanti=self.xQuanti)
        self.SHAPGroupKeys = None
        self.SHAPGroupvalues = None

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

        if self.droppedLabels != []:
            droppedLabels = self.droppedLabels
            self.XDf = XDf.drop(columns=droppedLabels)
            self.XDfunsc = self.XDfunsc.drop(columns=droppedLabels)

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

    def SamplePrediction(self, model):

        XDf = formatDf(self.XDf, model)
        yPred = model.Estimator.predict(XDf)
        return yPred

    def SamplePredictionBlender(self, blender):

        blendXDf = formatDf_toBlender(self.XDf, blender)
        yPred = blender.Estimator.predict(blendXDf)
        return yPred

    def SHAP_WaterfallPlot(self, model, explainer, DBpath, content = "WaterfallPlot", Grouped = False, Blender = False):

        # todo : def doesn't work vor Kernel Explainer

        if Blender:
            XDf = formatDf_toBlender(self.XDf, model)

        else:
            XDf = formatDf(self.XDf, model)
            features = formatDf(self.XDfunsc, model).round(3) # to indicate unscaled values on axis -  no attributre found - doesn't work

        sv = explainer.shap_values(XDf)
        bv = explainer.expected_value
        exp = shap.Explanation(sv, bv, XDf) #, feature_names=None
        idx = 0  # datapoint to explain

        myExplainer = exp[idx]

        extra = '_Ungrouped'
        pltheight = 10

        if Grouped : #only do this for no selector - for understanding of full group values

            pltheight = 5

            if model.learningDf.selector == 'NoSelector':

                SHAPGroupKeys, SHAPGroupvalues = self.group_data(model, exp[idx].values)
                myExplainer.__setattr__('feature_names', SHAPGroupKeys)
                myExplainer.__setattr__('values', SHAPGroupvalues)
                myExplainer.__setattr__('features', formatDf(self.input, model))
                myExplainer.__setattr__('data', formatDf(self.input, model).T.squeeze())
                extra = '_Grouped'

                # todo : pickledump this myExplainer >

            else:
                return

        # sample = self.input.to_string(index=False)
        name = self.name + '_' + content + extra + '_' + model.GSName

        shap_wf = shap.waterfall_plot(myExplainer, show=displayParams['showPlot'], max_display=24)

        # EDIT PLOT
        plt.gcf().set_size_inches(12, pltheight)
        plt.tight_layout()
        # plt.suptitle(name, ha='center', size='small', va='top')
        fontsize = 12
        # Adjust the font size of the plot labels
        plt.xticks(fontsize=fontsize)  # Increase the font size of the x-axis tick labels
        plt.yticks(fontsize=fontsize)  # Increase the font size of the y-axis tick labels
        plt.rcParams.update({'font.size': fontsize})  # Adjust the font size to your desired value

        # plt.suptitle(sample, ha='center', size='small', va = 'top')

        # SAVE
        reference = displayParams['reference']
        if displayParams['archive']:
            path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + 'PREDICTIONS/' + self.name + '/WATERFALL'
            import os
            outputFigPath = path + folder + subFolder
            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)
            plt.savefig(outputFigPath + '/' + name + '.png')
            print("file saved :", outputFigPath + '/' + name + '.png')
            if displayParams['showPlot']:
                plt.show()

        plt.close()

        self.__setattr__('explainer', myExplainer)
        self.explainer = myExplainer

    def group_data(self, model, shap_values):

        # compute new SHAP values

        #transform data
        SHAPGroupKeys = list(model.SHAPGroup_RemapDict.keys())
        lengthList = list(model.SHAPGroup_RemapDict.values())

        # split shap values into a list for each feature
        values_split = np.split(shap_values, np.cumsum(lengthList[:-1]))

        # sum values within each list
        SHAPGroupvalues = [sum(l) for l in values_split]
        #format values to array
        SHAPGroupvalues = np.array(SHAPGroupvalues)

        self.SHAPGroupKeys = SHAPGroupKeys
        self.SHAPGroupvalues = SHAPGroupvalues

        return SHAPGroupKeys, SHAPGroupvalues


    # def SHAP_WaterfallPlot_Grouped(self, model, explainer, DBpath, content = "WaterfallPlot_Grouped", Blender = False):
    #
    #     # todo : def doesn't work for Kernel Explainer
    #
    #     if Blender:
    #         XDf = formatDf_toBlender(self.XDf, model)
    #
    #     else:
    #         XDf = formatDf(self.XDf, model)
    #         features = formatDf(self.XDfunsc, model).round(3) # to indicate unscaled values on axis -  no attributre found - doesn't work
    #
    #
    #     sample = self.input.to_string(index=False)
    #     name = self.name + '_' + content + '_' + model.GSName
    #
    #     try :
    #         shap_values = explainer(XDf)  # explainer.shap_values(XDf)
    #         SHAPGroupKeys, SHAPGroupvalues = self.group_data(model, shap_values[0])
    #         shap_wf = shap.waterfall_plot(shap_values=SHAPGroupvalues, feature_names=SHAPGroupKeys,
    #                                       show=displayParams['showPlot'], max_display=24)
    #
    #     except Exception:
    #
    #         sv = explainer.shap_values(XDf)
    #         bv = explainer.expected_value
    #         exp = shap.Explanation(sv, bv, XDf) #, feature_names=None
    #         idx = 0  # datapoint to explain
    #         myExplainer = exp[idx]
    #
    #         SHAPGroupKeys, SHAPGroupvalues = self.group_data(model, exp[idx].values)
    #         myExplainer.__setattr__('feature_names', SHAPGroupKeys)
    #         myExplainer.__setattr__('values', SHAPGroupvalues)
    #
    #         shap_wf = shap.waterfall_plot(myExplainer,
    #                                       show=displayParams['showPlot'], max_display=24) #feature_names=SHAPGroupKeys,
    #
    #     # EDIT PLOT
    #     plt.gcf().set_size_inches(20, 10)
    #     plt.tight_layout()
    #     plt.suptitle(name, ha='center', size='small', va='top')
    #     # plt.suptitle(sample, ha='center', size='small', va = 'top')
    #
    #     # SAVE
    #     reference = displayParams['reference']
    #     if displayParams['archive']:
    #         path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + 'PREDICTIONS/' + self.name
    #         import os
    #         outputFigPath = path + folder + subFolder
    #         if not os.path.isdir(outputFigPath):
    #             os.makedirs(outputFigPath)
    #         plt.savefig(outputFigPath + '/' + name + '.png')
    #         print("file saved :", outputFigPath + '/' + name + '.png')
    #         if displayParams['showPlot']:
    #             plt.show()
    #
    #     plt.close()


    def SHAP_ForcePlot(self, model, explainer, DBpath, content = "ForcePlot", sampleOnly = True, Grouped = False, Blender = False ):

        name = self.name + '_' + content + '_' + model.GSName

        if sampleOnly:

            if Blender:
                XDf = formatDf_toBlender(self.XDf, model)
                features = []
            else:
                XDf = formatDf(self.XDf, model)
                features = formatDf(self.XDfunsc, model).round(3)
            name += 'Sample'

        else:
            features = model.learningDf.XTest
            name += 'Testing'
            # todo : UPDATE THIS - not working currently since matplotlib attribute doesn't work for multiple samples ..

        # try:
        #     if sampleOnly:
        #         shap_values = explainer(XDf)[0].values # plot force for data sample only
        #         feature_names = explainer(XDf)[0].feature_names
        #         print("1")
        #         print("shap_values", len(shap_values), shap_values)
        #         print("features", len(features), features)
        #
        #         if Grouped:
        #
        #
        #
        #             print("2")
        #             feature_names, shap_values = self.group_data(model, shap_values)
        #             print("shap_values", len(shap_values), shap_values)
        #             features = self.input
        #             print("features", len(features), features)
        #             name += '_Grouped'
        #
        #     else:
        #         shap_values = model.SHAPvalues #explainer(model.learningDf.XTest) # plot force for all testing data / exclude sample
        #
        #
        #     expected_value = model.SHAPexplainer.expected_value
        #
        #
        #     shap_wf = shap.force_plot(base_value = expected_value, shap_values = shap_values, features = features, feature_names = feature_names, matplotlib= True,
        #                                show = displayParams['showPlot'], text_rotation=45, plot_cmap = ["#ca0020", "#92c5de"]) #, show = True
        #
        # except Exception:
        sv = explainer.shap_values(XDf)
        bv = explainer.expected_value
        exp = shap.Explanation(sv, bv, XDf)  # , feature_names=None
        idx = 0  # datapoint to explain

        if Grouped: #only do this for no selector - for understanding full group values

            if model.learningDf.selector == 'NoSelector':

                feature_names, sv = self.group_data(model, exp[idx].values)
                features = formatDf(self.input, model)
                name += '_Grouped'

            else :
                pass

        shap_wf = shap.force_plot(base_value = bv, shap_values = sv, features = features, matplotlib= True,
                                   show = displayParams['showPlot'], text_rotation=45, plot_cmap = ["#ca0020", "#92c5de"]) #, show = True

        plt.gcf().set_size_inches(20, 6)
        plt.tight_layout()
        plt.suptitle(name, ha='right', size='large')

        reference = displayParams['reference']
        if displayParams['archive']:
            path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + 'PREDICTIONS/' + self.name + '/FORCE'
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
            XDf = formatDf_toBlender(self.XDf, model)
        else :
            XDf = formatDf(self.XDf, model)

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
                path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + 'PREDICTIONS/' + self.name + '/SCATTER'
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
            "didn't' work"
            pass



def import_SAMPLE(ref_single, name):

    path = DB_Values['DBpath'] + 'RESULTS/' + ref_single + 'RECORDS/PREDICTIONS/' + name + '.pkl'
    SA = pickleLoadMe(path=path, show=False)

    return SA

