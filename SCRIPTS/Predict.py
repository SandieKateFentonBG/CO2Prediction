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
        for label in self.xQuali.keys():
            self.xQuali[label] = [self.possibleQualities[label].index(value) for value in self.xQuali[label]]

        # FEATURES
        from Features import logitize
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

    def SamplePrediction(self, model):

        # if hasattr(model.learningDf, "droppedLabels"):
        print("here", model.learningDf, model.learningDf.droppedLabels)
        XDf = self.XDf
        if model.learningDf.droppedLabels != ['']:
            droppedLabels = model.learningDf.droppedLabels
            print("droppedLabels", len(droppedLabels), droppedLabels)
            XDf = self.XDf.drop(columns=droppedLabels)

        yPred = model.Estimator.predict(XDf)
        return yPred



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
Model_List = unpackGS_FSs(GS_FSs, remove = '')
model = Model_List[0] #CHECK FOR NoSelector
print("model", model.GSName)


test = Sample(displayParams["reference"], MyPred_Sample)
print(test.asDataframe())

pred = test.SamplePrediction(model)

print(pred)



# TODO : SHAP PLOT !



#
# self.model = model
#
# self.xQuali = MyPred_rdat.xQuali
# self.xQuanti = MyPred_rdat.xQuanti
# self.XDf = MyPred_df
# self.
#
# explainer = self.model.SHAPexplainer
# shap_values = explainer(self.XDf)
# shap.plots.waterfall(shap_values)


#PROBLEM WITH LOGITIZINGGGG