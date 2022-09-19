import csv
from dashBoard import *


def open_csv_at_given_line(path, dbName, delimiter, firstLine, folder="DATA/"):

    """
    Opens a csv at a given line
    returns :
    !! input csv file must be semicolon delimited !!
    header - list of all the feature categories
    reader - reader that reads through the feature values

    """

    reader = csv.reader(open(path + folder + dbName + '.csv', mode='r'), delimiter=delimiter)
    for i in range(firstLine):
        reader.__next__()
    header = reader.__next__()
    return header, reader

def setWorkingFeatures(header, reader, newLabels = False):

    """
    Set labels to use for training - else use default
    If new labels :
    Labels inserted in the prompt should be inserted as text in quotation marks, separated by commas, with no spacing
    ex :
        xQualLabels = 'Sector','Type'
        xQuantLabels = 'GIFA (m2)','Storeys'
        yLabels = 'Calculated tCO2e_per_m2'

    """

    if not newLabels:
        xQualLabels = ['Sector', 'Type', 'Basement', 'Foundations', 'Ground Floor', 'Superstructure', 'Cladding','BREEAM Rating']
        xQuantLabels = ['GIFA (m2)', 'Storeys', 'Typical Span (m)', 'Typ Qk (kN_per_m2)']  #
        yLabels = ['Calculated tCO2e_per_m2']  # 'Calculated Total tCO2e',
    else:
        print('List of labels from database :',  header)
        xQualLabels = [item.replace("'", '') for item in input("Enter xQualLabels items, in quotation marks, spaced by a comma: ").split(',')]
        xQuantLabels = [item.replace("'", '') for item in input("Enter xQuantLabels items, in quotation marks, spaced by a comma: ").split(',')]
        yLabels = [item.replace("'", '') for item in input("Enter yLabels items, in quotation marks, spaced by a comma: ").split(',')]

    return xQualLabels, xQuantLabels, yLabels

class RawData:
    def __init__(self, path = DBpath, dbName = DBname, delimiter = DBdelimiter, firstLine = DBfirstLine, newLabels = None):

        """
        Opens a csv at a given line
        returns :
        xQuali - dictionary of qualitative features and labels
        xQuanti - dictionary of quantitative features and labels

        """
        header, reader = open_csv_at_given_line(path, dbName, delimiter, firstLine)
        xQualLabels, xQuantLabels, yLabels = setWorkingFeatures(header, reader, newLabels)

        self.xQuali = {k: [] for k in xQualLabels}
        self.xQuanti = {k: [] for k in xQuantLabels}
        self.y = {k: [] for k in yLabels}

        for line in reader:
            for (labels, attribute) in [(xQuantLabels, self.xQuanti), (yLabels, self.y)]:
                for label in labels:
                    attribute[label].append(float(line[header.index(label)].replace(',', '.')))
            for label in xQualLabels:
                self.xQuali[label].append(line[header.index(label)])
        self.possibleQualities = dict()
        self.digitalize()


    def digitalize(self):
        # converts labels (string) to numbers (int)
        self.enumeratePossibleQualities()
        for label in self.xQuali.keys():
            self.xQuali[label] = [self.possibleQualities[label].index(value) for value in self.xQuali[label]]

    def enumeratePossibleQualities(self):
        for label, column in self.xQuali.items():
            self.possibleQualities[label] = []
            for value in column:
                if value not in self.possibleQualities[label]:
                    self.possibleQualities[label].append(value)

    def visualize(self, displayParams, path=DBpath, yLabel='Calculated tCO2e_per_m2', xLabel='Cladding',
                  title = "Features influencing CO2 footprint of Structures - Datasource : Price & Myers",
                  reference = "", figure_size = (8, 10), folder = "RESULTS/",subFolder ='visualizeRawData'):

        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        if xLabel in self.xQuali.keys():
            xList = self.xQuali[xLabel]
            labels = self.possibleQualities[xLabel]
        else:
            xList = self.xQuanti[xLabel]
        yList = self.y[yLabel]
        df = pd.DataFrame(list(zip(xList, yList)), columns=[xLabel, yLabel])
        fig, ax = plt.subplots(figsize=figure_size)
        ax.set_title(title + " " + reference)
        if xLabel in self.xQuali.keys():
            x = np.arange(len(labels))
            ax.set_ylabel(yLabel)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
                 rotation_mode="anchor")
        sns.scatterplot(data=df, x=xLabel, y=yLabel, hue=yLabel, ax=ax)

        if displayParams['archive']:
            import os
            outputFigPath = path + folder + subFolder

            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + xLabel + '-' + yLabel + '.png')

        if displayParams['showPlot']:
            plt.show()

        plt.close() #todo : check this

# csvPath = "C:/Users/sfenton/Code/Repositories/CO2Prediction/DATA/210413_PM_CO2_data-sepp"
# csvPath2 = "C:/Users/sfenton/Code/Repositories/CO2Prediction/DATA/210413_PM_CO2_data"
# outputPath = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'


rdat = RawData()
#
# rdat2 = RawData(csvPath2, ';', 5, newLabels = True)

print(rdat.xQuali)
print(rdat.xQuanti)
# print(rdat2.xQuali)
# print(rdat2.xQuanti)
print(rdat.y)
rdat.visualize(displayParams, DBpath)