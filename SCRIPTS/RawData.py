import csv


def open_csv_at_given_line(csvPath, delimiter, firstLine):
    reader = csv.reader(open(csvPath + '.csv', mode='r'), delimiter=delimiter)
    for i in range(firstLine):
        reader.__next__()
    header = reader.__next__()
    return header, reader


class RawData:
    def __init__(self, csvPath, delimiter, firstLine,
                 xQualLabels, xQuantLabels, yLabels):
        self.xQuali = {k: [] for k in xQualLabels}
        self.xQuanti = {k: [] for k in xQuantLabels}
        self.y = {k: [] for k in yLabels}
        header, reader = open_csv_at_given_line(csvPath, delimiter, firstLine)
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

    def visualize(self, displayParams, yLabel= 'Calculated Total tCO2e', xLabel= 'Cladding',
                  title = "Features influencing CO2 footprint of Structures - Datasource : Price & Myers",
                  reference = "", figure_size = (8, 10), folder = 'visualizeRawData'):

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
            outputFigPath = displayParams["outputPath"] + '/' + folder

            if not os.path.isdir(outputFigPath):
                os.makedirs(outputFigPath)

            plt.savefig(outputFigPath + '/' + xLabel + '-' + yLabel + '.png')

        if displayParams['showPlot']:
            plt.show()

        plt.close() #todo : check this