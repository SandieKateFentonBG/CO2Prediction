def saveStudy(displayParams, Content):

    if displayParams['archive']:
        import os
        outputPathStudy = displayParams["outputPath"] + displayParams["reference"]
        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy + '/Records' ".txt", 'a') as f:
            print('', file=f)
            if type(Content) == dict:
                for k,v in Content.items():
                    print(k, ":", v, file=f)
            else:
                for r in Content:
                    print(r, file=f)

        f.close()

def printStudy(displayParams, Content):

    if displayParams['showResults']:
        for m in Content:
            for k, v in m.items():
                print(k, ':', v)

def exportStudy(displayParams, inputData, prepData, modelsData):

    if displayParams['archive']:

        import os
        outputPathStudy = displayParams["outputPath"] + displayParams["reference"]
        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        import csv
        with open(outputPathStudy + '/Records' + ".csv", 'w', encoding='UTF8', newline='') as e:
            writer = csv.writer(e, delimiter = ";")

            writer.writerow('INPUT DATA')
            for inputk, inputv in inputData.items():
                writer.writerow([inputk, inputv])
            writer.writerow('')

            writer.writerow('PREPROCESSED DATA')
            for inputk, inputv in prepData.items():
                writer.writerow([inputk, inputv])
            writer.writerow('')

            writer.writerow('MODELS DATA')
            k = modelsData[0].keys()
            writer.writerow(k)
            for i in range(len(modelsData)):
                v = modelsData[i].values()
                writer.writerow(v)
        e.close()

def saveInput(csvPath, outputPath, displayParams, xQualLabels, xQuantLabels, yLabels, processingParams, modelingParams, powers, mixVariables   ):

    Content = dict()
    Content['csvPath'] = csvPath
    Content['outputPath'] = outputPath
    Content['displayParams'] = displayParams
    Content['xQualLabels'] = xQualLabels
    Content['xQuantLabels'] = xQuantLabels
    Content['yLabels'] = yLabels
    Content['processingParams'] = processingParams
    Content['modelingParams'] = modelingParams
    Content['powers'] = powers
    Content['mixVariables'] = mixVariables

    saveStudy(displayParams, Content)

    return Content