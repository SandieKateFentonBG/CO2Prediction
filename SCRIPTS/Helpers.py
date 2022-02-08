def saveStudy(displayParams, Content):

    # import os
    # if not os.path.isdir(displayParams["outputPath"]):
    #     os.makedirs(displayParams["outputPath"])

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