class CV_Blender:

    def __init__(self, modelList, blendingConstructor, baseFormatedDf):

        self.modelList = modelList
        self.GSName = blendingConstructor['name'] + '_Blender'
        self.predictorName = blendingConstructor['name']  # ex : SVR
        self.modelPredictor = blendingConstructor['modelPredictor']
        self.param_dict = blendingConstructor['param_dict']
        self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        self.refit = 'r2'  # Score used for refitting the blender
        self.accuracyTol = 0.15
        self.rounding = 3

        self.yTrain, self.yTest= baseFormatedDf.yTrain, baseFormatedDf.yTest
        self.XTrain, self.XTest = baseFormatedDf.XTrain, baseFormatedDf.XTest

        #create meta learning data
        blend_train_sets = []
        blend_test_sets = []

        for model in modelList:

            predictor = model.Estimator
            learningDf = model.learningDf

