from Dashboard_EUCB_FR_v2 import *




# https://scikit-learn.org/stable/modules/cross_validation.html7https://scikit-learn.org/stable/modules/cross_validation.html

# Maybe leave one aout?

def Stack_Learning_Data(modelList, type = 'XVal'):

    # create meta learning data
    blend_elem_sets = []

    for model in modelList:
        XVal = model.learningDf.__getattribute__(type).to_numpy()
        blend_elem_i = model.Estimator.predict(XVal)
        blend_elem_i = pd.DataFrame(blend_elem_i)
        blend_elem_sets.append(blend_elem_i)
        #data is already scaled

    blendDf = pd.concat(blend_elem_sets, axis=1)

    return blendDf


class Model_Stacker:

    def __init__(self, modelList, blendingConstructor, Gridsearch = True, Type ='NBest'):

        self.modelList = modelList
        self.GSName = blendingConstructor['name'] + '_Blender_' + Type
        self.Type = Type
        self.predictorName = blendingConstructor['name']  # ex : SVR
        self.modelPredictor = blendingConstructor['modelPredictor']
        self.param_dict = blendingConstructor['param_dict']
        self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        self.refit = 'r2'  # Score used for refitting the blender
        self.accuracyTol = 0.15
        self.rounding = 3

        # xVal and xCheck : same samples for every seeds and models, but different features depending on learningDf
        self.blendXtrain = Blend_Learning_Data(modelList, type='XVal')
        self.blendXtest = Blend_Learning_Data(modelList, type='XCheck')

        #todo: check this
        self.ScaleMean = self.blendXtrain.mean(axis=0)
        self.ScaleStd = self.blendXtrain.std(axis=0)

        self.blendXtrain = (self.blendXtrain - self.ScaleMean) / self.ScaleStd
        self.blendXtest = (self.blendXtest - self.ScaleMean) / self.ScaleStd
        #todo: check this

        # yVal, yCheck are identical for all modls and seed > fixed seed
        self.yTrain = modelList[0].learningDf.__getattribute__('yVal').to_numpy().ravel()
        self.yTest = modelList[0].learningDf.__getattribute__('yCheck').to_numpy().ravel()

        xtrainer, ytrainer = self.blendXtrain, self.yTrain

        # building the final model using the meta features # this should be done by a cv of 5 folds on the training set
        if Gridsearch:
            njobs = os.cpu_count() - 1
            grid = GridSearchCV(self.modelPredictor, param_grid=self.param_dict, scoring=self.scoring, refit=self.refit,
                                n_jobs=njobs, return_train_score=True)
            grid.fit(xtrainer, ytrainer)
            self.Param = grid.best_params_
            self.Estimator = grid.best_estimator_

        else:
            self.Estimator = self.modelPredictor.fit(xtrainer, ytrainer)
            self.Param = None

        self.yPred = self.Estimator.predict(self.blendXtest)
        self.TrainScore = round(self.Estimator.score(xtrainer, ytrainer), self.rounding)
        self.TestScore = round(self.Estimator.score(self.blendXtest, self.yTest), self.rounding)
        self.TestAcc = round(computeAccuracy(self.yTest, self.yPred, self.accuracyTol), self.rounding)
        self.TestMSE = round(mean_squared_error(self.yTest, self.yPred), self.rounding)
        self.TestR2 = round(r2_score(self.yTest, self.yPred), self.rounding)
        self.Resid = self.yTest - self.yPred

        self.ResidMean = round(np.mean(np.abs(self.Resid)), 2)  # round(np.mean(self.Resid),2)
        self.ResidVariance = round(np.var(self.Resid), 2)



        if hasattr(self.Estimator, 'coef_'):  # LR, RIDGE, ELASTICNET, KRR Kernel Linear, SVR Kernel Linear
            self.isLinear = True
            content = self.Estimator.coef_
            if type(content[0]) == np.ndarray:
                content = content[0]

        elif hasattr(self.Estimator, 'dual_coef_'):  # KRR
            self.isLinear = True
            content = self.Estimator.dual_coef_
            if type(content[0]) == np.ndarray:
                content = content[0]
        else:
            self.isLinear = False
            content = 'Estimator is non linear - no weights can be querried'

        weights = [round(num, self.rounding) for num in list(content)]
        self.ModelWeights = weights