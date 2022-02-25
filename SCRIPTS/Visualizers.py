import Visualizers as dx
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

"""
------------------------------------------------------------------------------------------------------------------------
DALEX
------------------------------------------------------------------------------------------------------------------------
"""

#https://dalex.drwhy.ai/python/api/model_explanations/index.html
#https://dalex.drwhy.ai/python-dalex-new.html
#https://wandb.ai/quickstart/pytorch

def predict_function(model, data):
    return np.exp(model.predict(data))

def dada(X, y):

    ylog = np.log(y)

    model_svm = Pipeline(steps=[('scale', StandardScaler()),
                                ('model', SVR(C=10, epsilon=0.2, tol=1e-4))])
    model_svm.fit(X, ylog)

    model_gbm = LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.15, random_state=0)
    model_gbm.fit(X, ylog)

    # exp_svm = dx.Explainer(model_svm, data=X, y=y, predict_function=predict_function, label='svm')
    exp_gbm = dx.Explainer(model_gbm, data=X, y=y, predict_function=predict_function, label='gbm')

    pd.concat((exp_svm.model_performance().result, exp_gbm.model_performance().result))
    mp = exp_gbm.model_parts(type='shap_wrapper', shap_explainer_type="TreeExplainer")
    # mp = exp_svm.model_parts(type='shap_wrapper', shap_explainer_type="TreeExplainer")

    mp.plot()
    mp.plot(plot_type='bar')

def dado(X, y, model):

    model.fit(X, y)

    # model_gbm = LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.15, random_state=0)
    # model_gbm.fit(X, ylog)

    exp_svm = dx.Explainer(model, data=X, y=y, predict_function=predict_function, label='svm')
    # exp_gbm = dx.Explainer(model_gbm, data=X, y=y, predict_function=predict_function, label='gbm')

    # pd.concat((exp_svm.model_performance().result, exp_gbm.model_performance().result))
    # mp = exp_gbm.model_parts(type='shap_wrapper', shap_explainer_type="TreeExplainer")
    mp = exp_svm.model_parts(type='shap_wrapper', shap_explainer_type="TreeExplainer")

    mp.plot()
    mp.plot(plot_type='bar')

"""
------------------------------------------------------------------------------------------------------------------------
WANDB
------------------------------------------------------------------------------------------------------------------------
"""
# https://colab.research.google.com/github/wandb/examples/blob/master/colabs/scikit/Simple_Scikit_Integration.ipynb#scrollTo=o3Rrp0-gctqc

#https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb#scrollTo=SyepLNEoBN9U

import wandb

def wandbtest(model, xTrain, xTest, yTrain, yTest):

    wandb.init()
    # wandb.login(key='a718e033cef0f91ed97a1ff015f2f4a1e644d11d')

    wandb.sklearn.plot_residuals(model, xTrain, yTrain)
    wandb.sklearn.plot_summary_metrics(model, xTrain, xTest, yTrain, yTest)

    wandb.sklearn.plot_regressor(model, xTrain, xTest, yTrain, yTest, model_name='Ridge')

    wandb.finish()