# MODEL
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import numpy as np

"RANGE VALUES "
coef0_range =  list(10.0 ** np.arange(-2, 2))
regul_range = list(10.0 ** np.arange(-4, 4))
influence_range = list(10.0 ** np.arange(-4, 4))
degree = [2, 3, 4]
margin_range = list(10.0 ** np.arange(-4, 4))
kernel_list = ['linear', 'polynomial', 'rbf']


"PARAMETER DICTIONNARIES"
LR_param_grid={'alpha': regul_range}
KRR_param_grid={'alpha': regul_range, 'gamma': influence_range, 'degree' : degree, 'kernel' : kernel_list, 'coef0' : coef0_range }
SVR_param_grid={'C': regul_range, 'gamma': influence_range, 'degree' : degree, 'epsilon':  margin_range, 'kernel': kernel_list, 'coef0' : coef0_range}

# Example for Single Hyperparameter plot
KRR_param_grid1={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['linear']}
KRR_param_grid2={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['polynomial']}
KRR_param_grid3={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['rbf']}


# "Comment" - if computing time is too high :
# dictionary grouping all kernel types computes unnecessary combinations
# change for a combination of dictionnaries - 1 dictionary per kernel type :
# KRR_lin={'alpha': regul_range, 'gamma': influence_range}
# KRR_poly={'alpha': regul_range, 'degree' : degree}
# KRR_param_grid = [KRR_lin, KRR_poly]

"""
________________________________________________________________________________________________________________________
MODEL PARAMETER DETAILS
------------------------------------------------------------------------------------------------------------------------

"""


"""
Linear Regression - Hyperparameters
1 - alpha - overfitting - higher lamda, less overfitting

"""
#LR_param_grid={'alpha': regul_range}
"""
REGRESSSION - LINEAR 
ERROR - Ordinary least squares Linear Regression
HYPERPARAM - /
"""

#LR = {'model': LinearRegression(), 'param' : None, 'Linear' : True}

"""
REGRESSSION - LINEAR 
ERROR - LASSO - L1 regularizer 
HYPERPARAM - Lamda 
"""
#LR_Lasso = {'model': Lasso(), 'param': LR_param_grid, 'Linear' : True}

"""
REGRESSSION - LINEAR 
ERROR - RIDGE - L2 regularizer (aka least squares) 
HYPERPARAM - Lamda 
"""
#LR_Ridge = {'model': Ridge(), 'param': LR_param_grid, 'Linear' : True}

"""
REGRESSSION - LINEAR 
ERROR - ELASTIC NET - combined L1 and L2 
HYPERPARAM - Lamda 
"""
#LR_ElasticNet = {'model': ElasticNet(), 'param': LR_param_grid, 'Linear' : True}

"""
 ! if not done :  scale your data before using these regularized linear regression methods. 
 Use StandardScaler or set ‘normalize’ in these estimators to ‘True (ex :  ElasticNet(normalize=True))
"""
"""
Kernel Regression - Hyperparameters
1 - alpha - regularization strength
 Regularization improves the conditioning of the problem and reduces the variance of the estimates. 
 Larger values specify stronger regularization - higher lamda, less overfitting
2 - gamma -  kernel coefficient
 how far the influence of a single training example reaches, low values meaning ‘far’ and high values meaning ‘close’. 
 seen as the inverse of the radius of influence of samples selected by the model as support vectors - 
 higher lamda, less overfitting
3 - degree - Degree of the polynomial kernel. Ignored by other kernels.
4 - coef0 - Zero coefficient for polynomial and sigmoid kernels.

"""
#KRR_param_grid={'alpha': regul_range, 'gamma': influence_range, 'degree' : degree} #, 'coef0' : coef0
"""
REGRESSSION - KERNEL 
ERROR - RIDGE - L2 regularizer (aka least squares)
KERNEL - LINEAR
HYPERPARAM -   alpha, gamma


"""

#KRR_Lin = {'model' : KernelRidge(kernel='linear'), 'param': KRR_param_grid, 'Linear' : False}
"""
REGRESSSION - KERNEL 
ERROR - RIDGE - L2 regularizer (aka least squares)
KERNEL - RBF
HYPERPARAM - alpha, gamma 
"""

#KRR_Rbf = {'model' : KernelRidge(kernel='rbf'), 'param': KRR_param_grid, 'Linear' : False}

"""
REGRESSSION - KERNEL 
ERROR - RIDGE - L2 regularizer (aka least squares)
KERNEL - POLYNOMIAL
HYPERPARAM - alpha, gamma, degree  
"""
#KRR_Pol = {'model' : KernelRidge(kernel='polynomial'), 'param': KRR_param_grid, 'Linear' : False}

"""
Support Vector Regression - Hyperparameters

1 - degree - Degree of the polynomial kernel. Ignored by other kernels.
2 - gamma - kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
 how far the influence of a single training example reaches, low values meaning ‘far’ and high values meaning ‘close’. 
 seen as the inverse of the radius of influence of samples selected by the model as support vectors - higher lamda, less overfitting
3 - coef0 - Zero coefficient for polynomial and sigmoid kernels. Ignored by other kernels.
4 - C - Regularization parameter. 
Trades off correct prediction on training examples against maximization of the decision function’s margin
 Must be strictly positive. The penalty is a squared l2 penalty.
The strength of the regularization is inversely proportional to C  Lower C , larger margin, lower theta, less overfitting;
5 - epsilon - Margin size - specifies the epsilon-tube within which no penalty is associated in the training loss function 
with points predicted within a distance epsilon from the actual value.
"""

#SVR_param_grid={'C': regul_range, 'gamma': influence_range, 'degree' : degree, 'epsilon' :  margin_range} #, 'coef0' : coef0
"""
REGRESSSION - SUPPORT VECTOR 
ERROR - ?
KERNEL - LINEAR
HYPERPARAM - C, Lamda , epsilon
"""

#SVR_Lin = {'model' : SVR(kernel='linear'), 'param': SVR_param_grid, 'Linear' : True}

"""
REGRESSSION - SUPPORT VECTOR 
ERROR - ?
KERNEL - RBF
HYPERPARAM - C, gamma, epsilon
"""
#SVR_Rbf = {'model' : SVR(kernel='rbf'), 'param': SVR_param_grid, 'Linear' : False}

"""
REGRESSSION - SUPPORT VECTOR 
ERROR - ?
KERNEL - POLYNOMIAL
HYPERPARAM - C, gamma, degree, epsilon
"""
#SVR_Pol = {'model' : SVR(kernel='poly'), 'param': SVR_param_grid, 'Linear' : False}

#predictors = [LR, LR_Lasso, LR_Ridge, LR_ElasticNet,KRR_Lin, KRR_Rbf, KRR_Pol,SVR_Lin, SVR_Rbf, SVR_Pol]

"""to keep
# modelingParams = {'test_size': 0.2, 'random_state' : random, 'RegulVal': list(10.0**np.arange(-4,4)), 'epsilonVal': list(10.0**np.arange(-4,4)),
#                   'accuracyTol': 0.15, 'CVFold': None, 'rankGridSearchModelsAccordingto' : 'r2', 'plotregulAccordingTo' : 'paramMeanMSETest'}
"""