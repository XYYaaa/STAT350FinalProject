import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import difflib
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LinearRegression as LR,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from math import log, sqrt, exp
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from random import randrange
from sklearn.neural_network import MLPRegressor
import pickle
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import time
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.fftpack import fft,ifft, fftfreq, fftshift
from scipy import signal
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
import random
from itertools import combinations
from sklearn.impute import KNNImputer
from missingpy import MissForest
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from catboost import Pool, CatBoostRegressor
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import re
import seaborn
seaborn.set()


imp = ['southwest', 'northeast', 'southeast', 'smoker', 'children',
       'bmi', 'age']
df = pd.read_csv('trans_insurance.csv')

def get_imp(X, model):
    features = X.columns
    importances = model.feature_importances_
    indices = np.argsort(importances[0:30])
    '''
    for i in indices:
        print(features[i])
    '''
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    #plt.savefig("plot/imp/Feature Importances", dpi=500)
    plt.show()
    plt.clf()

def corr_p(x, y):
    plt.plot(df[x], df[y], 'b.')
    #plt.xlim(-1, 2)
    #plt.ylim(-1, 2)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Correlation')
    #plt.savefig("plot/corr/Correlation-" + str(x), dpi=500)
    plt.show()
    plt.clf()
    #print('corr:', np.corrcoef(x, y)[1,0])

def model_test(X, y):
    kernel = DotProduct() + 1 * RBF(1.0)
    rs = random.randint(0, 100000000)
    rm = random.randint(0, 100000000)

    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l2', 'mae'],
        'learning_rate': 0.03,
        "max_depth": 20,
        "num_leaves": 10,
        "n_estimators": 100,
    }

    model = make_pipeline(
        #linear_model.LinearRegression()
        Ridge(alpha = 0.1)
        #CatBoostRegressor(learning_rate=0.01, depth=10, verbose=False)
        #lgb.LGBMRegressor(**hyper_params)
    )

    #model = CatBoostRegressor(learning_rate=0.01, depth=10, verbose=False)
    #model = linear_model.LinearRegression(fit_intercept=False)

    '''
    model0 = sm.OLS(y, X)
    results = model0.fit()
    print(results.summary())
    '''

    model = lgb.LGBMRegressor(**hyper_params)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print('CV MAE:', cv_scores)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=rs)
    #model.fit(X_train, y_train)
    model.fit(X, y)
    #print('Reg:', model.coef_, model.intercept_)
    get_imp(X, model)

    prev_y = model.predict(X_valid)
    pret_y = model.predict(X_train)
    val_mae = mean_absolute_error(prev_y, y_valid)
    train_mae = mean_absolute_error(pret_y, y_train)
    print(val_mae, mean_absolute_error(np.ones(y_valid.size) * np.median(y_valid), y_valid))
    print(train_mae, mean_absolute_error(np.ones(y_train.size) * np.median(y_train), y_train))

    train_score = model.score(X_train, y_train)
    valid_score = model.score(X_valid, y_valid)
    sdiff = np.absolute(train_score - valid_score)
    print('Score:', train_score, valid_score, sdiff, rm, rs)
    #joblib.dump(model, 'XGB/' + str(col) + '_CGB_' + str(train_score) + '.pkl')


def main():

    X = df.loc[:, (df.columns != 'charges')]
    #X = X[imp]
    #X = np.log(X+1)
    y = df['charges']


    for i in range(10):
        print(i)
        model_test(X, y)


    '''
    for col in X.columns:
        corr_p(col, 'charges')
    '''


    '''
    plt.hist(df['bmi'], log=True)
    print(df['bmi'].mean(0))
    print(df[df['bmi'] > 30.663396860986538].count())
    print(df[df['bmi'] < 30.663396860986538].count())
    #plt.hist(df['age'])
    plt.show()
    '''

if __name__=='__main__':
    main()