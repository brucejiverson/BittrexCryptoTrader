
# Importing from other modules in the package
from features.predictor import Predictor
from features.label_data import *
from tools.tools import f_paths, percent_change_column

# Miscellaneous
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt

# Data handling
import pickle
import pandas as pd
import numpy as np

# sklearn stuff
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


class knn_classifier(Predictor):
    """Constructs a knn classifier from the df passed. Hyperparameters should be a 
    dictionary, parameters are k, polynomial degrees. Note that this calculates 
    percentage change, not simple time differencing, as it makes the threshold more intuitive. 
    (temporarily changed as it was causing problems)"""

    def __init__(self, token, hyperparams, label_type='standard'):
        super().__init__(hyperparams)
        self.name = 'knn'
        self.path = f_paths['feature_models'] + '/' + self.name + '.pkl'
        self.config_path = f_paths['feature_models'] + '/' + self.name + '_config.pkl'
        print(f"Hyperparams: {hyperparams}")
        self.model = None
        self.build_pipeline()


    def build_pipeline(self):
        scaler = StandardScaler()
        degree = self.hyperparams['polynomial_features__degree'][0]
        polynomial_features = PolynomialFeatures(degree=degree)

        k = self.hyperparams['knn__n_neighbors'][0]
        knn = KNeighborsClassifier(n_neighbors=k)

        # The pipeline can be used as any other estimator
        # and avoids leaking the test set into the train set
        self.model = Pipeline([
                        ('scaler', scaler),
                        ('polynomial_features', polynomial_features),
                        ('knn', knn)])


    def plot_results(self):
        """Currently this is not built/configured"""
        pass
        # if plot == True:
        # for i in range(self.n_predictions):  # Loop over each time step in the future
        #     fig = plt.figure()
        #     ax1 = fig.add_subplot(111)
        #     ax1.set_ylabel('Real Value')
        #     ax1.set_xlabel('Predicted Value ' +
        #                    str(i + 1) + ' timesteps forwards')
        #     ax1.set_title(
        #         'Performance of Linear Regression Model On Test Data')
        #     if split == False:
        #         plt.scatter(y_train[:, i], y_train_predict[:, i])
        #     else:
        #         plt.scastter(y_test[:, i], y_test_predict[:, i])
        #         # df['']

        # plt.show()


class logistic_regression_classifier(Predictor):
    """Constructs a knn classifier from the df passed. Hyperparameters should be a 
    dictionary, parameters are k, polynomial degrees. Note that this calculates 
    percentage change, not simple time differencing, as it makes the threshold more intuitive."""

    def __init__(self, token, hyperparams, label_type = 'standard'):
        super().__init__(hyperparams)
        self.name = 'logistic regression'
        self.path = f_paths['feature_models'] + '/' + self.name + '.pkl'
        self.config_path = f_paths['feature_models'] + '/' + self.name + '_config.pkl'
        print(f"Hyperparams: {hyperparams}")

        scaler = StandardScaler()
        degree = hyperparams['polynomial_features__degree'][0]
        polynomial_features = PolynomialFeatures(degree=degree)

        reg = LogisticRegression(random_state=0)

        # The pipeline can be used as any other estimator
        # and avoids leaking the test set into the train set
        self.model = Pipeline([
                        ('scaler', scaler),
                        ('polynomial_features', polynomial_features),
                        ('logistic regression', reg)])


    def plot_results(self):
        """Currently this is not built/configured"""
        pass
        # if plot == True:
        # for i in range(self.n_predictions):  # Loop over each time step in the future
        #     fig = plt.figure()
        #     ax1 = fig.add_subplot(111)
        #     ax1.set_ylabel('Real Value')
        #     ax1.set_xlabel('Predicted Value ' +
        #                    str(i + 1) + ' timesteps forwards')
        #     ax1.set_title(
        #         'Performance of Linear Regression Model On Test Data')
        #     if split == False:
        #         plt.scatter(y_train[:, i], y_train_predict[:, i])
        #     else:
        #         plt.scastter(y_test[:, i], y_test_predict[:, i])
        #         # df['']

        # plt.show()


class mlp_classifier(Predictor):
    """Constructs a knn classifier from the df passed. Hyperparameters should be a 
    dictionary, parameters are k, polynomial degrees. Note that this calculates 
    percentage change, not simple time differencing, as it makes the threshold more intuitive. 
    (temporarily changed as it was causing problems)"""

    def __init__(self, token, hyperparams):
        super().__init__(hyperparams)
        self.name = 'mlp'
        self.path = f_paths['feature_models'] + '/' + self.name + '.pkl'
        self.config_path = f_paths['feature_models'] + '/' + self.name + '_config.pkl'
        print(f"Hyperparams: {hyperparams}")

        scaler = StandardScaler()
        degree = hyperparams['polynomial_features__degree'][2]
        polynomial_features = PolynomialFeatures(degree=degree)

        mlp = MLPClassifier(alpha=1, max_iter=10000)

        # The pipeline can be used as any other estimator
        # and avoids leaking the test set into the train set
        self.model = Pipeline([
                        ('scaler', scaler),
                        ('polynomial_features', polynomial_features),
                        ('mlp', mlp)])


class svc_classifier(Predictor):
    """Constructs a knn classifier from the df passed. Hyperparameters should be a 
    dictionary, parameters are k, polynomial degrees. Note that this calculates 
    percentage change, not simple time differencing, as it makes the threshold more intuitive. 
    (temporarily changed as it was causing problems)"""

    def __init__(self, token, hyperparams):
        super().__init__(hyperparams)
        self.name = 'svc'
        self.path = f_paths['feature_models'] + '/' + self.name + '.pkl'
        self.config_path = f_paths['feature_models'] + '/' + self.name + '_config.pkl'
        print(f"Hyperparams: {hyperparams}")

        scaler = StandardScaler()
        polynomial_features = PolynomialFeatures(degree=4)
        svc = LinearSVC(penalty='l2',
                        C=1,
                        dual=False,
                        max_iter=50000)

        # The pipeline can be used as any other estimator
        # and avoids leaking the test set into the train set
        self.model = Pipeline([
                        ('scaler', scaler),
                        ('polynomial_features', polynomial_features),
                        ('svc', svc)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,)
    args = parser.parse_args()
    name = args.name
    assert(name in ['knn', 'svc', 'mlp', 'reg'])
    from environments.environments import SimulatedCryptoExchange
    #date range to train on
    start = datetime(2020, 1, 1)
    end = datetime(2020, 9, 1) #- timedelta(days = 1)
    features = {  # 'sign': ['Close', 'Volume'],
        # 'EMA': [50, 80, 130],
        'OBV': [],
        'RSI': [],
        # 'high': [],
        # 'low': [],
        'BollingerBands': [3],
        'BBInd': [],
        'BBWidth': [],
        'discrete_derivative': ['BBWidth3'], #, 'BBWidth4', 'BBWidth5'],
        # 'time of day': [],
        # 'stack': [2],
        'rolling probability': ['BBInd3', 'BBWidth3']
        # 'probability': ['BBInd3', 'BBWidth3']
        }
    sim_env = SimulatedCryptoExchange(granularity=5, feature_dict=features, train_test_split=1/16)

    df = sim_env.df.copy()              # This is the dataframe with all of the features built.

    token = 'BTC'
    if name == 'knn':
        # Hyperparameters
        hyperparams = {'polynomial_features__degree': (1, 2), 
                        'knn__n_neighbors': (3, 4)}
        buy_knn = knn_classifier(token, hyperparams)
        df = make_criteria(df, buy_not_sell=True)
        # print('DF WITH BUY LABELS:')
        # print(df.head())
        print('Fitting buy data.')
        buy_knn.optimize_parameters(df, 'Buy Labels')
        df = buy_knn.build_feature(df, 'Buy', ignore_names=['Buy Labels'])
        print('Refitting on full dataset with buy knn.')
        buy_knn.build_pipeline()    # Explicitly reset the model to an empty model
        buy_knn.train(df, 'Buy Labels')
        df.drop(columns='Buy Labels', inplace=True)
        
        print('Fitting sell data.')
        df = make_criteria(df, buy_not_sell=False)
        sell_knn = knn_classifier(token, hyperparams)
        sell_knn.optimize_parameters(df, 'Sell Labels', ignore_names=['BTCknnknn Buy'])
        df = sell_knn.build_feature(df, 'Sell', ignore_names=['Sell Labels', 'BTCknnknn Buy'])
        print('Refitting on full dataset with sell knn.')
        sell_knn.build_pipeline()
        sell_knn.train(df, 'Sell Labels')
        df.drop(columns='Sell Labels', inplace=True)

        # knn.load()
        print(df.head())
        # knn.save()
    elif name == 'reg':
        # Hyperparameters
        hyperparams = {'polynomial_features__degree': (1, 2, 3),
                        }
        clf = logistic_regression_classifier(token, hyperparams, n_predictions=5)
        clf.optimize_parameters(df)
        df = clf.build_feature(df)
        print(df.head(100))
        clf.save()
    elif name == 'svc':
        hyperparams = {'polynomial_features__degree': (1, 2, 3, 4),
                        'svc__penalty': ['l1', 'l2'],
                        'svc__C': [.5, .75, 1],
                        # 'svc__max_iter': [50000]
                        }    
                        # strength of regularization is inversely proportional to C
        svc = svc_classifier(token, hyperparams)
        # svc.load()
        svc.optimize_parameters(df)
        # svc.train(df)
        df = svc.build_feature(df)
        # print(df.head())
        svc.save()
    elif name == 'mlp':
        hyperparams = {'polynomial_features__degree': (1, 2, 3, 4)}
        mlp = mlp_classifier(token, hyperparams)
        # mlp.load()
        mlp.optimize_parameters(df)
        # mlp.train(df)
        df = mlp.build_feature(df)
        # print(df.head())
        mlp.save()
