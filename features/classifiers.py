import numpy as np
from features.predictor import Predictor
from tools.tools import f_paths, percent_change_column
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


class knn_classifier(Predictor):
    """Constructs a knn classifier from the df passed. Hyperparameters should be a 
    dictionary, parameters are k, polynomial degrees. Note that this calculates 
    percentage change, not simple time differencing, as it makes the threshold more intuitive. 
    (temporarily changed as it was causing problems)"""

    def __init__(self, token, hyperparams, n_predictions=1):
        super().__init__(hyperparams, n_predictions)
        self.name = 'knn'
        self.path = f_paths['feature_models'] + '/' + self.name + '.pkl'
        self.config_path = f_paths['feature_models'] + '/' + self.name + '_config.pkl'
        print(f"Hyperparams: {hyperparams}")

        scaler = StandardScaler()
        degree = hyperparams['polynomial_features__degree'][0]
        polynomial_features = PolynomialFeatures(degree=degree)

        k = hyperparams['knn__n_neighbors'][0]
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


class mlp_classifier(Predictor):
    """Constructs a knn classifier from the df passed. Hyperparameters should be a 
    dictionary, parameters are k, polynomial degrees. Note that this calculates 
    percentage change, not simple time differencing, as it makes the threshold more intuitive. 
    (temporarily changed as it was causing problems)"""

    def __init__(self, token, hyperparams, n_predictions=1):
        super().__init__(hyperparams, n_predictions)
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

    def __init__(self, token, hyperparams, n_predictions=1):
        super().__init__(hyperparams, n_predictions)
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
    
    from environments.environments import SimulatedCryptoExchange
    #date range to train on
    end = datetime(2020,7,2) #- timedelta(days = 1)
    start = end - timedelta(days = 12)
    features = {  # 'sign': ['Close', 'Volume'],
        'EMA': [50, 80, 130],
        'OBV': [],
        'high': [],
        'low': [],
        'BollingerBands': [],
        'time of day': [],
        # 'stack': [0]
        }
    sim_env = SimulatedCryptoExchange(granularity=120, feature_dict=features)

    df = sim_env.df.copy()              # This is the dataframe with all of the features built.

    print('Testing all classifiers...')
    token = 'BTC'
    # Hyperparameters
    hyperparams = {'polynomial_features__degree': (1, 2, 3, 4), 
                    'knn__n_neighbors': (2, 3, 4, 5, 6, 7)}
    knn = knn_classifier(token, hyperparams)
    # knn.load()
    knn.optimize_parameters(df)
    # knn.train(df)
    df = knn.build_feature(df)
    df, new_name = percent_change_column('BTCClose', df, -1)
    # print(df.head(40))
    knn.save()

    # hyperparams = {'polynomial_features__degree': (1, 2, 3, 4),
    #                 'svc__penalty': ['l1', 'l2'],
    #                 'svc__C': [.5, .75, 1],
    #                 # 'svc__max_iter': [50000]
    #                 }    
    #                 # strength of regularization is inversely proportional to C
    # svc = svc_classifier(token, hyperparams)
    # # svc.load()
    # svc.optimize_parameters(df)
    # # svc.train(df)
    # df = svc.build_feature(df)
    # # print(df.head())
    # svc.save()

    hyperparams = {'polynomial_features__degree': (1, 2, 3, 4)}
    mlp = mlp_classifier(token, hyperparams)
    # mlp.load()
    mlp.optimize_parameters(df)
    # mlp.train(df)
    df = mlp.build_feature(df)
    # print(df.head())
    mlp.save()
