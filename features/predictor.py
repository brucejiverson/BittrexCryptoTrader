import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt

from features.label_data import *
from tools.tools import maybe_make_dir, percent_change_column
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

class Predictor():
    def __init__(self, hyperparams, n_predictions=1):
        self.hyperparams = hyperparams
        self.n_predictions = n_predictions
        self.is_classifier = True
        self.name = None
        self.path = None
        self.config_path = None
        self.model = None
        

    def build_feature(self, input_df):
        df = input_df.copy()
        df = df[np.isfinite(df).all(1)]     # keep only finite numbers
        X = df.values
        try:
            y_predict = self.model.predict(X)
        except ValueError:
            print('Prediction not properly trained. Training on test data set. Recommended retrain.')
            print(self.model.get_params().keys())
            self.optimize_parameters(df)
            y_predict = self.model.predict(X)
        # print(f'y_shape {y_predict.shape}')
        df['BTC' + self.name] = y_predict
        # print(y_predict[0:100])
        return df


    def optimize_parameters(self, input_df):
        """ Performs a grid search over all of the hyperparameters"""
        df = make_criteria(input_df)
        X = df.loc[:, df.columns != 'Labels'].values
        y = df.loc[:, 'Labels'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        gs = GridSearchCV(self.model, 
                        param_grid=self.hyperparams,
                        cv = 10,
                        verbose = 1, 
                        n_jobs = -1)
        # Fitting model CV GS
        gs.fit(X_train, y_train.ravel())
        score = gs.score(X_train, y_train.ravel())
        print(f'Score on train data: {score}')
        score = gs.score(X_test, y_test.ravel())
        print(f'Score on test data: {score}')
        print(gs.best_estimator_)
        self.model = gs
        self.save()
    

    def train(self, input_df):
        """Train on all data"""
        df = make_criteria(input_df)
        X = df.loc[:, df.columns != 'Labels'].values
        y = df.loc[:, 'Labels'].values

        n_cols = X.shape[1]
        print(f'Training {self.name} classifier on {n_cols} features...')

        self.model.fit(X, y.ravel())

        score = self.model.score(X, y)
        print(f"{self.name} score on all data after training: {round(score, 3)}.")


    def save(self):
        maybe_make_dir('features/models')
        # Save to file and the config in the current working directory
        with open(self.config_path, 'wb') as f:
            pickle.dump(self.hyperparams, f, protocol=pickle.HIGHEST_PROTOCOL)
        joblib.dump(self.model, self.path)


    def load(self):
        # Load the config/hyperparams
        with open(self.config_path, 'rb') as f:
            self.hyperparams = pickle.load(f)
        # Load the model
        self.model = joblib.load(self.path)
