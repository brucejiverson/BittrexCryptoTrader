import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt

from features.label_data import *
from tools.tools import maybe_make_dir, percent_change_column
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

class Predictor():
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.is_classifier = True
        self.name = None
        self.path = None
        self.config_path = None
        self.model = None
        

    def build_feature(self, input_df, prediction_name, ignore_names=None):
        
        # ignore name is usually a label that has not been purged yet
        df = input_df.copy()
        # df = df[np.isfinite(df).all(1)]     # keep only finite numbers. this should really be handled somewhere else

        
        x_cols_to_use = list(df.columns)
        if ignore_names is None:
            X = df.values
        else:
            for item in ignore_names:
                # Catch values values that are passed that are not in the columns of the data frame
                try:
                    x_cols_to_use.remove(item)
                except ValueError:
                    print(item)
                    pass
            X = df.loc[:, x_cols_to_use].values

        print(f'Building {self.name} on {len(x_cols_to_use)} features.')
        print(f'Features: {df.loc[:, x_cols_to_use].columns}')

        try:
            y_predict = self.model.predict(X)
        except ValueError:
            print('Prediction not properly trained. Training on test data set. Recommended retrain.')
            raise(ValueError)
            # print(self.model.get_params().keys())
            # self.optimize_parameters(df, )
            # y_predict = self.model.predict(X)
        # print(f'y_shape {y_predict.shape}')
        # print('HEY MY NAME IS ' + self.name)
        df['BTC' + self.name + ' ' + prediction_name] = y_predict
        # print(y_predict[0:100])
        return df


    def optimize_parameters(self, input_df, y_name, ignore_names=None):
        """ Performs a grid search over all of the hyperparameters"""
        df = input_df.copy()

        if ignore_names is None:
            X = df.loc[:, df.columns != y_name].values
        else:
            # Filter out the unwanted columns
            x_cols_to_use = list(df.columns)
            for item in [y_name, *ignore_names]:
                try:
                    x_cols_to_use.remove(item)
                except ValueError:
                    print(item)
                    pass
            # Get the data using the filtered column names
            X = df.loc[:, x_cols_to_use].values

        y = df.loc[:, y_name].values

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        gs = GridSearchCV(self.model, 
                        param_grid=self.hyperparams,
                        cv = 10,
                        verbose = 1, 
                        n_jobs = -1)
        # print(self.model)
        # print(gs)
        # Fitting model CV GS
        gs.fit(X_train, y_train)
        score = gs.score(X_train, y_train)
        print(f'Score on train data: {score}')
        score = gs.score(X_test, y_test)
        print(f'Score on test data: {score}')
        print(gs.best_estimator_)
        self.model = gs
        self.save()
    

    def train(self, input_df, y_name, ignore_names=None):
        """Train on all data"""
        df = input_df.copy()
        
        x_cols_to_use = list(df.columns)
        x_cols_to_use.remove(y_name)
        if ignore_names is None:
            X = df.loc[:, df.columns != y_name].values
        else:
            # Filter out the unwanted columns
            for item in ignore_names:
                try:
                    x_cols_to_use.remove(item)
                except ValueError:
                    print(item)
                    pass
            X = df.loc[:, x_cols_to_use].values
        
        y = df.loc[:, y_name].values

        n_cols = X.shape[1]
        
        print(f'Training {self.name} classifier on {n_cols} features...')
        print(f'Features: {df.loc[:, x_cols_to_use].columns}')

        self.model.fit(X, y)

        s = self.model.score(X, y)
        print(f"{self.name} score on all data after training: {round(s, 3)}.")


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
