import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt

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


    def get_data(self, input_df):
        """Builds labels (0, 1) or % depending on predictor type as a column on the dataframe, return np arrays for X and y"""
        df = input_df.copy()
        token = 'BTC'

        df = df[np.isfinite(df).all(1)]     # keep only finite numbers
        # print(df.isnull().values.any())

        # These do time differencing for data stationarity
        price_name = token + 'Close'
        vol_name = token + 'Volume'
        df = percent_change_column(price_name, df)
        df = percent_change_column(vol_name, df)
        
        if self.is_classifier:
            thresh = 0.001
            #Calculate what the next price will be
            all_labels = np.empty([df.shape[0],self.n_predictions])
            y_name = (f'Period {str(self.n_predictions)} Mean % Change')
            for i in range(self.n_predictions):
                step = (i+1)
                series = df['BTCClose'].shift(-step)
                
                all_labels = np.hstack((all_labels, np.atleast_2d(series).T)) 

            # Take the mean accross the prices
            summed_labels = np.sum(all_labels, axis = 1)

            labels = np.empty_like(summed_labels)
            for j, item in enumerate(summed_labels):
                if item > thresh:
                    labels[j] = 1
                else:
                    labels[j] = 0
            df[y_name] = labels
        else:
            #Calculate what the next price will be
            for i in range(self.n_predictions):
                step = (i+1)  # *-1
                df, y_name = percent_change_column('BTCClose', df, -step)

        # print('Training data:')
        # print(df.head(30))
        features_to_use = list(df.columns)
        # Remove the y labels from the 'features to use'
        features_to_use.remove(y_name)
            
        # Make sure you aren't building a classifier based on the outputs of other classifiers
        classifier_names = ['knn', 'mlp', 'svc', 'ridge']
        for item in features_to_use:
            for name in classifier_names:
                if name in item:
                    features_to_use.remove(item)

        df = df[np.isfinite(df).all(1)]     # keep only finite numbers
        X = df[features_to_use].values
        y = df[y_name].values
        return X, y


    def build_feature(self, input_df):
        df = input_df.copy()
        df = df[np.isfinite(df).all(1)]     # keep only finite numbers
        X = df.values
        try:
            y_predict = self.model.predict(X)
        except ValueError:
            print('Prediction not properly trained. Training on test data set. Recommended retrain.')
            self.optimize_parameters(df)
            y_predict = self.model.predict(X)
        # print(f'y_shape {y_predict.shape}')
        df['BTC' + self.name] = y_predict
        return df


    def optimize_parameters(self, input_df):
        """ Performs a grid search over all of the hyperparameters"""
        X, y = self.get_data(input_df)
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
        X, y = self.get_data(input_df)

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
