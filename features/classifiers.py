import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
# from sklearn.externals import joblib
from environments.environments import SimulatedCryptoExchange, f_paths


def knn_classifier(input_df, token, hyperparams, n_predictions=1, split=False, plot=False):
    """Constructs a knn classifier from the df passed. Hyperparameters should be a 
    dictionary, parameters are k, polynomial degrees. Note that this calculates 
    percentage change, not simple time differencing, as it makes the threshold more intuitive. 
    (temporarily changed as it was causing problems)"""

    df = input_df.copy()
    y_names = []

    # df.dropna(inplace=True)#fillna(method='ffill',inplace=True)
    # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)] # removes all nan, inf, -inf values
    df = df[np.isfinite(df).all(1)]     # keep only finite numbers
    # print(df.isnull().values.any())
    thresh = 0.005
    #Calculate what the next price will be
    for i in range(n_predictions):
        step = (i+1)#*-1
        y_names.append('Step ' + str(step) + ' % Change')
        series = (df['BTCClose'].shift(-step) - df['BTCClose'])#/df['BTCClose']
        labels = np.empty_like(series.values)
        for j, item in enumerate(series):
            if item > thresh:
                labels[j] = 1
            else: 
                labels[j] = 0
        # for item in :
        df[y_names[i]] = labels
        # df[y_names[i]] = df['BTCClose'].shift(-step) - df['BTCClose']     #/df['BTCClose']

    # These do time differencing for data stationarity
    # df['Last BTCClose'] = df['BTCClose'].shift(1) - df['BTCClose'].shift(2)
    price_name = token + 'Close'
    vol_name = token + 'Volume'
    df[price_name] = df[price_name] - df[price_name].shift(1, fill_value=0)#/df[price_name]
    df[vol_name] = df[vol_name] - df[vol_name].shift(1, fill_value=0)#/df[vol_name]
    
    # print(df.isnull().values.any())
    # df = df[np.isfinite(df).all(1)]     # keep only finite numbers
    
    features_to_use = list(df.columns) 
    # Remove the y labels from the 'features to use' 
    for i in range(n_predictions):
        features_to_use.remove(y_names[i])


    X = df[features_to_use].values
    y = df[y_names].values

    k = hyperparams['k'][0]
    degree = hyperparams['degrees'][2]
    polynomial_features = PolynomialFeatures(degree=degree)

    knn = KNeighborsClassifier(n_neighbors = k)

    # The pipeline can be used as any other estimator
    # and avoids leaking the test set into the train set
    pipe = Pipeline([('polynomial_features', polynomial_features), 
                    ('knn', knn)])

    n_rows, n_cols = X.shape
    print(f'Training knn classifier on {n_cols} features...')

    # This is for train/test splitting for model validation
    if split == True:
        # Split the data into testing and training dataframes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # , random_state=42)

        pipe.fit(X_train, y_train.ravel())

        score = pipe.score(X_train, y_train)
        print(f"KNN mean accuracy score on train data: {round(score, 3)}.")
        y_train_predict = pipe.predict(X_train)
        score = pipe.score(X_test, y_test)
        print(f"KNN mean accuracy score on test data: {round(score, 3)}.")
        y_test_predict = pipe.predict(X_test)
    # else:       # Don't split
    pipe.fit(X, y.ravel())

    score = pipe.score(X, y)
    print(f"KNN mean accuracy score on all data: {round(score, 3)}.")
    y_predict = pipe.predict(X)


    # for i in range(n_predictions):
    #     col = y_names[i]
    #     temp = testing[new_df[col] > thresh]                        #The real values for all predictions greate than 0
    #     n_pred_over_0, n_col = temp.shape
    #     n_success, n_col  = temp[temp[col] > thresh].shape

    #     print(f'Out of {n_pred_over_0} predictions the price will be over {thresh}, ', end = '')
    #     print(f'{100*round(n_success/n_pred_over_0,2)}% were above that threshhold.')
    
    # This section has not yet been configured. Not totally sure how to best show the data
    
    if plot == True:
        for i in range(n_predictions): #Loop over each time step in the future
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.set_ylabel('Real Value')
            ax1.set_xlabel('Predicted Value ' + str(i + 1) + ' timesteps forwards')
            ax1.set_title('Performance of Linear Regression Model On Test Data')
            if split == False:
                plt.scatter(y_train[:,i], y_train_predict[:,i])
            else:
                plt.scastter(y_test[:,i], y_test_predict[:,i])
                # df['']

        plt.show()
    # print(input_df.shape)
    # print(y_predict.shape)
    df[token + 'knn'] = y_predict
    return df


def mlp_classifier(input_df, token, hyperparams, n_predictions=1, split=False, plot=False):
    """Constructs a knn classifier from the df passed. Hyperparameters should be a 
    dictionary, parameters are k, polynomial degrees. Note that this calculates 
    percentage change, not simple time differencing, as it makes the threshold more intuitive."""
    df = input_df.copy()
    y_names = []

    thresh = 0.005
    #Calculate what the next price will be
    for i in range(n_predictions):
        step = (i+1)#*-1
        y_names.append('Step ' + str(step) + ' % Change')
        series = (df['BTCClose'].shift(-step) - df['BTCClose'])#/df['BTCClose']
        labels = np.empty_like(series.values)
        for j, item in enumerate(series):
            if item > thresh:
                labels[j] = 1
            else: 
                labels[j] = 0
        # for item in :
        df[y_names[i]] = labels
        # df[y_names[i]] = df['BTCClose'].shift(-step) - df['BTCClose']     #/df['BTCClose']

    # These do time differencing 
    # df['Last BTCClose'] = df['BTCClose'].shift(1) - df['BTCClose'].shift(2)
    price_name = token + 'Close'
    vol_name = token + 'Volume'
    df[price_name] = df[price_name] - df[price_name].shift(1, fill_value=0)#/df[price_name]
    df[vol_name] = df[vol_name] - df[vol_name].shift(1, fill_value=0) #/df[vol_name]

    df.dropna(inplace=True)#fillna(method='ffill',inplace=True)
    # print(df.isnull().values.any())
    # n_sample, n_feature = df.shape

    features_to_use = list(df.columns) # ['BTCClose', 'BTCVolume', 'BTCOBV', 'EMA_30', 'd/dt_EMA_30', 'EMA_78', 'd/dt_EMA_78']
    # Remove the y labels from the 'features to use' 
    for i in range(n_predictions):
        features_to_use.remove(y_names[i])

    X = df[features_to_use].values
    y = df[y_names].values

    degree = hyperparams['degrees'][2]
    polynomial_features = PolynomialFeatures(degree=degree)

    mlp = MLPClassifier(alpha=1, max_iter=10000)

    # The pipeline can be used as any other estimator
    # and avoids leaking the test set into the train set
    pipe = Pipeline([('polynomial_features', polynomial_features), 
                    ('mlp', mlp)])

    n_rows, n_cols = X.shape
    print(f'Training MLP classifier on {n_cols} features...')

    # This is for train/test splitting for model validation
    if split == True:
        # Split the data into testing and training dataframes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # , random_state=42)

        pipe.fit(X_train, y_train.ravel())

        score = pipe.score(X_train, y_train)
        print(f"MLP mean accuracy score on train data: {round(score,4)}.")
        y_train_predict = pipe.predict(X_train)
        score = pipe.score(X_test, y_test)
        print(f"MLP mean accuracy score on test data: {round(score, 4)}.")
        y_test_predict = pipe.predict(X_test)
    # else:       # Don't split

    pipe.fit(X, y.ravel())

    score = pipe.score(X, y)
    print(f"MLP mean accuracy score on all data: {round(score,4)}.")
    y_predict = pipe.predict(X)

    # for i in range(n_predictions):
    #     col = y_names[i]
    #     temp = testing[new_df[col] > thresh]                        #The real values for all predictions greate than 0
    #     n_pred_over_0, n_col = temp.shape
    #     n_success, n_col  = temp[temp[col] > thresh].shape

    #     print(f'Out of {n_pred_over_0} predictions the price will be over {thresh}, ', end = '')
    #     print(f'{100*round(n_success/n_pred_over_0,2)}% were above that threshhold.')
    
    # This section has not yet been configured. Not totally sure how to best show the data
    if plot == True:
        for i in range(n_predictions): #Loop over each time step in the future
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.set_ylabel('Real Value')
            ax1.set_xlabel('Predicted Value ' + str(i + 1) + ' timesteps forwards')
            ax1.set_title('Performance of Linear Regression Model On Test Data')
            if split == False:
                plt.scatter(y_train[:,i], y_train_predict[:,i])
            else:
                plt.scastter(y_test[:,i], y_test_predict[:,i])
                # df['']

        plt.show()
    
    print(input_df.shape)
    print(y_predict.shape)
    input_df[token + 'MLP'] = y_predict
    return df


if __name__ == '__main__':
    #date range to train on
    end = datetime.now() - timedelta(days = 2)
    start = end - timedelta(days = 9)
    features = {  # 'sign': ['Close', 'Volume'],
        'EMA': [50, 80, 130],
        'time of day': [],
        'stack': [1]}
    sim_env = SimulatedCryptoExchange(granularity=1, feature_list=features)

    df = sim_env.df.copy()              # This is the dataframe with all of the features built.

    print('Testing all classifiers...')
    token = 'BTC'
    # Hyperparameters
    hyperparams = {'degrees': [1, 2, 3, 4, 5], 'k': [7]}
    df = knn_classifier(df, token, hyperparams, split=True, plot=False)
    hyperparams = {'degrees': [1, 2, 3, 4, 5], }
    df = mlp_classifier(df, token, hyperparams, split=True, plot=False)
    print('Data with classifiers:')
    print(df.head())
