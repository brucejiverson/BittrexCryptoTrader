from features.predictor import Predictor
from tools.tools import f_paths, percent_change_column
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge


class ridge_regression(Predictor):
    """Constructs a knn classifier from the df passed. Hyperparameters should be a 
    dictionary, parameters are k, polynomial degrees. Note that this calculates 
    percentage change, not simple time differencing, as it makes the threshold more intuitive. 
    (temporarily changed as it was causing problems)"""

    def __init__(self, token, hyperparams, n_predictions=1):
        super().__init__(hyperparams, n_predictions)
        self.classifier = False
        self.name = 'ridge'
        self.path = f_paths['feature_models'] + '/' + self.name + '.pkl'
        self.config_path = f_paths['feature_models'] + '/' + self.name + '_config.pkl'
        print(f"Hyperparams: {hyperparams}")

        polynomial_features = PolynomialFeatures(degree=1)
        # linear_regression = LinearRegression()
        ridge = Ridge(alpha = 1.0, max_iter=10000)
        # The pipeline can be used as any other estimator
        # and avoids leaking the test set into the train set
        self.model = Pipeline([('polynomial_features', polynomial_features), 
                        ('ridge', ridge)])


    def plot_results(self, df):
        
        y_names = []
        y_predict = df['BTCridge'].values
        #Calculate what the next price will be
        for i in range(self.n_predictions):
            step = (i+1)  # *-1
            y_names.append('Step ' + str(step) + ' % Change')
            y = (df['BTCClose'].shift(-step) -
                    df['BTCClose'])  # /df['BTCClose']

        for i in range(self.n_predictions): #Loop over each time step in the future
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.set_ylabel('Real Value')
            ax1.set_xlabel('Predicted Value ' + str(i + 1) + ' timesteps forwards')
            ax1.set_title('Performance of Linear Regression Model On Test Data')
            # plt.scatter(y_train[:,i], y_train_predict[:,i])
            plt.scatter(y_predict, y)

        plt.show()

if __name__ == "__main__":
    from environments.environments import SimulatedCryptoExchange
    #date range to train on
    end = datetime.now() #- timedelta(days = 1)
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
    hyperparams = {'polynomial_features__degree': (1, 2, 3, 4, 5),
                    'ridge__fit_intercept':[False]}
    ridge_reg = ridge_regression(token, hyperparams)
    # ridge_reg.load()
    ridge_reg.optimize_parameters(df)
    # ridge_reg.train(df)
    df = ridge_reg.build_feature(df)
    df, new_name = percent_change_column('BTCClose', df, -1)
    print(df.head())
    ridge_reg.save()
    ridge_reg.plot_results(df)

#print out some info
# r_sq = model.score(X_train, y_train)
# print('coefficient of determination for training:', r_sq)

# r_sq = model.score(X_test, y_test)
# print('coefficient of determination for testing:', r_sq)

# X_test = scaler.transform(X_test)
# y_pred = model.predict(X_test)

# #Analyze the predictions
# new_df = pd.DataFrame(y_pred, index = testing.index)
# new_df.columns = y_names

# print(new_df.head())
# print(testing.head())
