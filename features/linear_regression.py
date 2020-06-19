from environments import SimulatedCryptoExchange, f_paths

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

#date range to train on
end = datetime.now() - timedelta(days = 2)
start = end - timedelta(days = 9)
sim_env = SimulatedCryptoExchange()

df = sim_env.df.copy()
y_names = []
n_predictions = 1       # The number of timesteps in the future that we are trying to predict

#Calculate what the next price will be
for i in range(n_predictions):
    step = (i+1)#*-1
    y_names.append('Step ' + str(step) + ' % Change')
    df[y_names[i]] = (df['BTCClose'].shift(-step) - df['BTCClose'])     #/df['BTCClose']

# These do time differencing 
# df['Last BTCClose'] = df['BTCClose'].shift(1) - df['BTCClose'].shift(2)
df['BTCClose'] = df['BTCClose'] - df['BTCClose'].shift(1)
df['BTCVolume'] = df['BTCVolume'] - df['BTCVolume'].shift(1)

df.dropna(inplace = True)
n_sample, n_feature = df.shape

features_to_use = list(df.columns) # ['BTCClose', 'BTCVolume', 'BTCOBV', 'EMA_30', 'd/dt_EMA_30', 'EMA_78', 'd/dt_EMA_78']
# Remove the y labels from the 'features to use' 
for i in range(n_predictions):
    features_to_use.remove(y_names[i])

X = df[features_to_use].values
y = df[y_names].values

# Split the data into testing and training dataframes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # , random_state=42)

# Hyperparameters
hyperparams = {'degrees': [1, 2, 3, 4, 5]}

degree = hyperparams['degrees'][2]
polynomial_features = PolynomialFeatures(degree=degree)
# linear_regression = LinearRegression()
ridge = Ridge(alpha = 1.0, max_iter=10000)
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe = Pipeline([('polynomial_features', polynomial_features), 
                ('ridge', ridge)])

n_rows, n_cols = X_train.shape
print(f'Training regression model on {n_cols} features...')
pipe.fit(X_train, y_train)

score = pipe.score(X_train, y_train)
print(f"R2 score on train data: {score}.")
y_train_predict = pipe.predict(X_train)
score = pipe.score(X_test, y_test)
print(f"R2 score on test data: {score}.")
y_test_predict = pipe.predict(X_test)


thresh = .01 #%threshold to check

# for i in range(n_predictions):
#     col = y_names[i]
#     temp = testing[new_df[col] > thresh]                        #The real values for all predictions greate than 0
#     n_pred_over_0, n_col = temp.shape
#     n_success, n_col  = temp[temp[col] > thresh].shape

#     print(f'Out of {n_pred_over_0} predictions the price will be over {thresh}, ', end = '')
#     print(f'{100*round(n_success/n_pred_over_0,2)}% were above that threshhold.')

for i in range(n_predictions): #Loop over each time step in the future
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Real Value')
    ax1.set_xlabel('Predicted Value ' + str(i + 1) + ' timesteps forwards')
    ax1.set_title('Performance of Linear Regression Model On Test Data')
    # plt.scatter(y_train[:,i], y_train_predict[:,i])
    plt.scatter(y_test[:,i], y_test_predict[:,i])

plt.show()

# model = LinearRegression().fit(X_train, y_train) #returns self

# model = make_pipeline(StandardScaler(), PolynomialFeatures(3), Ridge())
# model.fit(this_X, this_y)
# mse = mean_squared_error(model.predict(X_test), y_test)
# y_plot = model.predict(x_plot[:, np.newaxis])
# plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
#                 linewidth=lw, label='%s: error = %.3f' % (name, mse))
#
# legend_title = 'Error of Mean\nAbsolute Deviation\nto Non-corrupt Data'
# legend = plt.legend(loc='upper right', frameon=False, title=legend_title,
                # prop=dict(size='x-small'))

#print out some info
# r_sq = model.score(X_train, y_train)
# print('coefficient of determination for training:', r_sq)

# r_sq = model.score(X_test, y_test)
# print('coefficient of determination for testing:', r_sq)

# print('intercept:', model.intercept_)
# print('slope:', model.coef_)

# print('Saving weights...', end = ' ')
# #save the weights
# path = paths['models'] + '/regression.pkl'
# with open(path, 'wb') as file:
#     pickle.dump(model, file)
# print('done')

# X_test = scaler.transform(X_test)
# y_pred = model.predict(X_test)

# #Analyze the predictions
# new_df = pd.DataFrame(y_pred, index = testing.index)
# new_df.columns = y_names

# print(new_df.head())
# print(testing.head())
