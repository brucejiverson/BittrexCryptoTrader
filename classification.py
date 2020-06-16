import numpy as np
from environments import SimulatedCryptoExchange, f_paths
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

#date range to train on
end = datetime.now() - timedelta(days = 2)
start = end - timedelta(days = 9)
sim_env = SimulatedCryptoExchange()

df = sim_env.df.copy()


y_names = []
n_predictions = 1       # The number of timesteps in the future that we are trying to predict

thresh = 0.005
#Calculate what the next price will be
for i in range(n_predictions):
    step = (i+1)#*-1
    y_names.append('Step ' + str(step) + ' % Change')
    series = df['BTCClose'].shift(-step) - df['BTCClose']
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

k = 10
knn = KNeighborsClassifier(n_neighbors = k)

# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe = Pipeline([('polynomial_features', polynomial_features), 
                ('knn', knn)])

n_rows, n_cols = X_train.shape
print(f'Training regression model on {n_cols} features...')
pipe.fit(X_train, y_train)

score = pipe.score(X_train, y_train)
print(f"R2 score on train data: {round(score,4)}.")
y_train_predict = pipe.predict(X_train)
score = pipe.score(X_test, y_test)
print(f"R2 score on test data: {round(score, 4)}.")
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

# This is the old stuff. Got linear reg looking good so I'm just gonna copy it over
# thresh = 0
# df['Future change'] = ((df['BTCClose'].shift(-1) - df['BTCClose'])/df['BTCClose'])*100
# df['Future change'].loc[df['Future change'] > thresh] = 1
# df['Future change'].loc[df['Future change'] < thresh] = 0

# # df['Last BTCClose'] = df['BTCClose'].shift(1) - df['BTCClose'].shift(2)
# df['BTCClose'] = df['BTCClose'] - df['BTCClose'].shift(1)
# df['BTCVolume'] = df['BTCVolume'] - df['BTCVolume'].shift(1)
# # polynomial_features = PolynomialFeatures(degree=degrees[i],
#                                              # include_bias=False)
# print('PROCESSED DATA:')
# print(df.head())
# """I accidentally changed all values to be 0 or 1 based on the thresh hold, and it
#  gave > .8 coeeficient when training compared to .56 when fixed."""
# #Loop to feature map
# new_df = df.copy()
# # for i, col in enumerate(df.columns):
# #     if col != 'Future change':
# #         new_name = col + '-sign'
# #         new_df[new_name] = 0
# #         new_df[new_name].loc[df[col] > 0] = 1

# df = new_df
# # df.drop(columns = ['BTCClose'], inplace = True)
# # print(df.head())
# df.dropna(inplace = True)
# n_sample, n_feature = df.shape
# n_feature -=1 #account for the ys
# #Split the data into testing and training dataframes
# fraction_to_train_on = .8
# n_sample_train = int(n_sample*.8)
# training = df.iloc[0:n_sample_train]
# n_sample_test = n_sample - n_sample_train
# testing = df.iloc[n_sample_train:]

# feature_names = df.columns
# feature_names = feature_names.drop('Future change')
# x_train = training[feature_names].values
# y_train = training['Future change'].values

# x_test = testing[feature_names].values
# y_test = testing['Future change'].values

# #Standardize the inputs
# scaler = StandardScaler()
# scaler.fit(x_train)
# classifier = MLPClassifier(alpha=1, max_iter=1000)
# x_train = scaler.transform(x_train)

# classifier.fit(x_train, y_train.ravel())

# #print out some info
# r_sq = classifier.score(x_train, y_train)
# print('coefficient of determination for training:', r_sq)

# r_sq = classifier.score(x_test, y_test)
# print('coefficient of determination for testing:', r_sq)

# print('Saving weights...', end = ' ')
# #save the weights
# path = paths['models'] + '/classification.pkl'
# with open(path, 'wb') as file:
#     pickle.dump(classifier, file)
# print('done')

# x_test = scaler.transform(x_test)
# y_pred = classifier.predict(x_test)

# #Analyze the predictions
# new_df = pd.DataFrame(y_pred, index = testing.index)#, columns = )
# new_df.columns = ['Future change']

# # temp = testing[new_df['Future change'] > thresh]                        #The real values for all predictions greate than 0
# # n_pred_over_0, n_col = temp.shape
# # n_success, n_col  = temp[temp['Future change'] > thresh].shape
# #
# # print(f'Out of {n_pred_over_0} predictions the price will be over {thresh}, ', end = '')
# # print(f'{100*round(n_success/n_pred_over_0,2)}% were above that threshhold.')

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_ylabel('Real Value')
# ax1.set_xlabel('Predicted Value ' + str(1) + ' timesteps forwards')
# ax1.set_title('Performance of Linear Regression Model On Test Data')
# plt.scatter(y_test, y_pred)

# plt.show()
