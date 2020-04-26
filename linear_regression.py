import numpy as np
from sklearn.linear_model import LinearRegression
from environments import SimulatedCryptoExchange, paths
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
import pandas as pd

#date range to train on
end = datetime.now() - timedelta(days = 2)
start = end - timedelta(days = 9)
sim_env = SimulatedCryptoExchange()

df = sim_env.df.copy()
y_names = []
n_predictions = 1
#Calculate what the real
for i in range(n_predictions):
    step = (i+1)#*-1
    y_names.append('Step ' + str(step) + ' % Change')
    df[y_names[i]] = 100*(df['BTCClose'].shift(-step) - df['BTCClose'])/df['BTCClose']


# df['Last BTCClose'] = df['BTCClose'].shift(1) - df['BTCClose'].shift(2)
df['BTCClose'] = df['BTCClose'] - df['BTCClose'].shift(1)

df.dropna(inplace = True)
n_sample, n_feature = df.shape

#Split the data into testing and training dataframes
fraction_to_train_on = .8
n_sample_train = int(n_sample*.8)
training = df.iloc[0:n_sample_train]
n_sample_test = n_sample - n_sample_train
testing = df.iloc[n_sample_train:]

features_to_use = list(df.columns) #['BTCClose', 'BTCVolume', 'BTCOBV', 'EMA_30', 'd/dt_EMA_30', 'EMA_78', 'd/dt_EMA_78']
for i in range(n_predictions):
    features_to_use.remove(y_names[i])

x_train = training[features_to_use].values
y_train = training[y_names].values

x_test = testing[features_to_use].values
y_test = testing[y_names].values

model = LinearRegression().fit(x_train, y_train) #returns self

r_sq = model.score(x_train, y_train)
print('coefficient of determination for training:', r_sq)

r_sq = model.score(x_test, y_test)
print('coefficient of determination for testing:', r_sq)

print('intercept:', model.intercept_)
print('slope:', model.coef_)

print('Saving weights...', end = ' ')
#save the weights
path = paths['models'] + '/regression.pkl'
with open(path, 'wb') as file:
    pickle.dump(model, file)
print('done')

y_pred = model.predict(x_test)

#Analyze the predictions
new_df = pd.DataFrame(y_pred, index = testing.index)
new_df.columns = y_names

print(new_df.head())
print(testing.head())

thresh = .05 #%threshold to check
for i in range(n_predictions):
    col = y_names[i]
    temp = testing[new_df[col] > thresh]                        #The real values for all predictions greate than 0
    n_pred_over_0, n_col = temp.shape
    n_success, n_col  = temp[temp[col] > thresh].shape

    print(f'Out of {n_pred_over_0} predictions the price will be over {thresh}, ', end = '')
    print(f'{100*round(n_success/n_pred_over_0,2)}% were in the same direction.')

for i in range(n_predictions): #Loop over each time step in the future
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Real Value')
    ax1.set_xlabel('Predicted Value ' + str(i + 1) + ' timesteps forwards')
    ax1.set_title('Performance of Linear Regression Model On Test Data')
    plt.scatter(y_test[:,i], y_pred[:,i])

plt.show()
