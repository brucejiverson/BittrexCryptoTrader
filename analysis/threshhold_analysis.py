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
y_names = ['Criteria']
# Calculate what the sum for the next n timesteps are in terms of percentage
df, label_name = percent_change_column('BTCClose', df, -5)
df, label_name2 = percent_change_column('BTCClose', df, -8)
df['Criteria'] = df[[label_name, label_name2]].mean(axis=1) 


# df['Last BTCClose'] = df['BTCClose'].shift(1) - df['BTCClose'].shift(2)
df['BTCClose'] = df['BTCClose'] - df['BTCClose'].shift(1)
df['BTCVolume'] = df['BTCVolume'] - df['BTCVolume'].shift(1)

df.dropna(inplace = True)
n_sample, n_feature = df.shape

#Split the data into testing and training dataframes
fraction_to_train_on = .8
n_sample_train = int(n_sample*.8)
training = df.iloc[0:n_sample_train]
n_sample_test = n_sample - n_sample_train
testing = df.iloc[n_sample_train:]

x_train = training[features_to_use].values
y_train = training[y_names].values

x_test = testing[features_to_use].values
y_test = testing[y_names].values

#Analyze the predictions
new_df = pd.DataFrame(y_pred, index = testing.index)
new_df.columns = y_names

print(new_df.head())
print(testing.head())

thresh = .1 # % threshold to check

col = y_names[0]
temp = testing[new_df[col] > thresh]                        #The real values for all predictions greate than 0
n_pred_over_0, n_col = temp.shape
n_success, n_col  = temp[temp[col] > thresh].shape

print(f'Out of {n_pred_over_0} predictions the price will be over {thresh}, ', end = '')
print(f'{100*round(n_success/n_pred_over_0,2)}% were above that threshhold.')

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Real Value')
ax1.set_xlabel('Predicted Value')
ax1.set_title('Performance of Linear Regression Model On Test Data')
plt.scatter(y_test[:,i], y_pred[:,i])

plt.show()
