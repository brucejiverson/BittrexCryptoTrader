import numpy as np
from sklearn.linear_model import LinearRegression
from environments import SimulatedCryptoExchange, paths
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle

end = datetime.now() - timedelta(days = 2)
start = end - timedelta(days = 9)
sim_env = SimulatedCryptoExchange()

df = sim_env.df.copy()
df['Future Percent Change'] = 100*(df['BTCClose'].shift(1) - df['BTCClose'])/df['BTCClose']
df.dropna(inplace = True)
n_sample, n_feature = df.shape

fraction_to_train_on = .8
n_sample_train = int(n_sample*.8)
training = df.iloc[0:n_sample_train]
n_sample_test = n_sample - n_sample_train
testing = df.iloc[n_sample_train:]

features_to_use = ['BTCClose', 'BTCVolume', 'BTCOBV', 'SMA_60', 'd/dt_SMA_60']#, 'SMA_80', 'd/dt_SMA_80']
x_train = training[features_to_use].values
y_train = training['Future Percent Change'].values

x_test = testing[features_to_use].values
y_test = testing['Future Percent Change'].values

model = LinearRegression().fit(x_train,y_train) #returns self

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

# print(x_test)
# print(type(x_test))
y_pred = model.predict(x_test)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Real Value')
ax1.set_xlabel('Predicted Value')
ax1.set_title('Performance of Linear Regression Model On Test Data')
plt.scatter(y_test, y_pred)

plt.show()

# print('predicted response:', y_pred, sep='\n')
