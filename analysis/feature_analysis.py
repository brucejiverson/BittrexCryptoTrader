from environments.environments import SimulatedCryptoExchange
from tools.tools import percent_change_column
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

end = datetime(2020, 7, 1)
start = end - timedelta(hours = 18)

last_time_scraped = datetime.now() - timedelta(days = .25)

features = {    # 'sign': ['Close', 'Volume'],
    'EMA': [50, 80, 130],
    'OBV': [],
    'RSI': [],
    'time of day': [],
    'stack': [1]}
sim_env = SimulatedCryptoExchange(start, end, granularity=1, feature_dict=features)

df = sim_env.df.copy()
df, new_name = percent_change_column('BTCClose', df, -1)
df = percent_change_column('BTCClose', df, 1)
df = percent_change_column('BTCVolume', df, 1)

#Time difference the prices
# df['Last Percent Change'] = df['Future Percent Change'].shift(1)
df.dropna(inplace = True)
print(df.tail())

# Plot the y axis
fig, ax = plt.subplots(1, 1)  # Create the figure
feat1_name = 'BTCOBV'
feat2_name = 'BTCRSI'
thresh = 0.01
df[df[new_name] > thresh].plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'b')
df[df[new_name] < thresh].plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'r')

#Implement a grid search for best features
n_s, n_f = df[df['BTCClose'] < 0].shape
bool = (df[new_name] > thresh) & (df['BTCClose'] < -1 )
n_profit, n_f2 = df[bool].shape

avg = df[bool][new_name].mean()#axis = 1)

print(f'Out of {n_s} samples where the price signal was < {0}, {round(100*n_profit/n_s)}% of next return was above {thresh}%.')
print(f'Average was {round(avg,3)}%.')

fig, ax = plt.subplots(1, 1)  # Create the figure

fig.suptitle(' Feature and Future Price Correlation', fontsize=14, fontweight='bold')
df.plot(x='BTCOBV', y=new_name, ax=ax, kind = 'scatter')
plt.show()

# assert not sim_env.df.empty
# for market in sim_env.markets:
#     token = market[4:7]
#     # sim_env.df['Combo'] = sim_env.df[token + 'MACD']*sim_env.df[token + 'RSI']
#
#     features = [token + 'MACD', token + 'RSI', token + 'OBV']#, 'Combo']
#
#     for feature in features:
#         fig, ax = plt.subplots(1, 1)  # Create the figure
#
#         fig.suptitle(feature + ' Feature and Future Price Correlation For' + token, fontsize=14, fontweight='bold')
#         df.plot(x=feature, y='Future Price', ax=ax, kind = 'scatter')
