from bittrex_trader.environments.environments import *
from datetime import datetime, timedelta
import pandas as pd

end = datetime.now() - timedelta(days = 5)
start = end - timedelta(hours = 18)

last_time_scraped = datetime.now() - timedelta(days = .25)

sim_env = SimulatedCryptoExchange(start, end)
# sim_env.plot_
df = sim_env.df.copy()
df['Future Percent Change'] = 100*(df['BTCClose'].shift(-1) - df['BTCClose'])/df['BTCClose']
df['Last BTCClose'] = df['BTCClose'].shift(1) - df['BTCClose'].shift(2)
df['BTCClose'] = df['BTCClose'] - df['BTCClose'].shift(1)
#Time difference the prices
# df['Last Percent Change'] = df['Future Percent Change'].shift(1)
df.dropna(inplace = True)
print(df.tail())
fig, ax = plt.subplots(1, 1)  # Create the figure
feat1_name = 'BTCVolume'
feat2_name = 'BTCClose'
thresh = 0
df[df['Future Percent Change'] > thresh].plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'b')
df[df['Future Percent Change'] < thresh].plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'r')

#Implement a grid search for best features
n_s, n_f = df[df['BTCClose'] < 0].shape
bool = (df['Future Percent Change'] > thresh) & (df['BTCClose'] < -1 )
n_profit, n_f2 = df[bool].shape

avg = df[bool]['Future Percent Change'].mean()#axis = 1)

print(f'Out of {n_s} samples where the price signal was < {0}, {round(100*n_profit/n_s)}% of next return was above {thresh}%.')
print(f'Average was {round(avg,3)}%.')




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


fig, ax = plt.subplots(1, 1)  # Create the figure

fig.suptitle(' Feature and Future Price Correlation', fontsize=14, fontweight='bold')
df.plot(x='BTCVolume', y='Future Percent Change', ax=ax, kind = 'scatter')


plt.show()
