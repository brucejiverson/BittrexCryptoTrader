from environments.environments import SimulatedCryptoExchange
from tools.tools import percent_change_column
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

end = datetime(2020, 7, 1)
start = end - timedelta(hours = 18)

last_time_scraped = datetime.now() - timedelta(days = .25)

features = {    # 'sign': ['Close', 'Volume'],
    'EMA': [50, 80, 130],
    'OBV': [],
    'RSI': [],
    'BollingerBands': [1, 5, 10],
    'BBInd': [],
    'BBWidth': [],
    'time of day': [],
    # 'stack': [1]
    }
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
ax = plt.axes(projection='3d')  # Create the figure
feat1_name = 'BBWidth10'
feat2_name = 'BBInd10'
thresh = 0.01
x = df[feat1_name].values
y = df[feat2_name].values
z = df[new_name].values
ax.scatter3D(x, y ,z, c=z, cmap='Greens')
# x_up = df[df[new_name] > thresh].loc[feat1_name]
# x_down = df[df[new_name] < thresh].loc[feat1_name]
# y_up = df[df[new_name] > thresh].loc[feat2_name]
# y_down = df[df[new_name] < thresh].loc[feat2_name]

# .plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'b')
# .plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'r')

#Implement a grid search for best features
n_s, n_f = df[df['BTCClose'] < 0].shape
bool = (df[new_name] > thresh) & (df['BTCClose'] < -1 )
n_profit, n_f2 = df[bool].shape

avg = df[bool][new_name].mean()#axis = 1)

print(f'Out of {n_s} samples where the price signal was < {0}, {round(100*n_profit/n_s)}% of next return was above {thresh}%.')
print(f'Average was {round(avg,3)}%.')

fig, ax = plt.subplots()  # Create the figure

fig.suptitle(' Feature and Future Price Correlation', fontsize=14, fontweight='bold')
df.plot(x='BTCOBV', y=new_name, ax=ax, kind = 'scatter')
plt.show()