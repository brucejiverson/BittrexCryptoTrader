from environments.environments import SimulatedCryptoExchange
from tools.tools import percent_change_column
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

start = datetime(2020, 7, 1)
end = datetime(2020, 8, 7)

features = {    # 'sign': ['Close', 'Volume'],
    'EMA': [50, 80, 130],
    'OBV': [],
    'RSI': [],
    'BollingerBands': [1, 3, 5, 10],
    'BBInd': [],
    'BBWidth': [],
    'discrete_derivative': ['BBWidth3'],
    # 'time of day': [],
    # 'stack': [5]
    }

sim_env = SimulatedCryptoExchange(start, end, granularity=5, feature_dict=features)

df = sim_env.df.copy()

# Calculate what the sum for the next n timesteps are in terms of percentage

df, label_name = percent_change_column('BTCClose', df, -5)
# df, label_name2 = percent_change_column('BTCClose', df, -)
# df['Criteria'] = df[[label_name, label_name2]].mean(axis=1)
df['Criteria'] = df[label_name]
df = percent_change_column('BTCClose', df, 1)
df = percent_change_column('BTCVolume', df, 1)

print(df.head(50))
print(df.tail())

ax = plt.axes(projection='3d')  # Create the figure
feat1_name = 'BTCRSI'
feat2_name = 'BTCOBV'       # 'ddt_BBWidth3'
feat3_name = 'BBWidth3'     # To be used later
x = df[feat1_name].values
y = df[feat2_name].values
z = df['Criteria'].values
# fig.suptitle('Feature and price comparison', fontsize=14, fontweight='bold')
ax.set_xlabel(feat1_name)
ax.set_ylabel(feat2_name)
ax.set_zlabel('Criteria')
# ax.scatter3D(x, y ,z, c=z, cmap='Greens')

thresh = 0.08
# Color based plot for viewing 3 features at a time
ax = plt.axes(projection='3d')  # Create the figure

# Do some formatting
# fig.suptitle('Comparing 3 features', fontsize=14, fontweight='bold')
ax.set_xlabel(feat1_name)
ax.set_ylabel(feat2_name)
ax.set_zlabel(feat3_name)

# Get the up data
x = df[df['Criteria'] >= thresh][feat1_name].values
y = df[df['Criteria'] >= thresh][feat2_name].values
z = df[df['Criteria'] >= thresh][feat3_name].values
ax.scatter3D(x, y ,z, color='g')
# Get the down data
x = df[df['Criteria'] < thresh][feat1_name].values
y = df[df['Criteria'] < thresh][feat2_name].values
z = df[df['Criteria'] < thresh][feat3_name].values
ax.scatter3D(x, y ,z, color='r')

n_s, n_f = df[df['BTCClose'] < 0].shape
criteria = (df['Criteria'] > thresh) & (df['BBInd3'] < -.45 )
n_profit, n_f2 = df[criteria].shape

avg = df[criteria]['Criteria'].mean() # axis = 1)

print(f'Out of {n_s} samples where the price signal was < {0}, {round(100*n_profit/n_s)}% of next return was above {thresh}%.')
print(f'Average was {round(avg,3)}%.')

fig, ax = plt.subplots()  # Create the figure
fig.suptitle(' Feature and Future Price Correlation', fontsize=14, fontweight='bold')
# df.plot(x='BBInd3', y='Criteria', ax=ax, kind = 'scatter')
print(df.iloc[0])
up = df[df['Criteria'] > thresh][[feat1_name, feat2_name]]
down = df[df['Criteria'] < thresh][[feat1_name, feat2_name]]

up.plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'g')
down.plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'r')
plt.show()