from environments.environments import SimulatedCryptoExchange
from tools.tools import percent_change_column
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

start = datetime(2020, 1, 1)
end = datetime(2020, 5, 1)

features = {    # 'sign': ['Close', 'Volume'],
    'EMA': [50, 80, 130],
    'OBV': [],
    'RSI': [],
    'BollingerBands': [2, 3, 5, 10],
    'BBInd': [],
    'BBWidth': [],
    # 'time of day': [],
    'stack': [5, 'Close']
    }

sim_env = SimulatedCryptoExchange(start, end, granularity=5, feature_dict=features)

df = sim_env.df.copy()

df, new_name = percent_change_column('BTCClose', df, -1)
df = percent_change_column('BTCClose', df, 1)
df = percent_change_column('BTCClose_shift_1', df, 1)
df = percent_change_column('BTCClose_shift_2', df, 1)
df = percent_change_column('BTCClose_shift_3', df, 1)
df = percent_change_column('BTCClose_shift_4', df, 1)
df = percent_change_column('BTCClose_shift_5', df, 1)
df = percent_change_column('BTCVolume', df, 1)

def criteria(input_df):
    # Returns a mask of the input data frame
    thresh = -.45
    profit_target = .5                  # percent
    df = input_df.copy()
    df['Labels'] = None
    df['Values'] = None
    for i in range(5):
        df.loc[(df['BBInd3'] < thresh) & (df['BTCClose_shift_' + str(i+1)] > profit_target), 'Labels'] = 1
        df.loc[(df['BBInd3'] < thresh) & (df['BTCClose_shift_' + str(i+1)] < profit_target), 'Labels'] = 0
        
    df.loc[df['Labels'] != None, 'Values'] = df.loc[df['Labels'] != None, 'BTCClose']
    return df

stats_df = criteria(df)
n_success = stats_df[stats_df['Labels'] == 1].shape[0]
n_signals = stats_df[stats_df['Labels'] != None].shape[0]
probability_of_success = n_success/n_signals
expected_value = stats_df['Values'].mean()
std_dev = stats_df[stats_df['Values']].std()
print(f'Probability of success: {probability_of_success}')
print(f'Expected Value: {expected_value}')
print(f'Dev: {std_dev}')

#Time difference the prices
# df['Last Percent Change'] = df['Future Percent Change'].shift(1)
df.dropna(inplace = True)
# print(df.tail())

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

# print(f'Out of {n_s} samples where the price signal was < {0}, {round(100*n_profit/n_s)}% of next return was above {thresh}%.')
# print(f'Average was {round(avg,3)}%.')

fig, ax = plt.subplots()  # Create the figure

fig.suptitle(' Feature and Future Price Correlation', fontsize=14, fontweight='bold')
df.plot(x='BTCOBV', y=new_name, ax=ax, kind = 'scatter')
# plt.show()