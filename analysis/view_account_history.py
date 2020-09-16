from environments.environments import SimulatedCryptoExchange
from tools.tools import f_paths
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

path = f_paths['live log']

# new_path = path[:-4] + '.pkl'
# # print(new_path)
# df = pd.read_pickle(new_path)
# print(df)

# df['Total Value'] = round(df['Total Value'], 2) #This round to the nearest cent, which makes plotting nicer

#date range to train on
start = datetime(2020, 1, 1)
end = datetime(2020, 9, 1) #- timedelta(days = 1)
features = {  # 'sign': ['Close', 'Volume'],
    # 'EMA': [50, 80, 130],
    'OBV': [],
    'RSI': [],
    # 'high': [],
    # 'low': [],
    'BollingerBands': [3],
    'BBInd': [],
    'BBWidth': [],
    # 'discrete_derivative': ['BBWidth3'], #, 'BBWidth4', 'BBWidth5'],
    # 'time of day': [],
    # 'stack': [2],
    # 'rolling probability': ['BBInd3', 'BBWidth3']
    # 'probability': ['BBInd3', 'BBWidth3']
    }
sim_env = SimulatedCryptoExchange(granularity=1, feature_dict=features, train_amount=0)
sim_env.log.get_all_live_log()
sim_env.log.plot(sim_env.df, 'Live agent')
# fig, (ax1, ax2) = plt.subplots(2, 1)  # Create the figure

# # df.to_pickle(new_path)
# print(df)

# # df.plot(y = 'Total Value', ax = ax1)
# df.plot(x='Timestamp', y = 'Total Value', ax = ax1, style = 'bo')
# df.plot(x='Timestamp', y = '$ of BTC', ax = ax2)
plt.show()
