from environments.environments import BittrexExchange
from tools.tools import f_paths
import matplotlib.pyplot as plt
import pandas as pd
path = f_paths['live log']

new_path = path[:-4] + '.pkl'
print(new_path)
df = pd.read_pickle(new_path)
df.to_pickle(new_path)

df['Total Value'] = round(df['Total Value'], 2) #This round to the nearest cent, which makes plotting nicer

fig, (ax1, ax2) = plt.subplots(2, 1)  # Create the figure

print(df.head())
df.plot(y = 'Total Value', ax = ax1)
df.plot(y = 'Total Value', ax = ax1, style = 'bo')
df.plot(y = '$ of BTC', ax = ax2)
plt.show()
