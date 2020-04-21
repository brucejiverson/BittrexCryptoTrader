import matplotlib.pyplot as plt
from environments import *

path = paths['account log']
date_format = "%Y-%m-%d %I-%p-%M"

def dateparse(x): return pd.datetime.strptime(x, date_format)
df = pd.read_csv(path, parse_dates = ['Timestamp'], date_parser=dateparse)
df.set_index('Timestamp', inplace = True, drop = True)

df['Total Value'] = round(df['Total Value'], 2) #This round to the nearest cent, which makes plotting nicer

fig, (ax1, ax2) = plt.subplots(2, 1)  # Create the figure

df.plot(y = 'Total Value', ax = ax1)
df.plot(y = '$ of BTC', ax = ax2)
plt.show()
