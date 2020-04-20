import matplotlib.pyplot as plt
from environments import *

path = paths['account log']
date_format = "%Y-%m-%d %I-%p-%M"

def dateparse(x): return pd.datetime.strptime(x, date_format)
df = pd.read_csv(path, parse_dates = ['Timestamp'], date_parser=dateparse)
df.set_index('Timestamp', inplace = True, drop = True)
df.plot(y = 'Total Value')
plt.show()
