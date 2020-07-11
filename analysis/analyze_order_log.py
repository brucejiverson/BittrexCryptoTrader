import pandas as pd
import numpy as np
from environments.environments import SimulatedCryptoExchange
from tools.tools import f_paths
from datetime import datetime, timedelta

start = datetime(2020, 4, 18)
end = datetime(2020, 4, 22)

path = f_paths['order log']
df = pd.read_pickle(path)

#Filter for the desired time range
df = df.loc[df.index > start + timedelta(hours = 7)]
df = df.loc[df.index < end + timedelta(hours = 7)]

print('ORDER DATA: ')
print(df.head())

# success_rate
n_success, cols = df[df['Price'] == 0].shape
n_order, cols = df.shape

success_rate = 100*round(n_success/(n_order),3)

print(f'{success_rate}% of the {n_order} orders during the specified period were filled.')
