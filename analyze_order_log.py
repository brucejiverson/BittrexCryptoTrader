import pandas as pd
import numpy as np
from environments import *
from datetime import datetime, timedelta

start = datetime(2020, 4, 18)
end = datetime(2020, 4, 22)

path = paths['order log']
date_format = "%Y-%m-%d %I-%p-%M"

#Load the old log
#The format is the same in csv as the bittrex API returns for order data
def dateparse(x):
    try:
        return pd.datetime.strptime(x, date_format)
    except ValueError:  #handles cases for incomplete trades where 'Closed' is NaT
        return x
df = pd.read_csv(path, parse_dates = ['Opened', 'Closed'], date_parser=dateparse)
df.set_index('Opened', inplace = True, drop = True)

#Filter for the desired time range
df = df.loc[df.index > start + timedelta(hours = 7)]
df = df.loc[df.index < end + timedelta(hours = 7)]

print('ORDER DATA: ')
print(df.head())

# success_rate
n_success, cols = df[df['IsOpen'] == False].shape
n_order, cols = df.shape

success_rate = 100*round(n_success/(n_order),3)

print(f'{success_rate}% of the {n_order} orders during the specified period were filled.')
