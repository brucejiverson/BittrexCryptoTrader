import pandas as pd
import numpy as np
from environments import *
from datetime import datetime, timedelta

start = datetime(2020, 4, 15)
end = datetime(2020, 4, 1)

path = paths['account log']
date_format = "%Y-%m-%d %I-%p-%M"

#Load the old log
# try:
def dateparse(x): return pd.datetime.strptime(x, date_format)

try:
    df = pd.read_csv(path, parse_dates = ['Timestamp'], date_parser=dateparse)
    df.set_index('Timestamp', inplace = True, drop = True)
    df = df.append(self.log, sort = True)
except pd.errors.EmptyDataError:
    print('There was no data in the log.')

#Filter for the desired time range
df = df.loc[df.index > start_date + timedelta(hours = 7)] #not the culprit in the altered gran
df = df.loc[df.index < end_date + timedelta(hours = 7)]

# success_rate 
