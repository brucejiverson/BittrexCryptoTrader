from bittrex_tools import *
import pandas as pd


symbols = 'BTCUSD' #Example: 'BTCUSD'

path = 'C:/Python Programs/crypto_trader/historical data/updated_history_' + symbols + '.csv'

def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p-%M")
up_df = pd.read_csv(path, usecols=['Date', 'Close'], parse_dates=['Date'], date_parser=dateparse)

print('The earliest date is: ', up_df.Date.min())
print('The latest date is: ', up_df.Date.max())

plot_history(up_df)
