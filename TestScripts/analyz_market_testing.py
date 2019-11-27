import pandas as pd
import statistics
from datetime import datetime


symbols = 'BTCUSD'  # Example: 'BTCUSD'
market = symbols[3:6] + '-' + symbols[0:3]

paths = {'downloaded history': 'C:/Python Programs/crypto_trader/historical data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv',
         'updated history': 'C:/Python Programs/crypto_trader/historical data/updated_history_' + symbols + '.csv',
         'secret': "/Users/biver/Documents/crypto_data/secrets.json",
         'rewards': 'agent_rewards',
         'models': 'agent_models',
         'test trade log':  'C:/Python Programs/crypto_trader/historical data/trade_testing' + symbols + '.csv'}


path = paths['test trade log']

def dateparse(x):
    try:
        return pd.datetime.strptime(x, "%Y-%m-%d %I-%p-%M")
    except ValueError:
        return datetime(year = 2000, month = 1, day = 1)

df = pd.read_csv(path, parse_dates=['Opened', 'Closed'], date_parser=dateparse)

df['Closed'] = df['Closed'].dt.strftime("%Y-%m-%d %I-%p-%M")
df['Opened'] = df['Opened'].dt.strftime("%Y-%m-%d %I-%p-%M")

#Count how many attempted orders there were
n_attempts = df.shape[0]

#Count how many attempted trades Failed
filled_orders = df[df['IsOpen'] == False]
n_unfilled = df[df['IsOpen'] == True].shape[0]

#Count how many trades succeeded
n_filled = filled_orders.shape[0]

#Out of the trades that succeeded, what was the mean and the std dev of the trade time
mean_fill_time = filled_orders['Order Duration'].mean()
std_fill_time = filled_orders['Order Duration'].std()

print(f'Out of {n_attempts} attempts, {n_filled} were filled ({100*n_filled/n_attempts:.1f} %).')
print(f'The mean time a successful order was filled was {mean_fill_time:.2f}, +- {2*std_fill_time:.2f}.')
