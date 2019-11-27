from bittrex_tools import *
import pandas as pd


symbols = 'BTCUSD' #Example: 'BTCUSD'

os = 'windows'
if os == 'linux':
    paths = {'downloaded history': '/home/bruce/AlgoTrader/BittrexTrader/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
             'updated history': '/home/bruce/AlgoTrader/updated_history_' + symbols + '.csv',
             'secret': "/home/bruce/Documents/crypto_data/secrets.json",
             'rewards': 'agent_rewards',
             'models': 'agent_models',
              'test trade log':  'C:/Python Programs/crypto_trader/historical data/trade_testing' + symbols + '.csv'}

    # TODO: add a loop here that appends the asset folders

elif os == 'windows':
    paths = {'downloaded history': 'C:/Python Programs/crypto_trader/historical data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv',
             'updated history': 'C:/Python Programs/crypto_trader/historical data/updated_history_' + symbols + '.csv',
             'secret': "/Users/biver/Documents/crypto_data/secrets.json",
             'rewards': 'agent_rewards',
             'models': 'agent_models',
             'test trade log':  'C:/Python Programs/crypto_trader/historical data/trade_testing' + symbols + '.csv'}

orig_path = paths['downloaded history']

def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
orig_df = pd.read_csv(paths['downloaded history'], usecols=['Timestamp', 'Close'], parse_dates=[
    'Timestamp'], date_parser=dateparse)
orig_df.rename(columns={'Timestamp': 'Date', 'Close': 'BTCUSD'}, inplace=True)

# save_historical_data(paths, orig_df)  #just use this line if you want to paste in data.

#Use the below to purge and overwrite the data
