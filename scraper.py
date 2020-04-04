from main import *
from datetime import datetime, timedelta
import pandas as pd

start = datetime.now() - timedelta(days = 10)
end = datetime.now()

last_time_scraped = datetime.now() - timedelta(days = 20)
os = 'windows' #linux or windows
des_granularity = 1 #in minutes
symbols = 'BTCUSD' #Example: 'BTCUSD'

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
else:
    print('Unknown OS passed when defining the paths')  # this should throw and error

#get my keys
with open(paths['secret']) as secrets_file:
    keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

market = symbols[3:6] + '-' + symbols[0:3]


while True:
    df = fetch_historical_data(paths, market, start, end, my_bittrex)  #gets all data
