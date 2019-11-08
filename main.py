from bittrex_tools import *
from datetime import datetime, timedelta
from artemis import strategy
import pandas as pd

#cryptodatadownload has gaps
#Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********
os = 'windows' #linux or windows
des_granularity = 1 #in hours
symbols = 'BTCUSD' #Example: 'BTCUSD'
start_date = datetime(2019,1, 1)
end_date = datetime(2019, 9, 6)
# end_date = datetime.today()
data_start = start_date - timedelta(hours= 24)

#The below should be updated to be simplified
if os == 'linux':
    paths = {'downloaded history': '/home/bruce/AlgoTrader/BittrexTrader/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
     'updated history': '/home/bruce/AlgoTrader/updated_history_' + symbols + '.csv',
     'secret': "/home/bruce/Documents/crypto_data/secrets.json",
     'rewards': 'agent_rewards',
     'models': 'agent_models'}

     #TODO: add a loop here that appends the asset folders

elif os == 'windows':
    paths = {'downloaded history': '/Users/biver/Documents/crypto_data/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
    'updated history': '/Users/biver/Documents/crypto_data/updated_history_' + symbols + '.csv',
    'secret': "/Users/biver/Documents/crypto_data/secrets.json",
    'rewards': 'agent_rewards',
    'models': 'agent_models'}
else:
    print('Unknown OS passed when defining the paths') #this should throw and error

# get my keys
with open(paths['secret']) as secrets_file:
    keys = json.load(secrets_file) #loadBs the keys as a dictionary with 'key' and 'secret'
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)
os = 'windows' #linux or windows
des_granularity = 1 #in hours
symbols = 'BTCUSD' #Example: 'BTCUSD'

market = symbols[3:6] + '-' + symbols[0:3]
data = fetchHistoricalData(paths, market, des_granularity, data_start, end_date, my_bittrex)  # oldest date info

print('Historical data has been fetched from CSV.')

overwrite_csv_file(paths, data)
print(data)

# roi = backtest(data, start_date, end_date, extrema_filter)
