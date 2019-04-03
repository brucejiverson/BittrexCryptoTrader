from bittrex_tools import *
from datetime import datetime, timedelta
from artemis import strategy
import pandas as pd

#New place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********

os = 'linux' #linux or windows

symbol = 'BTCUSD' #Example: 'BTCUSD'
if os == 'linux':
    paths = {'Download': '/home/bruce/AlgoTrader/BittrexTrader/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv', 'Updated': 'updated_file_' + symbol + '.csv'}
    secret_path = "/home/bruce/Documents/Crypto/secrets.json"
elif os == 'windows':
    paths = {'Download': '/Users/biver/Downloads/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv', 'Updated': '/Users/biver/Downloads/Updated_' + symbol + '.csv'}
    secret_path = "/Users/biver/Documents/Crypto/secrets.json"
data = original_csv_to_df(paths, symbol, 1, 1, 2018)
# data = updated_csv_to_df(paths, 8, 1, 2018)  # oldest date info

# get my keys
#linux path /home/bruce/Documents/Crypto/secrets.json
#windows path /Documents/Crypto/secrets.json
with open(secret_path) as secrets_file:
    keys = json.load(secrets_file)
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

# candle_dict = my_bittrex.get_candles('USD-BTC', 'hour')
# if candle_dict['success']:
#     new_data = process_bittrex_dict(candle_dict)
# else:
#     print("Failed to get candle data")

# data = data.append(new_data)
data = data.sort_values(by='Date')
data = data.drop_duplicates(['Date'])
data.reset_index(inplace=True, drop=True)
# overwrite_csv_file(paths['Updated'], data)

start_date = datetime(2018, 1, 1)
end_date = datetime(2018, 7, 3)

#print(data.head())
plot_market(data, start_date)
#backtest(data, start_date, end_date, 30)
