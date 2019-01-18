from bittrex_tools import *
from datetime import datetime, timedelta
from artemis import strategy
import pandas as pd


symbol = 'BTCUSD'
paths = {'Download': '/Users/biver/Documents/Crypto/Kraken_' + symbol + '_1h.csv', 'Updated': '/Users/biver/Documents/Crypto/Updated_' + symbol + '_1h.csv'}
# data = original_csv_to_df(paths, 1, 1, 2017)
data = updated_csv_to_df(paths, 1, 1, 2018)  # oldest date info

# get my keys
with open("/Users/biver/Documents/Crypto/secrets.json") as secrets_file:
    keys = json.load(secrets_file)
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

candle_dict = my_bittrex.get_candles('USD-BTC', 'hour')
if candle_dict['success']:
    new_data = process_bittrex_dict(candle_dict)
else:
    print("Failed to get candle data")

data = data.append(new_data)
data = data.sort_values(by='Date')
data = data.drop_duplicates(['Date'])
data.reset_index(inplace=True, drop=True)
overwrite_csv_file(paths['Updated'], data)

start_date = datetime(2018, 1, 1)
end_date = datetime(2018, 7, 3)
backtest(data, start_date, end_date, 30)
