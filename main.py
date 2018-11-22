from bittrex_tools import *
from datetime import datetime, timedelta
from artemis import strategy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


symbol = 'BTCUSD'
orig_path = '/Users/biver/Documents/Crypto/Kraken_'+symbol+'_1h.csv'
updated_path = '/Users/biver/Documents/Crypto/Updated_'+symbol+'_1h.csv'
#data = original_csv_to_df(orig_path)
data = updated_csv_to_df(updated_path, 5, 1, 2018)  # oldest date info


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
# overwrite_csv_file(updated_path, data)


start_date = datetime(2018, 1, 26)
end_date = datetime(2018, 11, 21)
backtest(data, start_date, end_date, 100)
