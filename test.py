#pd.date_range gives DatetimeIndex, which is comprised of dtype datetime64. but when those objects
#are individually check they are pandas._libs.tslibs.timestamps.Timestamp
from bittrex_tools import *
from datetime import datetime, timedelta
from artemis import strategy
import pandas as pd


start_date = datetime(2018, 10, 26)
end_date = datetime(2018, 10, 27)
backtest(start_date, end_date, 3)


symbol = 'BTCUSD'
orig_path = '/Users/biver/Documents/Crypto/Kraken_'+symbol+'_1h.csv'
updated_path = '/Users/biver/Documents/Crypto/Updated_'+symbol+'_1h.csv'
# df = original_csv_to_df(orig_path, 5, 1, 2018)
df = updated_csv_to_df(updated_path, 5, 1, 2018)  # oldest date info

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

df = df.append(new_data)
df = df.sort_values(by='Date')
df = df.drop_duplicates(['Date'])
df.reset_index(inplace=True, drop=True)
# overwrite_csv_file(updated_path, df)

fig, ax = plt.subplots()
df.plot(x='Date', y='Close', ax=ax)

# set font and rotation for date tick labels
fig.autofmt_xdate()
plt.show()  # this is the timerange to look for local extremes
