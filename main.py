from bittrex_tools import *
from datetime import datetime, timedelta
from artemis import strategy
import pandas as pd

#New place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********

os = 'windows' #linux or windows
des_granularity = 1 #in hours
symbol = 'BTCUSD' #Example: 'BTCUSD'
start_date = datetime(2017,12, 1)
end_date = datetime(2018, 8, 1)
extrema_filter = 10*24 #in hours
data_start = start_date - timedelta(hours= 10*extrema_filter)

if os == 'linux':
    paths = {'Download': '/home/bruce/AlgoTrader/BittrexTrader/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv', 'Updated': '/home/bruce/AlgoTrader/updated_file_' + symbol + '.csv'}
    secret_path = "/home/bruce/Documents/Crypto/secrets.json"
elif os == 'windows':
    paths = {'Download': '/Users/biver/Downloads/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv', 'Updated': '/Users/biver/Downloads/Updated_' + symbol + '.csv'}
    secret_path = "/Users/biver/Documents/Crypto/secrets.json"
else:
    print('Unknown OS')
#TODO the below should all be merged into one function fetch that handles it all
# data = original_csv_to_df(paths, symbol, des_granularity, data_start)
data = updated_csv_to_df(paths, symbol, des_granularity, data_start)  # oldest date info
print('Historical data has been fetched from CSV.')
# get my keys
with open(secret_path) as secrets_file:
    keys = json.load(secrets_file)
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)


candle_dict = my_bittrex.get_candles('USD-BTC', 'hour')
if candle_dict['success']:
    new_data = process_bittrex_dict(candle_dict)
else:
    print("Failed to get candle data")

data = data.append(new_data)
data = data.drop_duplicates(['Date'])
data = data.sort_values(by='Date')
data.reset_index(inplace=True, drop=True)
overwrite_csv_file(paths, data)


roi = backtest(data, start_date, end_date, extrema_filter, False)
def experiment():
    df = pd.DataFrame(columns = ['ROI', 'n'])
    i = 0

    for n in np.linspace(5*24, 14*24, 19):

        roi = backtest(data, start_date, end_date, int(n), True)
        print("Completed trial # ", i)
        df.loc[i, 'ROI':'n'] = [roi, n]
        i += 1

    experiment_path = '/Python Programs/CryptoTrader/BittrexTrader/experiment_data_dump.csv'
    df.to_csv(experiment_path)
    print(df)

# experiment()
