from bittrex_tools import *
from datetime import datetime, timedelta
from artemis import strategy
import pandas as pd

#cryptodatadownload has gaps
#Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********
os = 'windows' #linux or windows
des_granularity = 1 #in hours
symbol = 'BTCUSD' #Example: 'BTCUSD'
start_date = datetime(2019,1, 1)
end_date = datetime(2019, 9, 6)
# end_date = datetime.today()
extrema_filter = 11*24 #in hours
data_start = start_date - timedelta(hours= 10*extrema_filter)

if os == 'linux':
    paths = {'Download': '/home/bruce/AlgoTrader/BittrexTrader/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
     'Updated': '/home/bruce/AlgoTrader/updated_file_' + symbol + '.csv',
     'Secret': "/home/bruce/Documents/Crypto/secrets.json",
     'Test Trade Log': '/home/bruce/AlgoTrader/TestTradeLog',
     'Trade Log': '/home/bruce/AlgoTrader/TradeLog'}

elif os == 'windows':
    paths = {'Download': '/Users/biver/Documents/Crypto/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
    'Updated': '/Users/biver/Documents/Crypto//Updated_' + symbol + '.csv',
    'Secret': "/Users/biver/Documents/Crypto/secrets.json",
    'Test Trade Log': '/Users/biver/Documents/Crypto/TestTradeLog',
    'Trade Log': '/Users/biver/Documents/Crypto/TradeLog',
    'Experiment': '/Users/biver/Documents/Crypto/experiment_data_dump.csv'}
else:
    print('Unknown OS') #this should throw and error

# get my keys
with open(paths['Secret']) as secrets_file:
    keys = json.load(secrets_file) #loadBs the keys as a dictionary with 'key' and 'secret'
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)
os = 'windows' #linux or windows
des_granularity = 1 #in hours
symbol = 'BTCUSD' #Example: 'BTCUSD'

market = symbol[3:6] + '-' + symbol[0:3]
data = fetchHistoricalData(paths, market, des_granularity, data_start, end_date, my_bittrex)  # oldest date info

print('Historical data has been fetched from CSV.')

overwrite_csv_file(paths, data)


# roi = backtest(data, start_date, end_date, extrema_filter)
def experiment():
    df = pd.DataFrame(columns = ['ROI', 'n'])
    i = 0

    for n in np.linspace(5*24, 14*24, 19):

        roi = backtest(data, start_date, end_date, int(n), True)
        print("Completed trial # ", i)
        df.loc[i, 'ROI':'n'] = [roi, n]
        i += 1

    df.to_csv(paths['Experiment'])
    print(df)

# experiment()
