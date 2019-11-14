from bittrex_tools import *
from datetime import datetime, timedelta
import pandas as pd

#cryptodatadownload has gaps
#Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********
os = 'windows' #linux or windows
des_granularity = 1 #in minutes
symbols = 'BTCUSD' #Example: 'BTCUSD'

# end = datetime.today()
# data_start = start# data_start = start - timedelta(hours= 24)

#The below should be updated to be simplified to use parent directory? unsure how that works...
#https://stackoverflow.com/questions/48745333/using-pandas-how-do-i-save-an-exported-csv-file-to-a-folder-relative-to-the-scr?noredirect=1&lq=1

if os == 'linux':
    paths = {'downloaded history': '/home/bruce/AlgoTrader/BittrexTrader/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
             'updated history': '/home/bruce/AlgoTrader/updated_history_' + symbols + '.csv',
             'secret': "/home/bruce/Documents/crypto_data/secrets.json",
             'rewards': 'agent_rewards',
             'models': 'agent_models'}

    # TODO: add a loop here that appends the asset folders

elif os == 'windows':
    paths = {'downloaded history': 'C:/Python Programs/crypto_trader/historical data/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv',
             'updated history': 'C:/Python Programs/crypto_trader/historical data/updated_history_' + symbols + '.csv',
             'secret': "/Users/biver/Documents/crypto_data/secrets.json",
             'rewards': 'agent_rewards',
             'models': 'agent_models'}
else:
    print('Unknown OS passed when defining the paths')  # this should throw and error

#get my keys
with open(paths['secret']) as secrets_file:
    keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

market = symbols[3:6] + '-' + symbols[0:3]

mode = 'train'

if mode == 'train':
    #train
    start = datetime(2016,1, 1)
    end = datetime(2017, 1, 1)
    # start = datetime.now() - timedelta(days = 9, hours = 23)
    # end = datetime.now() - timedelta(days = 5)
    df = fetch_historical_data(paths, market, start, end, my_bittrex)  # oldest date info
    save_historical_data(paths, df)
else:
    assert(mode == 'test')
    #test
    start = datetime(2019,11, 10)
    end = datetime(2019, 11, 13)
    # start = datetime.now() - timedelta(days = 5)
    # end = datetime.now()
    df = fetch_historical_data(paths, market, start, end, my_bittrex)  # oldest date info

    # save_historical_data(paths, df)

print('Historical data has been fetched, updated, and resaved.')

plot_history(df)

run_agent(mode, df.drop('Date', axis = 1).values, my_bittrex, paths) #note: only passing through numpy array not df

plot_history(df)


# roi = backtest(data, start, end, extrema_filter)
