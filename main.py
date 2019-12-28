from bittrex_tools import *
from datetime import datetime, timedelta
import pandas as pd

#cryptodatadownload has gaps
#Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********
os = 'windows' #linux or windows
des_granularity = 1 #in minutes
symbols = 'BTCUSD' #Example: 'BTCUSD'

#The below should be updated to be simplified to use parent directory? unsure how that works...
#https://stackoverflow.com/questions/48745333/using-pandas-how-do-i-save-an-exported-csv-file-to-a-folder-relative-to-the-scr?noredirect=1&lq=1

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


mode = 'train'

if mode in ['train', 'add_train']:
    #train
    # start = datetime(2019,11, 8)
    # end = datetime(2019,11,18)
    start = datetime(2019,12, 14)
    end = datetime(2019, 12, 28)
    # end = datetime.now() - timedelta(hours = 6)

else:
    assert(mode == 'test')  #make sure that a proper mode was passed
    #test
    # start = datetime(2018,1, 1)
    # end = datetime(2018, 3, 1)
    start = datetime(2019,12, 14)
    end = datetime(2019, 12, 27)
    # start = datetime(2017,11, 1)
    # end = datetime(2018, 1, 1)


run_agent(mode, paths, start, end)
