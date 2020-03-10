from main import *
from datetime import datetime, timedelta
import pandas as pd

#cryptodatadownload has gaps
#Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********
os = 'windows' #linux or windows
des_granularity = 1 #in minutes
symbols = 'BTCUSD' #Example: 'BTCUSD'

#The below should be updated to be simplified to use parent directory? unsure how that works...
#https://stackoverflow.com/questions/48745333/using-pandas-how-do-i-save-an-exported-csv-file-to-a-folder-relative-to-the-scr?noredirect=1&lq=1

start = datetime(2016, 1, 1)
end = datetime.now()
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


df = fetch_historical_data(paths, market, start, end, my_bittrex)  #gets all data
df = df[df['Date'] >= start_date]
df = df[df['Date'] <= end_date]
df = format_df(df)

save_historical_data(paths, df)


print('Historical data has been fetched, updated, and resaved.')


assert not df.empty
fig, ax = plt.subplots(1, 1)  # Create the figure

market_perf = ROI(df.BTCClose.iloc[0], df.BTCClose.iloc[-1])
fig.suptitle('Market performance: ' + str(market_perf), fontsize=14, fontweight='bold')
df.plot(x='Date', y='BTCClose', ax=ax)


bot, top = plt.ylim()
cushion = 200
plt.ylim(bot - cushion, top + cushion)
fig.autofmt_xdate()
plt.show()
