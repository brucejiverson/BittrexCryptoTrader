from main import *
from datetime import datetime, timedelta
import pandas as pd

#cryptodatadownload has gaps
#Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********

#get my keys
with open(paths['secret']) as secrets_file:
    keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD'] #Example: 'BTCUSD'
markets = [sym[3:6] + '-' + sym[0:3] for sym in symbols]

start = datetime(2016, 1, 1)
end = datetime.now()

df = fetch_historical_data(paths, markets, start, end, my_bittrex)  # oldest date info

print('Historical data has been fetched, updated, and resaved.')

assert not df.empty
for sym in symbols:
    token = sym[0:3]
    fig, ax = plt.subplots(1, 1)  # Create the figure

    market_perf = ROI(df[token + 'Close'].iloc[0], df[token + 'Close'].iloc[-1])
    fig.suptitle('Market performance: ' + str(market_perf), fontsize=14, fontweight='bold')
    df.plot(x='Date', y= token + 'Close', ax=ax)

    bot, top = plt.ylim()
    cushion = 200
    plt.ylim(bot - cushion, top + cushion)
    fig.autofmt_xdate()
plt.show()
