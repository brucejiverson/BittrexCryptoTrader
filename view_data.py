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

market = symbols[3:6] + '-' + symbols[0:3]


# start = datetime(2012,1, 1)
# end = datetime.now() - timedelta(hours = 1)

start = datetime(2016, 1, 1)
end = datetime.now()

df = fetch_historical_data(paths, market, start, end, my_bittrex)  # oldest date info

# save_historical_data(paths, df)


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
