from main import *
from datetime import datetime, timedelta
import pandas as pd

start = datetime.now() - timedelta(days = 10)
end = datetime.now()

last_time_scraped = datetime.now() - timedelta(days = 20)

symbols = ['BTCUSD'] #, 'ETHUSD', 'LTCUSD'] #Example: 'BTCUSD'
markets = [sym[3:6] + '-' + sym[0:3] for sym in symbols]


#get my keys
with open(paths['secret']) as secrets_file:
    keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

df = fetch_historical_data(paths, markets, start, end, my_bittrex)  #gets all data

print("ORIGINAL DATA: ")
print(df.head())
df = change_df_granulaty(df, 5)
for param in [30]:
    add_features(df, renko_block = param)
    print("DATA TO RUN ON: ")
    print(df.head())

    for symbol in symbols:
        token = symbol[0:3]
        df['Future Price'] = df[token + 'Close'].shift(1) - df[token + 'Close']

        assert not df.empty

        features = ['Renko', token + 'MACD', token + 'RSI']

        for feature in features:
            fig, ax = plt.subplots(1, 1)  # Create the figure

            fig.suptitle(feature + ' Feature and Future Price Correlation For' + token, fontsize=14, fontweight='bold')
            df.plot(x=feature, y='Future Price', ax=ax, kind = 'scatter')

            # bot, top = plt.ylim()
            # cushion = 200
            # plt.ylim(bot - cushion, top + cushion)

plt.show()
