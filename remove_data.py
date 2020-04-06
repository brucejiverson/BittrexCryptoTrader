from main import *
from datetime import datetime, timedelta
import pandas as pd

#cryptodatadownload has gaps
#Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********


#The below should be updated to be simplified to use parent directory? unsure how that works...
#https://stackoverflow.com/questions/48745333/using-pandas-how-do-i-save-an-exported-csv-file-to-a-folder-relative-to-the-scr?noredirect=1&lq=1

start = datetime(2017, 1, 1)
end = datetime.now()



#get my keys
with open(paths['secret']) as secrets_file:
    keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

market = symbols[3:6] + '-' + symbols[0:3]

df = fetch_historical_data(paths, market, start, end, my_bittrex)  #gets all data
df = df[df['Date'] >= start]
df = df[df['Date'] <= end]
df = format_df(df)


print('Writing data to CSV.')

path = paths['updated history']
# must create new df as df is passed by reference
# # datetimes to strings

df_to_save = df
df_to_save['Date'] = df_to_save['Date'].dt.strftime("%Y-%m-%d %I-%p-%M")

df_to_save.to_csv(path, index=False)

# df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d %I-%p-%M")               # added this so it doesnt change if passed by object... might be wrong but appears to make a difference. Still dont have a great grasp on pass by obj ref.``
print('Data written.')


print('Historical data has been fetched, filtered by date, and resaved.')


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
