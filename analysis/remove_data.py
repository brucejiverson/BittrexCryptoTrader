from environments import *
from datetime import datetime, timedelta
import pandas as pd

#cryptodatadownload has gaps
#Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********


#The below should be updated to be simplified to use parent directory? unsure how that works...
#https://stackoverflow.com/questions/48745333/using-pandas-how-do-i-save-an-exported-csv-file-to-a-folder-relative-to-the-scr?noredirect=1&lq=1

start = datetime(2017, 1, 1)
end = datetime.now()

path = paths['cum data csv']

# datetimes to strings
print('Fetching data from the download file...', end = ' ')
# get the historic data. Columns are TimeStamp	Open	High	Low	Close	Volume_(BTC)	Volume_(Currency)	Weighted_Price

def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
cols_to_use = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(Currency)']
orig_df = pd.read_csv(paths['downloaded csv'], usecols=cols_to_use, parse_dates = ['Timestamp'], date_parser=dateparse)

name_map = {'Open': 'BTCOpen', 'High': 'BTCHigh', 'Low': 'BTCLow', 'Close': 'BTCClose', 'Volume_(Currency)': 'BTCVolume'}
orig_df.rename(columns= name_map, inplace=True)
orig_df.set_index('Timestamp', inplace = True, drop = True)
print('Fetched.')
df_to_save = self.df.append(orig_df, sort = True)

#Drop features and indicators, all non savable data
for col in df_to_save.columns:
    if not col[3::] in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_to_save.drop(col, axis = 1, inplace = True)

df_to_save = self._format_df(df_to_save)

# df_to_save = filter_error_from_download_data(df_to_save)
df_to_save.to_csv(path, index = True, index_label = 'TimeStamp', date_format = "%Y-%m-%d %I-%p-%M")

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
