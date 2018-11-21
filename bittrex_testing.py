from bittrex.bittrex import *
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#I had a ton of trouble getting the plots to look right with the dates. I eventually figured out to use ___ type date formatting
#This link was really helpful http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html

def process_bittrex_dict(my_dict):
    # reorder the columns (default to alphabetic)
    # V and BV refer to volume and base volume
    my_df = pd.DataFrame(my_dict['result'])
    my_df.drop(columns=["BV", "V"])
    my_df = my_df.rename(index=str, columns={'T': "Date",
                                             'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close'})
    my_df = my_df[['Date', 'Open', 'High', 'Low', 'Close']]
    # dates into datetimes
    # my_df.Date = pd.to_datetime(my_df.Date, format = "%Y-%m-%dT%H")
    for i, row in my_df.iterrows():
        my_df.loc[i, 'Date'] = datetime.strptime(my_df.loc[i, 'Date'][0:13], "%Y-%m-%dT%H")
    return my_df


def original_csv_to_df(path, oldest):
    # get the historic data
    my_df = pd.read_csv(path, header=1)

    my_df.drop(columns=["Symbol", "Volume From", "Volume To"])
    my_df = my_df[['Date', 'Open', 'High', 'Low', 'Close']]
    # dates into datetimes
    my_df.Date = pd.to_datetime(my_df.Date, format="%Y-%m-%d %I-%p")
    my_df.sort_values(by='Date', inplace=True)
    return my_df


def updated_csv_to_df(path, oldest_month, oldest_day, oldest_year):
    # get the historic data
    my_df = pd.read_csv(path)

    # my_df.drop(columns=["Symbol"])
    my_df = my_df[['Date', 'Open', 'High', 'Low', 'Close']]
    # dates into datetimes
    # my_df.Date = pd.to_datetime(my_df.Date, format="%Y-%m-%d %I-%p")
    for i, row in my_df.iterrows():
        my_df.loc[i, 'Date'] = datetime.strptime(my_df.loc[i, 'Date'], "%Y-%m-%d %I-%p")

    oldest = datetime(day=oldest_day, month=oldest_month, year=oldest_year)
    my_df = my_df[my_df['Date'] > oldest]
    my_df.sort_values(by='Date', inplace=True)
    return my_df


def overwrite_csv_file(path, my_df):
    # datetimes to strings
    for i, row in my_df.iterrows():
        date = row.loc['Date']
        my_df.loc[i, 'Date'] = date.strftime("%Y-%m-%d %I-%p")

    df.to_csv(path)


def process(my_df, win):
    # window is the timerange to look for local extremes in hours
    window = timedelta(hours=win)
    cushion = win/4
    cushion = timedelta(hours=cushion)
    # for i, row in my_df[my_df['Date'] < ].iterrows():
    #     # if
    #     pass

    # define the convention: lmin, lmax, cmin, cmax
    # find the most recent critical max/min
    # try:
    #     most_recent_date = my_df.loc[my_df['Critical'] != '' ].iloc[0]
    # except IndexError:
    #     #there is no data, this is the initialization case
    #     #identify all local min and maxes
    #     most_recent_date = my_df.iloc[0]['Date'] #why do i have this again...
    #     my_df[my_df[]]

    # my_df.loc[my_df['Date'] >= most_recent_date, ]
    # for i, row in reversed(my_df.iterrows()):
    #     if (row):
    #         pass
    # #identitfy new local mins/maxes

    # filter all mins/maxes for critical points

    # identify any new local mins/maxes


symbol = 'BTCUSD'
orig_path = '/Users/biver/Documents/Crypto/Kraken_'+symbol+'_1h.csv'
updated_path = '/Users/biver/Documents/Crypto/Updated_'+symbol+'_1h.csv'
#df = original_csv_to_df(orig_path)
df = updated_csv_to_df(updated_path, 5, 1, 2018)  # oldest date info

# get my keys
with open("/Users/biver/Documents/Crypto/secrets.json") as secrets_file:
    keys = json.load(secrets_file)
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

candle_dict = my_bittrex.get_candles('USD-BTC', 'hour')
if candle_dict['success']:
    new_data = process_bittrex_dict(candle_dict)
else:
    print("Failed to get candle data")


df = df.append(new_data)
df = df.sort_values(by='Date')
df = df.drop_duplicates(['Date'])
df.reset_index(inplace=True, drop=True)
overwrite_csv_file(updated_path, df)


# #This is where we backtest
# for i, row in df.iterrows():
#     if i > 0:
#         process(df.loc[0:i, :], 36)

fig, ax = plt.subplots()
# x_text = df.loc[0, 'Date']
# y_text = df['Close'].max()
# ax.text(x_text, y_text, 'ROI = {} %, Market Performance = {} %'.format(
# strat_return, market_performance))

# plotting code only considers numeric columns. Pandas uses the matplotlib plot() method, but we should use matplotlib plot_date method

df.plot(x='Date', y='Close', ax=ax)

# set font and rotation for date tick labels
fig.autofmt_xdate()
plt.show()  # this is the timerange to look for local extremes
