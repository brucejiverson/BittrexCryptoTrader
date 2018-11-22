from bittrex.bittrex import *
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#I had a ton of trouble getting the plots to look right with the dates. I eventually figured out to use ___ type date formatting
#This link was really helpful http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html

def process_bittrex_dict(dict):
    # V and BV refer to volume and base volume
    df = pd.DataFrame(dict['result'])
    df.drop(columns=["BV", "V"])
    df = df.rename(columns={'T': "Date", 'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close'})

    # reorder the columns (defaults to alphabetic)
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    df.reset_index(inplace = True, drop = True)
    # dates into datetimes
    df.Date = pd.to_datetime(df.Date, format = "%Y-%m-%dT%H:%M:%S")
    return df


def original_csv_to_df(path, oldest_month, oldest_day, oldest_year):
    # get the historic data
    dateparse = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
    df = pd.read_csv(path, header=1, parse_dates = ['Date'], date_parser = dateparse)

    df.drop(columns=["Symbol", "Volume From", "Volume To"])
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]

    oldest = datetime(day=oldest_day, month=oldest_month, year=oldest_year)
    df = df[df['Date'] > oldest]
    df.sort_values(by='Date', inplace=True)
    df.reset_index(inplace = True, drop = True)
    return df


def updated_csv_to_df(path, oldest_month, oldest_day, oldest_year):
    # get the historic data
    dateparse = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
    df = pd.read_csv(path, parse_dates = ['Date'], date_parser = dateparse)

    # df.drop(columns=["Symbol"])
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]

    oldest = datetime(day=oldest_day, month=oldest_month, year=oldest_year)
    df = df[df['Date'] > oldest]
    df.sort_values(by='Date', inplace=True)
    df.reset_index(inplace = True, drop = True)
    return df


def overwrite_csv_file(path, df):
    #must create new df as df is passed by reference
    # datetimes to strings
    for i, row in df.iterrows():
        date = row.loc['Date']
        df.loc[i, 'Date'] = date.strftime("%Y-%m-%d %I-%p")
    df.to_csv(path)
    df.Date = pd.to_datetime(df.Date, format = "%Y-%m-%d %I-%p")


def plot_market(df):
    fig, ax = plt.subplots()
    x_text = df.loc[0, 'Date']
    y_text = df['Close'].max()
    # ax.text(x_text, y_text, 'ROI = {} %, Market Performance = {} %'.format(
    # strat_return, market_performance))

    df.plot(x='Date', y='Close', ax=ax)
    fig.autofmt_xdate()
    plt.show()  # this is the timerange to look for local extremes


def process(df, win):
    #this function should take in a historical df, assuming that the most recent
    #date is the most current information, and should identify/update the
    #critical mins and maxes, and then construct top and bottom lines based on those points

    # window is the timerange to look for local extremes in hours
    f = 0.7
    small_win = timedelta(hours=win)
    large_win = timedelta(hours = win/f)
    cushion = (large_win - small_win)/2


    #iterrate backwards in time through the df looking for crit points
    df.sort_values(by = 'Date', inplace = True, ascending = False)

    #go through every date... this shouldnt be necessary, should only have to go through a few
    #the window is defined with 'date as the largest value in the slice, so backwards in time from date
    #assumes a 1 hr
    for idx, date in df.Date.iteritems():
        #stop if the window extends past the range there is data for
        if date - large_win < df.Date.min():
            break
        #robust enough to check acutal dates and not just index so that smaller granularities are possible
        small_slice =
        large_slice = df[df['Date'] > date - large_win].loc[i::]
        is_critmax =
        is_critmin =
        if is_critmax:

        elif is_critmin:



    # define the convention: lmin, lmax, cmin, cmax
    # find the most recent critical max/min
    # try:
    #     most_recent_date = df.loc[df['Critical'] != '' ].iloc[0]
    # except IndexError:
    #     #there is no data, this is the initialization case
    #     #identify all local min and maxes
    #     most_recent_date = df.iloc[0]['Date'] #why do i have this again...
    #     df[df[]]

    # df.loc[df['Date'] >= most_recent_date, ]
    # for i, row in reversed(df.iterrows()):
    #     if (row):
    #         pass
    # #identitfy new local mins/maxes

    # filter all mins/maxes for critical points

    # identify any new local mins/maxes


def backtest(df, start, end, desired_trading_period):

    df['Critical'] = ''
    account_log = pd.DataFrame(
        columns=['Date', 'Account Value (USD)', 'Trades'])

    in_market = False
    money = 100
    fees = 0.003  # standard fees are 0.3% per transaction
    # for each data point from start to finish, check the strategy and calculate the money
    date = start
    # while date <= end:
    #     # get the next data point and append it to the data frame
    #     date = min(start + timedelta(seconds = data.granularity), end)
    #     data.new_datapoint(date)
    #
    #     type = ''
    #     strategy_result = strategy(data, desired_trading_period)
    #     if (strategy_result == "bullish" and not in_market):
    #         # buy
    #         # log trade
    #         money *= slice.loc['close']
    #         type = "buy"
    #     elif (strategy_result == "bearish" and in_market):
    #         # sell
    #         # log trade
    #         money = money / slice.loc['close']
    #         type = "sell"
    #         pass
    #
    #     # determine the account value
    #     if in_market:
    #         account_value = money / slice.loc['close']
    #     else:
    #         account_value = money
    #     # update log w date from row, account value, and if a trade was executed
    #     account_log.append(pd.DataFrame(data=[slice.loc['date'], account_value, type], columns=[
    #                        'Date', 'Account Value (USD)', 'Trades']), ignore_index=True)

    # plot the account value

    #return ROI(account_value, 100), account_log


def run():
    pass


def ROI(final, initial):
    return round(final / initial - 1, 5) * 100
