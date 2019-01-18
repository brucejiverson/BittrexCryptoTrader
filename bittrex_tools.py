
from bittrex.bittrex import *
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.optimize import curve_fit
import sys
sys.path.append("C:\Python Programs\CryptoTrader\CryptoTraderStrategy")
from helios import *
# I had a ton of trouble getting the plots to look right with the dates. I eventually figured out to use ___ type date formatting
# This link was really helpful http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html

def process_bittrex_dict(dict):
    # V and BV refer to volume and base volume
    df = pd.DataFrame(dict['result'])
    df.drop(columns=["BV", "V"])
    df = df.rename(columns={'T': "Date", 'O': 'Open',
                            'H': 'High', 'L': 'Low', 'C': 'Close'})

    # reorder the columns (defaults to alphabetic)
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    df.reset_index(inplace=True, drop=True)
    # dates into datetimes
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%dT%H:%M:%S")
    return df


def original_csv_to_df(path_dict, oldest_month, oldest_day, oldest_year):
    # get the historic data
    path = path_dict['Download']

    def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
    df = pd.read_csv(path, header=1, parse_dates=[
                     'Date'], date_parser=dateparse)

    df.drop(columns=["Symbol", "Volume From", "Volume To"])
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]

    oldest = datetime(day=oldest_day, month=oldest_month, year=oldest_year)
    df = df[df['Date'] > oldest]
    df.sort_values(by='Date', inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def updated_csv_to_df(path_dict, oldest_month, oldest_day, oldest_year):
    # get the historic data
    path = path_dict['Updated']

    def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
    df = pd.read_csv(path, parse_dates=['Date'], date_parser=dateparse)

    # df.drop(columns=["Symbol"])
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    oldest = datetime(day=oldest_day, month=oldest_month, year=oldest_year)
    oldest_in_df = df.Date.min()

    if oldest_in_df > oldest:  # need to fetch from original
        orig_df = original_csv_to_df(
            path_dict, oldest_month, oldest_day, oldest_year)
        df = df.append(orig_df)

    df = df[df['Date'] > oldest]
    df.sort_values(by='Date', inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def overwrite_csv_file(path, df):
    # must create new df as df is passed by reference
    # datetimes to strings
    for i, row in df.iterrows():
        date = row.loc['Date']
        df.loc[i, 'Date'] = date.strftime("%Y-%m-%d %I-%p")
    df.to_csv(path)
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d %I-%p")


def plot_market(df, strt, lines=None):
    starting_info = df[df.Date == strt]
    starting_amnt = starting_info.loc[:,'Account Value']
    starting_price = starting_info.loc[:,'Close']
    final_amnt = df['Account Value'].iloc[-1]
    strat_return = ROI(final_amnt, starting_amnt)
    market_performance = ROI(starting_price, df.Close.iloc[-1])

    fig, ax = plt.subplots()
    x_text = df.loc[0, 'Date']
    y_text = df['Close'].max()
    ax.text(x_text, y_text, 'ROI = {} %, Market Performance = {} %'.format(
        strat_return, market_performance))

    df.plot(x='Date', y='Close', ax=ax)
    try:
        df[df['Critical'] != ''].plot(
            x='Date', y='Close', ax=ax, color='red', style='.')
    except TypeError:
        print("No critical points to plot")

    if lines is not None:
        lines.plot(x='Date', y='mins', ax=ax)
        lines.plot(x='Date', y='maxs', ax=ax)

    if 'Account Value' in df:
        df[df.Type == 'buy'].plot(x='Date', y='Close', ax=ax)
        df[df.Type == 'sell'].plot(x='Date', y='Close', ax=ax)

    fig.autofmt_xdate()
    plt.show()


def constructLines(df):
    maxs = df[df.Critical == 'cmax']
    mins = df[df.Critical == 'cmin']

    # function to fit
    def f(x, m, b):
        return m * x + b

    # calculate the lines
    min_p, min_cov = curve_fit(f, mdates.date2num(mins.Date), mins.Close)
    max_p, max_cov = curve_fit(f, mdates.date2num(maxs.Date), maxs.Close)

    # min_p = np.polyfit(mins.Date, mins.Close, 1)
    # max_p = np.polyfit(maxs.Date, maxs.Close, 1)

    lines_data = pd.DataFrame(data={'Date': df.Date})
    lines_data['mins'] = f(mdates.date2num(df.Date), *min_p)
    lines_data['maxs'] = f(mdates.date2num(df.Date), *max_p)

    # for i, x in enumerate(xs):
    #     min = min_p[0] * x + min_p[1]
    #     max = max_p[0] * x + max_p[1]
    #     lines_data.loc[i, 'mins'] = min
    #     lines_data.loc[i, 'maxs'] = max

    return [min_p, max_p], lines_data


def backtest(df, start, end, trdng_prd):
    # TODO throw an error here if end > start

    #trdng_prd in days

    # account_log = pd.DataFrame(
    #     columns=['Date', 'Account Value (USD)', 'Trades'])

    in_market = False
    money = 100
    fees = 0.003  # standard fees are 0.3% per transaction
    fees = 1 - fees

    df['Critical'] = ''
    df['Account Value'] = money
    df['Trades'] = ''

    # for each data point from start to finish, check the strategy and calculate the money
    df = df[df['Date'] >= start - timedelta(hours=trdng_prd)] #TODO
    df = df[df['Date'] <= end]
    df.sort_values(by='Date', inplace=True)
    df.reset_index(inplace=True, drop=True)

    date = start

    # df = findCriticalPoints(df, trdng_prd)
    #
    # coeffs, line_data = constructLines(df)
    # TO DO: eliminate the need for the market df. need to increase efficiency.
    # While it takes more memory and is probably slower, a new df is created
    # (market) in order to ensure the strategy is operating on only the information
    # it should be
    market = pd.DataFrame(columns=df.columns)
    # df is arranged oldest to newest
    for i, row in df.iterrows():
        market.append(row)
        if row.Date <= start + timedelta(hours=trdng_prd * 24):
            continue

        market = findCriticalPoints(market, trdng_prd)
        # Check that there are enough critical points to fit lines
        maxs_num = market[market.Critical == 'cmax'].Date.count()
        mins_num = market[market.Critical == 'cmin'].Date.count()
        if maxs_num < 2 or mins_num < 2:
            continue

        coeffs, line_data = constructLines(market)
        strategy_result = strategy(market, coeffs, trdng_prd)

        type = ''
        if (strategy_result == "bullish" and not in_market):
            # buy
            money *= row.loc['Close'] * fees
            type = "buy"
            print('Buy at: ', row.loc['Close'], 'on: ', row.loc['Date'])

        elif (strategy_result == "bearish" and in_market):
            # sell
            money = money / row.loc['Close']
            type = "sell"
            print('Sell at: ', row.loc['Close'], 'on: ', row.loc['Date'])

        if in_market:
            market.loc[i, 'Account Value'] = money / row.loc['Close']
        else:
            market.loc[i, 'Account Value'] = money

        market.loc[i, 'Trades'] = type

    #     # update log w date from row, account value, and if a trade was executed
    #     account_log.append(pd.DataFrame(df=[row.loc['date'], account_value, type], columns=[
    #                        'Date', 'Account Value (USD)', 'Trades']), ignore_index=True)
    if "line_data" in locals():
        plot_market(market, start, line_data)
    else:
        plot_market(market, start)

    # return ROI(account_value, 100), account_log


def run():
    pass


def ROI(final, initial):
    return round(final / initial - 1, 5) * 100
