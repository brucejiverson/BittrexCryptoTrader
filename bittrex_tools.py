from bittrex.bittrex import *
import json
import pandas as pd
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

# I had a ton of trouble getting the plots to look right with the dates. I eventually figured out to use ___ type date formatting
# This link was really helpful http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html

def process_bittrex_dict(dict):
    # This function converts the dictionaries recieved from bittrex into a
    # Dataframe formatted the same as
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


def original_csv_to_df(path_dict, symbol, desired_granularity, oldest):
    # This is now outdated as all formatting has been standardized to the
    # Format of the csv as originally downloaded.

    # get the historic data
    path = path_dict['Download']

    def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
    df = pd.read_csv(path, parse_dates=['Timestamp'], date_parser=dateparse)

    # Format according to my specifications
    df.rename(columns={'Timestamp': 'Date'}, inplace=True)
    df.drop(columns=["Volume_(" + symbol[0:3] + ')',
                     'Volume_(Currency)', 'Weighted_Price'])
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    df = df[df['Date'] > oldest]
    df = df[df['Close'].notnull()]  # Remove non datapoints from the set
    df.sort_values(by='Date', inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Remove datapoints according to desired des_granularity assume input in minutes

    # This should be updated to save the on the hour rows and be made more effcient
    df = df[df['Date'].dt.minute == 0]
    # print(type(df['Date'].dt.minute))

    return df


def updated_csv_to_df(path_dict, symbol, desired_granularity, oldest):
    # get the historic data
    path = path_dict['Updated']

    def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
    df = pd.read_csv(path, parse_dates=['Date'], date_parser=dateparse)

    # df.drop(columns=["Symbol"])
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    oldest_in_df = df.Date.min()

    if oldest_in_df > oldest:  # need to fetch from original
        orig_df = original_csv_to_df(
            path_dict, symbol, desired_granularity, oldest)
        df = df.append(orig_df)

    df = df[df['Close'].notnull()]  # Remove non datapoints from the set
    df = df[df['Date'] > oldest]
    df.sort_values(by='Date', inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def overwrite_csv_file(path_dict, df):
    # This function writes the information in the original format to the csv file
    # including new datapoints that have been fetched
    path = path_dict['Updated']
    # must create new df as df is passed by reference
    # datetimes to strings
    df['Date'] = df['Date'].dt.strftime("%Y-%m-%d %I-%p")
    df.to_csv(path)
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d %I-%p")


def findCriticalPoints(orig_df, win):
    # this function should take in a historical df, assuming that the most recent
    # date is the most current information, and should identify/update the
    # critical mins and maxes, and then construct top and bottom lines based on those points

    # win is the timerange to look for local extremes in days
    win = win * 24
    f = 0.8
    small_win = timedelta(hours=f * win)
    large_win = timedelta(hours=win)
    cushion = (large_win - small_win) / 2

    # iterrate backwards in time through the df looking for crit points
    # new df is created to help ensure that nothing is changed if df passed by reference
    # sorted so that most recent is index 1
    df = orig_df.sort_values(by='Date', ascending=False)

    # the window is defined with 'date as the largest value in the slice, so backwards in time from date
    # assumes a 1 hr difference between each price
    # note: the way that this works, the indexs need to be preserved
    last_type_found = ''
    mins_num = 0
    maxs_num = 0
    looking_for = 'either'
    points_to_find = 3

    # go until a previous is found
    # Change this so that the object being modified is not the object being iterated over
    for idx, date in df.Date.iteritems():
        # stop if the window extends past the range there is data for
        if date - large_win < df.Date.min():
            break
        # stop if a previous is reached
        elif df.loc[idx, 'Critical'] != '':
            break
        # robust enough to check acutal dates and not just index so that smaller granularities are possible
        s_slice = df[df['Date'] <= date - cushion]
        s_slice = s_slice[s_slice['Date'] >= date - cushion - small_win]
        l_slice = df[df['Date'] >= date - large_win]
        l_slice = l_slice[l_slice['Date'] <= date]

        s_min_idx = s_slice.Close.idxmin()
        s_max_idx = s_slice.Close.idxmax()
        l_min_idx = l_slice.Close.idxmin()
        l_max_idx = l_slice.Close.idxmax()

        # TO DO: skip ahead when one is found
        if s_min_idx == l_min_idx and df.loc[s_min_idx, 'Critical'] == '':
            df.loc[s_min_idx, 'min'] = df.loc[s_min_idx, 'Close']
            mins_num += 1

        if s_max_idx == l_max_idx and df.loc[s_max_idx, 'Critical'] == '':
            df.loc[s_max_idx, 'max'] = df.loc[s_max_idx, 'Close']
            maxs_num += 1

        if mins_num >= points_to_find and maxs_num >= points_to_find:
            break

    # in some places, the critical points are unclear. In those places, the data is usually
    # roughly linear and a solution can be found by refering to a larger scale
    # Construct top and bottom lines

    return df.sort_values(by='Date')


def constructLines(df):
    maxs = df[df.maxs.notnull()]
    mins = df[df.mins.notnull()]

    # function to fit
    def f(x, m, b):
        return m * x + b

    # calculate the lines
    def calcLine(data_frame):
        #data_frame should be mins or maxs df
        covar_limits = [10**2, 10**(-8), 10**(-8), 15**13]
        covar = [2*i for i in covar_limits]

        i = 2 #indexs for the loop
        while all([covar[idx] > covar_limits[idx] for idx in [0, 1, 2, 3]]):
            #Keep adding points
            try:
                number_of_points = data_frame.Date.size
                if  i > number_of_points:
                    break

                p, covar = curve_fit(f, mdates.date2num(mins.Date[:i]), mins.Close[:i])
                covar = np.concatenate((covar[0], covar[1]))

            except IndexError:
                i += 1
                continue
            i += 1

        return p, covar

    min_p, min_cov = calcLine(mins)
    max_p, max_cov = calcLine(maxs)

    lines_data = pd.DataFrame(data={'Date': df.Date})
    lines_data['mins'] = f(mdates.date2num(df.Date), *min_p)
    lines_data['maxs'] = f(mdates.date2num(df.Date), *max_p)
    ps = [min_p, max_p]
    covars = [min_cov, max_cov]
    return ps, covars, lines_data


def plot_market(df, strt, lines=None):

    starting_info = df[df.Date == strt]

    # Create the figure
    fig, ax = plt.subplots()
    x_text = df.loc[0, 'Date']
    y_text = df['Close'].max()

    df.plot(x='Date', y='Close', ax=ax)

    if 'Account Value' in df:
        try:  # In case there are no buys or sells, only an issue while in development
            print('HERE')

            df[df.Trades == 'buy'].plot(x='Date', y='Close', color = 'green', ax=ax)
            df[df.Trades == 'sell'].plot(x='Date', y='Close', color = 'red', ax=ax)
            starting_amnt = starting_info.loc[:, 'Account Value']
            starting_price = starting_info.loc[:, 'Close']
            final_amnt = df['Account Value'].iloc[-1]
            strat_return = ROI(final_amnt, starting_amnt)
            market_performance = ROI(starting_price, df.Close.iloc[-1])
            ax.text(x_text, y_text, 'ROI = {} %, Market Performance = {} %'.format(
                strat_return, market_performance))
        except TypeError:
            print('Error plotting Trades column')
        # Plot lines if any
        if lines is not None:
            lines.plot(x='Date', y='mins', ax=ax)
            lines.plot(x='Date', y='maxs', ax=ax)

        # Plot the critical points if any
        try:
            df[df.mins.notnull()].plot(x='Date', y='Close',
                                       ax=ax, color='orange', style='.')
            df[df.maxs.notnull()].plot(x='Date', y='Close',
                                       ax=ax, color='orange', style='.')

        except TypeError:
            print("Type error plotting critical")
        except AttributeError:
            print('Attribute error plotting critical')

    fig.autofmt_xdate()
    plt.show()
    return fig, ax


def backtest(data, start, end, trdng_prd, fig=0, ax=0):
    # TODO throw an error here if end > start
    #trdng_prd in days

    in_market = False
    money = 100
    fees = 0.003  # standard fees are 0.3% per transaction

    # Ensure that only the necessary dates are in the df
    data = data[data['Date'] >= start - timedelta(hours=trdng_prd)]
    data = data[data['Date'] <= end]
    data.drop_duplicates()
    data.sort_values(by='Date', inplace=True)
    data.reset_index(inplace=True, drop=True)
    data['Account Value'] = ''
    data['Trades'] = ''
    n = 24*3   # number of points to be checked before and after 24*3 is a reasonable filter
    # Find local extrema and filter noise
    data['mins'] = data.iloc[argrelextrema(
        data.Close.values, np.less_equal, order=n)[0]]['Close']
    data['maxs'] = data.iloc[argrelextrema(
        data.Close.values, np.greater_equal, order=n)[0]]['Close']

    df = pd.DataFrame(columns=data.columns)
    df['Account Value'] = money
    df['Trades'] = float('nan')
    coeffs, covars, lines_data  = constructLines(data)
    # print('Covars are: ', covars)
    # plot_market(df, start, lines_data)
    # Bad practice to modify same object you loop over, so 'df' is used
    for i, row in data.iterrows():
        df = df.append(row)
        date = row.Date
        # Check that there are enough crit points to calculate, new loop if not
        maxs_num = df[df.mins.notnull()].Date.count()
        mins_num = df[df.maxs.notnull()].Date.count()
        if maxs_num < 2 or mins_num < 2:
            continue

        # Filter crit points with findCriticalPoints fo flat areas are not over emphasized
        # construct lines
        #min then max
        coeffs, covars, lines_data = constructLines(df[df.Date <= date])
        # strategy_result = strategy(market, coeffs, trdng_prd)
        price = row.Close

        bot_line_val = np.polyval(coeffs[0], mdates.date2num(date))
        top_line_val = np.polyval(coeffs[1], mdates.date2num(date))

        # TODO: add handling of negative ratios
        type = float('nan')
        high_low_rat = (top_line_val - price) / (bot_line_val - price)
        key_val = 0.8
        if high_low_rat > key_val and not in_market:
            # buy
            money *= row.loc['Close'] * (1 - fees)
            type = "buy"
            in_market = True
            print('Buy at: ', row.loc['Close'], 'on: ', date)
        elif high_low_rat < 1 - key_val and in_market:
            # sell
            money = money / row.loc['Close']
            type = "sell"
            in_market = False
            print('Sell at: ', row.loc['Close'], 'on: ', date)
        # else:
        #     strategy_result = 'neutral'

        # Calculate account value in USD
        if in_market:
            df.loc[i, 'Account Value'] = money / row.loc['Close']
        else:
            df.loc[i, 'Account Value'] = money
        df.loc[i, 'Trades'] = type

    #previously had checked for lines_data in locals(), removed as it didnt seem to make a difference
    plot_market(data, start, lines_data)

    account_value = df['Account Value'].iloc[-1]
    print('Account Return: ', ROI(account_value, 100))
    market_start = df.loc[df.Date == start,'Close'].to_numpy()[0]
    market_end = df.loc[df.Date == end,'Close'].to_numpy()[0]
    print('Market Performance: ', ROI(market_end, market_start))

    # return ROI(account_value, 100), account_log


def run():
    pass


def ROI(final, initial):
    #Returns the percentage increase/decrease
    return round(final / initial - 1, 5) * 100
