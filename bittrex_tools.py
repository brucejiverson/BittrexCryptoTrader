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
import statistics as stat


# I had a ton of trouble getting the plots to look right with the dates. I eventually figured out to use ___ type date formatting
# This link was really helpful http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html


# NEED A BETTER WAY TO ALTER GRANULARITY
def get_candles(bittrex_obj, df, market, granularity):
    print('Fetching candles from Bittrex')
    candle_dict = bittrex_obj.get_candles(
        market, granularity)
    if candle_dict['success']:
        # Dataframe formatted the same as
        # V and BV refer to volume and base volume
        new_df = pd.DataFrame(candle_dict['result'])
        new_df.drop(columns=["BV", "V", 'O', 'H', 'L'])
        new_df = new_df.rename(columns={'T': "Date", 'C': 'Close'})

        # reorder the columns (defaults to alphabetic)
        new_df = new_df[['Date', 'Close']]
        new_df.reset_index(inplace=True, drop=True)
        # dates into datetimes
        new_df.Date = pd.to_datetime(new_df.Date, format="%Y-%m-%dT%H:%M:%S")
        print("Success getting candle data")

    else:
        print("Failed to get candle data")

    df = df.append(new_df)
    df = df.drop_duplicates(['Date'])
    df = df.sort_values(by='Date')
    df.reset_index(inplace=True, drop=True)
    return df


def original_csv_to_df(path_dict, desired_granularity, oldest, end):

    print('Fetching historical data from download CSV.')

    # get the historic data
    path = path_dict['Download']

    def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
    df = pd.read_csv(path, usecols=['Timestamp', 'Close'], parse_dates=[
                     'Timestamp'], date_parser=dateparse)
    df.rename(columns={'Timestamp': 'Date'}, inplace=True)

    # cryptodatadownload def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d %I-%p')
    #df = pd.read_csv(path, header = 1, usecols = ['Date', 'Close'], parse_dates=['Date'], date_parser=dateparse)

    # Format according to my specifications

    df = df[['Date', 'Close']]
    df = df[df['Date'] >= oldest]
    df = df[df['Date'] <= end]

    df = df[df['Close'].notnull()]  # Remove non datapoints from the set
    df.sort_values(by='Date', inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Remove datapoints according to desired des_granularity assume input in minutes
    df = df[df['Date'].dt.minute == 0]

    return df


def fetchHistoricalData(path_dict, market, desired_granularity, start_date, end_date, bittrex_obj):
    # this function is useful as code is ran for the same period in backtesting several times consecutively,
    # and fetching from original CSV takes longer as it is a much larger file

    print('Fetching historical data from updated CSV.')

    # get the historic data
    path = path_dict['Updated']

    def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
    df = pd.read_csv(path, parse_dates=['Date'], date_parser=dateparse)

    df = df[['Date', 'Close']]
    start_date_in_df = df.Date.min()

    if start_date_in_df > start_date:  # need to fetch from original
        orig_df = original_csv_to_df(
            path_dict, desired_granularity, start_date, end_date)
        if orig_df.Date.max() >= end_date:
            return orig_df
            end()
        else:
            df = df.append(orig_df)

    if end_date > df.Date.max():  # need to fetch data from Bittrex
        df = get_candles(bittrex_obj, df, market, 'hour')

    df = df[df['Close'].notnull()]  # Remove non datapoints from the set
    df = df[df['Date'] >= start_date]
    df = df[df['Date'] <= end_date]
    df.sort_values(by='Date', inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def overwrite_csv_file(path_dict, df):
    # This function writes the information in the original format to the csv file
    # including new datapoints that have been fetched

    print('Writing data to CSV.')

    path = path_dict['Updated']
    # must create new df as df is passed by reference
    # datetimes to strings
    df['Date'] = df['Date'].dt.strftime("%Y-%m-%d %I-%p")
    df.to_csv(path)
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d %I-%p")


def update_trade_log(path, data):
    df = data[df['type'] != '']
    # This function writes the information in the original format to the csv file
    # including new datapoints that have been fetched

    # must create new df as df is passed by reference
    # datetimes to strings
    df['Date'] = df['Date'].dt.strftime("%Y-%m-%d %I-%p")
    df.to_csv(path)
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d %I-%p")
    print('The trade log has been updated')


def constructLines(df):
    maxs = df[df.maxs.notnull()]
    mins = df[df.mins.notnull()]

    # function to fit
    def f(x, m, b):
        return m * x + b

    # calculate the lines
    def calcLine(data_frame):
        # data_frame should be mins or maxs df
        covar_limits = [10**2, 10**(-8), 10**(-8), 15**13] #minimum limits
        covar = [2 * i for i in covar_limits] #initialize this list as greater than cover_limits
        i = 2  # number of points to include
        while all([covar[idx] > covar_limits[idx] for idx in [0, 1, 2, 3]]):
            # Keep adding points
            size = data_frame.Date.size
            if i > size: #break loop if exceeding the size of the df
                break
            try:
                # This
                p, covar = curve_fit(f, mdates.date2num(
                    data_frame.Date[size - i:]), data_frame.Close[size - i:])
                covar = np.concatenate((covar[0], covar[1])) #join the covar list together into a un nested array
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


def filterCrits(orig_df):
    df = orig_df #need
    #Making this quick and dirty for now
    #ordered dict may be better?
    #index then value

    def line(x, m, b): # function to fit
        return m * x + b

    #going down: current, last, 2 last
    my_mins = {'Indexs':[-1, -1, -1], 'Date': [float('nan'), float('nan'), float('nan')], 'Values': [0, 0, 0]}
    mins_df = pd.DataFrame.from_dict(my_mins)
    maxs_df = mins_df

    for idx, row in orig_df.iterrows():

        if not math.isnan(row.mins):
            #Note: index is preserved with these operations
            mins_df.at[2] = mins_df.iloc[1]
            mins_df.at[1] = mins_df.iloc[0]
            mins_df.loc[0, 'Indexs'] = idx
            mins_df.loc[0, 'Values'] = row.mins
            mins_df.loc[0, 'Date'] = row.Date

        else:
            continue
        if all([x != 0 for x in mins_df['Values']]): #have found at least three mins
            print(mins_df)
            p, covar = curve_fit(line, mdates.date2num(mins_df.Date), mins_df.Values)
            value =  line(mdates.date2num(mins_df.loc[1, 'Date']), *p)

            if value > mins_df.loc[0, 'Values'] and value > mins_df.loc[2, 'Values']:
                orig_df.at[mins_df.loc[1, 'Indexs'], 'mins'] = float('nan')
                print('Filtered min',mins_df.loc[1, 'Date'])

        # if all([x != 0 for x[1] in my_maxs.values()]): #have found at least two maxs
        #         if my_maxs['current'][1] < my_maxs['last1'][1] and my_maxs['current'][1] < my_maxs['last2'][1]:
        #             df.at[my_maxs['last1'][0], 'maxs'] = float('nan')

    return orig_df


def plot_market(df, strt, roi=0, market_performance=0, lines=None):
    # Create the figure
    fig, ax = plt.subplots()

    df.plot(x='Date', y='Close', ax=ax)

    if 'Account Value' in df:
        # Plot the critical points if any

        df[df.mins.notnull()].plot(x='Date', y='Close',
                                   ax=ax, color='orange', style='.')
        df[df.maxs.notnull()].plot(x='Date', y='Close',
                                   ax=ax, color='orange', style='.')
        try:
            df[df.Trades == 'Buy'].plot(
                x='Date', y='Close', color='green', ax=ax, style='.', marker='o')
        except TypeError:
            print("No Buys.")
        try:
            df[df.Trades == 'Sell'].plot(
                x='Date', y='Close', color='red', ax=ax, style='.', marker='o')
        except TypeError:
            print("No sells.")
        plt.axvline(x = strt)
        x_text = df.loc[0, 'Date']
        y_text = df['Close'].max()
        ax.text(x_text, y_text, 'ROI = {} %, Market Performance = {} %'.format(
            roi, market_performance))
        # print('Error plotting Trades column')
        # print('Zero Division Error calculating return')
        # Plot lines if any
        if lines is not None:
            lines.plot(x='Date', y='mins', ax=ax)
            lines.plot(x='Date', y='maxs', ax=ax)

        # except TypeError:
        #     print("Type error plotting critical")
        # except AttributeError:
        #     print('Attribute error plotting critical')
    df.plot(x='Date', y='Account Value', ax=ax)
    bot, top = plt.ylim()
    plt.ylim(0, top)
    fig.autofmt_xdate()
    plt.show()


def test_trade(money, in_market, trade, price, date, fees, sig_reason):
    if in_market == False:
        money = (money / price) * (1 - fees)
        trade = "Buy"
        print('Buy  at: ', round(price, 1), 'on: ', date, sig_reason)
    else:
        money *= price
        trade = "Sell"
        print('Sell at: ', round(price, 1), 'on: ', date, sig_reason)

    in_market = not in_market
    return money, in_market, trade


def strategy(df, date, price, last_min_num, last_max_num):
    # probably dont have to pass price, can calculate based on given date
    # Check that there are enough crit points to calculate, new loop if not (this is for initial cases)
    mins_num = df[df.mins.notnull()].Date.size
    maxs_num = df[df.maxs.notnull()].Date.size
    if maxs_num < 2 or mins_num < 2:
        last_min_num = mins_num
        last_max_num = maxs_num
        return 'insufficient data', 'none', last_max_num, last_min_num
        end()


    elif mins_num == last_min_num and maxs_num == last_max_num:
        return 'no new data', 'none', last_max_num, last_min_num
        end()

    # Calculate the lines if there are new values
    coeffs, covars, lines_data = constructLines(df)  # min then max
    last_max_num = maxs_num
    last_min_num = mins_num

    # Filter crit points with findCriticalPoints fo flat areas are not over emphasized

    bot_line_val = np.polyval(coeffs[0], mdates.date2num(date))
    top_line_val = np.polyval(coeffs[1], mdates.date2num(date))
    key_val = 0.8

    sell_price = bot_line_val + key_val * (top_line_val - bot_line_val)
    buy_price = bot_line_val + (1 - key_val) * (top_line_val -
                                 bot_line_val)
    min_slope = coeffs[0][0]
    max_slope = coeffs[1][0]

    signal = 'neutral'
    slope_lim = 35
    mean_slope = stat.mean([max_slope, min_slope])

    signal_reason = 'normal'
    # Place below so that subseque  ntasdfterms supercede/overwrite
    #Mixed signals are occuring, buys are closing and sells are normal

    # Normal buy and sell
    if price > sell_price:
        # Normal sell
        signal = 'bearish'
    elif price < buy_price:
        # Normal buy
        signal = 'bullish'

    # if min_slope > max_slope + 5:
    #     #How is this different than normal?
    #     cushion = .05  # percentage
    #     # TODO account for if the lines have fully crossed
    #     local_sell_price = bot_line_val - \
    #         cushion * (top_line_val - bot_line_val)
    #     local_buy_price = bot_line_val + \
    #         (1 + cushion) * (top_line_val - bot_line_val)
    #     if price > local_buy_price:
    #         signal = 'bullish'
    #         signal_reason = 'closing triangle'
    #
    #     elif price < local_sell_price:
    #         signal = 'bearish'
    #         signal_reason = 'closing triangle'

    if bot_line_val > top_line_val - 0.02*(top_line_val -bot_line_val) :

        signal = 'neutral'
        signal_reason = 'lines are really close!'

    #If slope_lim is dominating, the local prices are to detect trend reversal
    #cant make this larger than the lines cause new crit points would be added
    slope_lim_key = .05
    local_sell_price = bot_line_val + slope_lim_key  * (top_line_val -
                                 bot_line_val)
    local_buy_price = bot_line_val + (1 + key_val) * (top_line_val - bot_line_val)

    if mean_slope > slope_lim:

        # if price < local_sell_price:
        #     signal_reason = 'slope_lim trend reversal'
        #     signal = 'bearish'
        # else:
        #     signal_reason = 'slope_lim'
        #     signal = 'bullish'
      signal_reason = 'slope_lim'
      signal = 'bullish'
    elif mean_slope < -slope_lim:
        # if price > local_buy_price:
        #     signal_reason = 'slope_lim trend reversal'
        #     signal = 'bullish'
        # else:
        signal_reason = 'slope_lim'
        signal = 'bearish'

    return signal, signal_reason, last_max_num, last_min_num


def backtest(data, start, end, extrema_filter, experiment=False, fig=0, ax=0):
    # TODO throw an error here if end > start

    print('Beginning backtesting.')

    in_market = False
    init_money = 100
    money = init_money
    fees = 0.0025  # standard fees are 0.3% per transaction

    data['Account Value'] = 0
    data['Trades'] = ''

    df = pd.DataFrame(columns=data.columns)
    # Bad practice to modify same object you loop over, so 'df' is used
    last_min_num = 0
    last_max_num = 0
    market_start = 0

    for i, row in data.iterrows():

        df = df.append(row)
        date = row.Date
        price = row.Close
        if date < start:
            continue
        elif date >= start and market_start == 0:
           market_start = price

        # Find local extrema and filter noise
        df['mins'] = df.iloc[argrelextrema(
            df.Close.values, np.less_equal, order=extrema_filter)[0]]['Close']
        df['maxs'] = df.iloc[argrelextrema(
            df.Close.values, np.greater_equal, order=extrema_filter)[0]]['Close']
        # df = filterCrits(df)

        trade = ''

        signal, sig_reason, last_max_num, last_min_num = strategy(
            df, date, price, last_min_num, last_max_num)

        # Trade based on signal
        if signal == 'bullish' and not in_market:
            [money, in_market, trade] = test_trade(
                money, in_market, trade, price, date, fees, sig_reason)
        elif signal == 'bearish' and in_market:
            [money, in_market, trade] = test_trade(
                money, in_market, trade, price, date, fees, sig_reason)
        # Calculate account value in USD
        if in_market:
            account_value = money * row.loc['Close']
        else:
            account_value = money

        df.at[i, 'Account Value'] = account_value
        df.at[i, 'Trades'] = trade

    #Note: can access by index
    roi = ROI(account_value, init_money)
    print('Account Return: ', roi, ' %')

    try:
        market_end = df.loc[df.Date == end, 'Close'].to_numpy()[0]
        market_performance = ROI(market_end, market_start)
        print('Market Performance: ', market_performance, ' %')
    except IndexError:
        print('No date in DF == to end')

    if experiment == False:  # only want to plot if not experimenting to prevent 100s of plots
        try:

            plot_market(df, start, roi, market_performance)
        except UnboundLocalError:  # i think this was for plotting lines
            plot_market(df, start)
    return roi


def testRun(df, bittrex_obj, extrema_filter, path_dict):
    # maybe it would be helpful to run this through command line argv etc

    # every hour
        # check if it should buy or not
        # execute Trade if ...
        # update the trade log
        # drop the old and unnecessary datapoints

    in_market = False
    account_info = bittrex_obj.get_balance('BTC')
    money = account_info['result']['Balance']
    fees = 0.0025  # standard fees are 0.3% per transaction

    df['Account Value'] = 0
    df['Trades'] = ''

    last_min_num = 0
    last_max_num = 0
    first_check = True

    while True:

        candle_dict = bittrex_obj.get_candles('USD-BTC', 'hour')
        if candle_dict['success']:
            new_data = process_bittrex_dict(candle_dict)
        else:
            print("Failed to get candle data")
            continue

        df = df.append(new_df)
        df = df.drop_duplicates(['Date'])
        df = df.sort_values(by='Date')
        df.reset_index(inplace=True, drop=True)

        # Find local extrema and filter noise
        data['mins'] = data.iloc[argrelextrema(
            data.Close.values, np.less_equal, order=extrema_filter)[0]]['Close']
        data['maxs'] = data.iloc[argrelextrema(
            data.Close.values, np.greater_equal, order=extrema_filter)[0]]['Close']
        # data = filterCrits(data)


        if first_check:
            market_start = price
            first_check = False

        # Find local extrema and filter noise, for now this is done everytime but eventually should be more efficient
        df['mins'] = df.iloc[argrelextrema(
            df.Close.values, np.less_equal, order=extrema_filter)[0]]['Close']
        df['maxs'] = df.iloc[argrelextrema(
            df.Close.values, np.greater_equal, order=extrema_filter)[0]]['Close']

        trade = ''

        # def buy_limit(self, market, quantity, rate):
        # """
        # Used to place a buy order in a specific market. Use buylimit to place
        # limit orders Make sure you have the proper permissions set on your
        # API keys for this call to work







        signal = strategy(df, date, price, last_min_num, last_max_num)

        # Trade based on signal
        if signal == 'bullish' and not in_market:
            [money, in_market, trade] = test_trade(
                money, in_market, trade, price, date, fees)
        elif signal == 'bearish' and in_market:
            [money, in_market, trade] = test_trade(
                money, in_market, trade, price, date, fees)

        # Calculate account value in USD
        if in_market:
            account_value = money * row.loc['Close']
        else:
            account_value = money

        df.at[i, 'Account Value'] = account_value
        df.at[i, 'Trades'] = trade

        # This allows for accessing by index
        # df.reset_index(inplace=True, drop=True) necessary for plotting

        roi = ROI(account_value, 100)
        print('Account Return: ', roi, ' %')
        market_performance = ROI(price, market_start)
        print('Market Performance: ', market_performance, ' %')

        time.sleep(60 * 60)  # in seconds. wait an hour


def run():
    pass


def ROI(final, initial):
    # Returns the percentage increase/decrease
    return round(final / initial - 1, 2) * 100
