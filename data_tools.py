import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from datetime import datetime, timedelta
import itertools
from empyrical import sortino_ratio
import ta

from statistics import mean


def format_df(input_df):
    # input_df = input_df[['Date', 'BTCClose']]
    input_df.drop_duplicates(subset='Date', inplace=True)
    input_df = input_df[input_df['BTCClose'].notnull()]  # Remove non datapoints from the set
    input_df.sort_values(by='Date', inplace=True)
    input_df.reset_index(inplace=True, drop=True)
    return input_df


def fetch_historical_data(path_dict, market, start_date, end_date, bittrex_obj):
    # this function is useful as code is ran for the same period in backtesting several times consecutively,
    # and fetching from original CSV takes longer as it is a much larger file

    print('Fetching historical data...')

    # get the historic data
    path = path_dict['updated history']

    df = pd.DataFrame(columns=['Date', 'BTCOpen', 'BTCHigh', 'BTCLow', 'BTCClose', 'BTCVolume'])

    # Fetch candle data from bittrex
    if end_date > datetime.now() - timedelta(days=10):
        candle_df = get_candles(bittrex_obj, df, market, 'oneMin')
        df = df.append(candle_df)

    if df.empty or df.Date.min() > start_date:  # try to fetch from updated
        print('Fetching data from cumulative data repository.')

        def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p-%M")
        up_df = pd.read_csv(path, parse_dates=['Date'], date_parser=dateparse)

        if up_df.empty:
            print('Cumulative data repository was empty.')
        else:
            print('Success fetching from cumulative data repository.')
            df = df.append(up_df)

    if df.empty or df.Date.min() > start_date:  # Fetch from download file (this is last because its slow)

        print('Fetching data from the download file.')
        # get the historic data. Columns are Timestamp	Open	High	Low	Close	Volume_(BTC)	Volume_(Currency)	Weighted_Price

        def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
        orig_df = pd.read_csv(path_dict['downloaded history'], usecols=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(Currency)'], parse_dates=[
            'Timestamp'], date_parser=dateparse)
        orig_df.rename(columns={'Timestamp': 'Date', 'O': 'BTCOpen', 'H': 'BTCHigh',
                                'L': 'BTCLow', 'C': 'BTCClose', 'V': 'BTCVolume'}, inplace=True)

        assert not orig_df.empty

        df = df.append(orig_df)

    # Double check that we have a correct date date range. Note: will still be triggered if missing the exact data point
    assert(df.Date.min() <= start_date)

    # if df.Date.max() < end_date:
    #     print('There is a gap between the download data and the data available from Bittrex. Please update download data.')
    #     assert(df.Date.max() >= end_date)

    df = df[df['Date'] >= start_date]
    df = df[df['Date'] <= end_date]
    df = format_df(df)

    return df


def filter_error_from_download_data(input_df):

    print('Filtering data for errors...')
    for i, row in input_df.iterrows():
        if i > 0 and i < len(input_df.Date) - 2:
            try:
                if input_df.loc[i, 'BTCClose'] < 0.5 * mean([input_df.loc[i - 1, 'BTCClose'], input_df.loc[i + 1, 'BTCClose']]):
                    input_df.drop(i, axis=0, inplace=True)
                    print('Filtered a critical point.')
            except KeyError:
                print(i, len(input_df.Date))
    input_df = format_df(input_df)
    return input_df


def strip_open_high_low(input_df):

    df_cols = input_df.columns

    # Structured to be currency agnostic
    for col in df_cols:
        if 'Open' in col or 'High' in col or 'Low' in col:
            input_df.drop(columns=[col])

    return input_df


def save_historical_data(path_dict, df):
    # This function writes the information in the original format to the csv file
    # including new datapoints that have been fetched

    print('Writing data to CSV.')

    path = path_dict['updated history']
    # must create new df as df is passed by reference
    # # datetimes to strings
    # df = pd.DataFrame({'Date': data[:, 0], 'BTCClose': np.float_(data[:, 1])})   #convert from numpy array to df

    def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p-%M")
    old_df = pd.read_csv(path, parse_dates=['Date'], date_parser=dateparse)

    df_to_save = df.append(old_df)

    df_to_save = format_df(df_to_save)

    # df_to_save = filter_error_from_download_data(df_to_save)

    df_to_save['Date'] = df_to_save['Date'].dt.strftime("%Y-%m-%d %I-%p-%M")
    df_to_save.to_csv(path, index=False)

    # df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d %I-%p-%M")               # added this so it doesnt change if passed by object... might be wrong but appears to make a difference. Still dont have a great grasp on pass by obj ref.``
    print('Data written.')


def add_sma_as_column(df, p):
    # p is a number
    price = df['BTCClose'].values  # returns an np price

    sma = np.empty_like(price)
    for i, item in enumerate(np.nditer(price)):
        if i == 0:
            sma[i] = item
        elif i < p:
            sma[i] = price[0:i].mean()
        else:
            sma[i] = price[i - p:i].mean()

    # subtract
    indicator = np.empty_like(sma)
    for i, item in enumerate(np.nditer(price)):
        indicator[i] = sma[i] - price[i]

    df['SMA_' + str(p)] = indicator  # modifies the input df

# def add_features(df):

    # add the momentum indicators


def make_price_history_static(df):
    df['Static'] = np.log(df['BTCClose']) - np.log(df['BTCClose']).shift(1)
    df['Static'].iloc[0] = 0
    # df['Static'] = df['BTCClose']
    df = df[['Date', 'BTCClose', 'Static']]


def plot_data(df):
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


def plot_sim_trade_history(df, log, roi=0):
    # df = pd.DataFrame({'Date': data[:, 0], 'BTCClose': np.float_(data[:, 1])})

    assert not df.empty
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)  # Create the figure

    market_perf = ROI(df.BTCClose.iloc[0], df.BTCClose.iloc[-1])
    fig.suptitle('Market performance: ' + str(market_perf), fontsize=14, fontweight='bold')
    df.plot(x='Date', y='BTCClose', ax=ax1)

    for col in df.columns:
        if col[0:3] == 'SMA':
            df.plot(x='Date', y=col, ax=ax3)

    # df.plot(x='Date', y='Account Value', ax=ax)

    log['Date'] = df.Date
    my_roi = ROI(log.Value.iloc[0], log.Value.iloc[-1])

    log.plot(x='Date', y='Value', ax=ax2)
    # df.plot(x='Date', y='AccounV Value', ax=ax)

    bot, top = plt.ylim()
    cushion = 200
    plt.ylim(bot - cushion, top + cushion)
    fig.autofmt_xdate()
    plt.show()


def process_trade_history(dict):
    # Example input: {'success': True, 'message': '', 'result': [{'OrderUuid': '3d87588d-70d6-4b40-a723-11248aaaff8b', 'Exchange': 'USD-BTC', 'TimeStamp': '2019-11-19T07:42:48.85', 'OrderType': 'LIMIT_SELL', 'Limit': 1.3, 'Quantity': 0.00123173, 'QuantityRemaining': 0.0, 'Commission': 0.02498345, 'Price': 9.99338392, 'PricePerUnit': 8113.29099722, 'IsConditional': False, 'Condition': '', 'ConditionTarget': 0.0, 'ImmediateOrCancel': False, 'Closed': '2019-11-19T07:42:48.85'}]}

    trade_df = pd.DataFrame(dict['result'])
    trade_df.drop(columns=['IsConditional', 'Condition', 'ConditionTarget',
                           'ImmediateOrCancel', 'Closed'], inplace=True)

    trade_df.reset_index(inplace=True, drop=True)
    # dates into datetimess
    trade_df.TimeStamp = pd.to_datetime(trade_df.TimeStamp, format="%Y-%m-%dT%H:%M:%S")
    # trade_df.Closed = pd.to_datetime(trade_df.Closed, format="%Y-%m-%dT%H:%M:%S")
    return trade_df
