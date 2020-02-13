from bittrex.bittrex import *
# from data_tools import *

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import argparse

from datetime import datetime, timedelta
import json
import re
import os
import pickle
import itertools
import ta
import math

from statistics import mean

from sklearn.preprocessing import StandardScaler
import warnings

#Set warnings to be caught as errors


# I had a ton of trouble getting the plots to look right with the dates.
# This link was really helpful http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'I made a directory at {directory}')


def process_candle_dict(candle_dictionary):
    # Dataframe formatted the same as
    # V and BV refer to volume and base volume
    df = pd.DataFrame(candle_dictionary['result'])
    df.drop(columns=["BV", "V", 'O', 'H', 'L'])
    df = df.rename(columns={'T': "Date", 'O': 'BTCOpen', 'H': 'BTCHigh', 'L': 'BTCLow', 'C': 'BTCClose', 'V': 'BTCVolume'})

    df = df[['Date', 'BTCOpen', 'BTCHigh', 'BTCLow', 'BTCClose', 'BTCVolume']]  #Reorder
    df.reset_index(inplace=True, drop=True)
    # dates into datetimess
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%dT%H:%M:%S")
    return df


def ROI(initial, final):
    # Returns the percentage increase/decrease
    return round(final / initial - 1, 4) * 100


def process_order_data(dict):
    # Example input: {'success': True, 'message': '', 'result': {'AccountId': None, 'OrderUuid': '3d87588d-70d6-4b40-a723-11248aaaff8b', 'Exchange': 'USD-BTC', 'Type': 'LIMIT_SELL', 'Quantity': 0.00123173, 'QuantityRemaining': 0.0, 'Limit': 1.3, 'Reserved': None, 'ReserveRemaining': None, 'CommissionReserved': None, 'CommissionReserveRemaining': None, 'CommissionPaid': 0.02498345, 'Price': 9.99338392, 'PricePerUnit': 8113.29099722, 'Opened': '2019-11-19T07:42:48.85', 'Closed': '2019-11-19T07:42:48.85', 'IsOpen': False, 'Sentinel': None, 'CancelInitiated': False, 'ImmediateOrCancel': False, 'IsConditional': False, 'Condition': 'NONE', 'ConditionTarget': 0.0}}

    # in order to construct a df, the values of the dict cannot be scalars, must be lists, so convert to lists
    results = {}
    for key in dict['result']:
        results[key] = [dict['result'][key]]
    order_df = pd.DataFrame(results)

    order_df.drop(columns=['AccountId', 'Reserved', 'ReserveRemaining', 'CommissionReserved', 'CommissionReserveRemaining',
                           'Sentinel', 'IsConditional', 'Condition', 'ConditionTarget', 'ImmediateOrCancel', 'CancelInitiated'], inplace=True)

    order_df.reset_index(inplace=True, drop=True)
    # dates into datetimess
    order_df.Closed = pd.to_datetime(order_df.Closed, format="%Y-%m-%dT%H:%M:%S")
    order_df.Opened = pd.to_datetime(order_df.Opened, format="%Y-%m-%dT%H:%M:%S")

    return order_df


def save_trade_data(trade_df, path_dict):
    save_path = path_dict['test trade log']

    try:
        def dateparse(x):
            try:
                return pd.datetime.strptime(x, "%Y-%m-%d %I-%p-%M")
            except ValueError:  #handles cases for incomplete trades where 'Closed' is NaT
                return datetime(year = 2000, month = 1, day = 1)

        old_df = pd.read_csv(save_path, parse_dates=['Opened', 'Closed'], date_parser=dateparse)

        trade_df = trade_df.append(old_df)
        trade_df.sort_values(by='Opened', inplace=True, sort = True)
        trade_df.reset_index(inplace=True, drop=True)

        trade_df['Closed'] = trade_df['Closed'].dt.strftime("%Y-%m-%d %I-%p-%M")
        trade_df['Opened'] = trade_df['Opened'].dt.strftime("%Y-%m-%d %I-%p-%M")

        trade_df.to_csv(save_path, index=False)
        print('Data written to test trade log.')

    except KeyError:
        print('Order log is empty.')


def format_df(input_df):
    #Note that this should only be used before high low open are stripped from the data
    # input_df = input_df[['Date', 'BTCClose']]
    input_df = input_df[input_df.BTCClose.notnull()]  # Remove non datapoints from the set
    input_df = input_df[input_df.BTCOpen.notnull()]  # Remove non datapoints from the set
    input_df.drop_duplicates(subset='Date', inplace=True)
    input_df.sort_values(by='Date', inplace=True)   #This was causing a warning about future deprecation/changes to pandas

    input_df.drop_duplicates(subset='Date', inplace=True)
    input_df = input_df[input_df['BTCClose'].notnull()]  # Remove non datapoints from the set
    input_df.sort_values(by='Date', inplace=True)
    input_df.reset_index(inplace=True, drop=True)
    input_df = input_df[['Date', 'BTCOpen', 'BTCHigh', 'BTCLow', 'BTCClose', 'BTCVolume']]  #Reorder

    return input_df


def fetch_historical_data(path_dict, market, start_date, end_date, bittrex_obj):
    # this function is useful as code is ran for the same period in backtesting several times consecutively,
    # and fetching from original CSV takes longer as it is a much larger file

    print('Fetching historical data...')

    # get the historic data
    path = path_dict['updated history']

    df = pd.DataFrame(columns=['Date', 'BTCOpen', 'BTCHigh', 'BTCLow', 'BTCClose', 'BTCVolume'])

    # Fetch candle data from bittrex
    if end_date > datetime.now() - timedelta(days=9):

        attempts = 0
        while True:
            print('Fetching candles from Bittrex...')
            candle_dict = bittrex_obj.get_candles(
                market, 'oneMin')

            if candle_dict['success']:
                candle_df = process_candle_dict(candle_dict)
                print("Success getting candle data.")
                break
            else:
                print("Failed to get candle data.")
                time.sleep(2*attempts)
                attempts +=1

                if attempts == 5:
                    raise(TypeError)
                print('Retrying...')

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
    #
    # if df.empty or df.Date.min() > start_date:  # Fetch from download file (this is last because its slow)
    #
    #     print('Fetching data from the download file.')
    #     # get the historic data. Columns are Timestamp	Open	High	Low	Close	Volume_(BTC)	Volume_(Currency)	Weighted_Price
    #
    #     def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
    #     orig_df = pd.read_csv(path_dict['downloaded history'], usecols=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(Currency)'], parse_dates=[
    #         'Timestamp'], date_parser=dateparse)
    #     orig_df.rename(columns={'Timestamp': 'Date', 'Open': 'BTCOpen', 'High': 'BTCHigh',
    #                             'Low': 'BTCLow', 'Close': 'BTCClose', 'Volume_(Currency)': 'BTCVolume'}, inplace=True)
    #
    #     assert not orig_df.empty
    #
    #     df = df.append(orig_df)
    #

    # Double check that we have a correct date date range. Note: will still be triggered if missing the exact data point
    assert(df.Date.min() <= start_date)

    # if df.Date.max() < end_date:
    #     print('There is a gap between the download data and the data available from Bittrex. Please update download data.')
    #     assert(df.Date.max() >= end_date)

    df = format_df(df)
    df = df[df['Date'] >= start_date]
    df = df[df['Date'] <= end_date]

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
    currency = "BTC"
    # Structured to be currency agnostic

    for col in df_cols:

        if col in [currency + 'Open', currency + 'High', currency + 'Low']:
            input_df.drop(columns=[col], inplace = True)


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

# def make_col_static(input_df, col):
#     input_df['Static'] = np.log(input_df['BTCUSD']) - np.log(input_df['BTCUSD']).shift(1)
#     input_df['Static'].iloc[0] = 0
#     input_df['Static'] = input_df['BTCUSD']
#     input_df = input_df[['Date', 'BTCUSD', 'Static']]

def add_sma_as_column(input_df, p):
    # p is a number
    price = input_df['BTCClose'].values  # returns an np price

    sma = np.empty_like(price)
    # for i, item in enumerate(np.nditer(price)):
    #     if i == 0:
    #         sma[i] = item
    #     elif i < p:
    #         sma[i] = price[0:i].mean()
    #     else:
    #         sma[i] = price[i - p:i].mean()
    #
    # for i, item in enumerate(np.nditer(price)):
    #     if i == 0:
    #         sma[i] = 0
    #     elif i < p:
    #         sma[i] = np.log(price[0:i].mean()) - np.log(price[0:i-1].mean())
    #     else:
    #         sma[i] = np.log(price[i - p:i].mean()) - np.log(price[i - p-1:i-1].mean())

    for i, item in enumerate(np.nditer(price)):
        if i == 0:
            sma[i] = 0
        elif i < p:
            sma[i] = price[0:i].mean() - price[0:i-1].mean()
        else:
            sma[i] = price[i - p:i].mean() - price[i - p-1:i-1].mean()


    col = 'SMA_' + str(p)
    # df_col = pd.Series(data = sma)
    input_df[col] = sma  # modifies the input df
    return input_df

def add_features(input_df):
    base = 100
    # input_df = add_sma_as_column(input_df, base)
    # input_df = add_sma_as_column(input_df, int(base*8/5))
    input_df = add_sma_as_column(input_df, int(base*13/5))
    # subtract
    indicator = np.empty_like(sma)
    for i, item in enumerate(np.nditer(price)):
        indicator[i] = sma[i] - price[i]

    df['SMA_' + str(p)] = indicator  # modifies the input df

    input_df['BTCMACD'] = ta.trend.macd_diff(input_df['BTCClose'], fillna  = True)
    input_df['BTCRSI'] = ta.momentum.rsi(input_df['BTCClose'], fillna  = True)
    input_df['BTCRSI'] = input_df['BTCRSI'] - 50 #center at 0
    # input_df['BTCOBV'] = ta.volume.on_balance_volume(input_df['BTCClose'], input_df['BTCVolume'], fillna  = True)
    # print(input_df.head())

    return input_df



def make_static(df):
    new_df = pd.DataFrame(columns = df.columns)

    for col in df.columns:
        if col == 'Date':
            continue

        new_df[col] = np.log(df['BTCClose']) - np.log(df['BTCClose']).shift(1)
        new_df[col].iloc[0] = 0

    return new_df


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
        if not col[3:] in ['Open', 'High', 'Low', 'Close'] and not col in ['Date', 'BTCVolume']:
            df.plot(x='Date', y=col, ax=ax3)

    my_roi = ROI(log.Value.iloc[0], log.Value.iloc[-1])
    sharpe = my_roi/log.Value.std()
    print(f'Sharpe Ratio: {sharpe}') #one or better is good
    log.plot(y='Value', ax=ax2)

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


def get_scaler(env):
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here
    """From video:
    Need some data --> play an episode randomly
    Running for multiple episodes will make more accurate
    """

    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, val, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


class LinearModel:
    """ A linear regression model """

    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / \
            np.sqrt(input_dim)  # Random matrix
        self.b = np.zeros(n_action)  # Vector of zeros

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []

    def predict(self, X):
        # make sure X is N x D
        # throw error if X not 2D to abide by (skikitlearn dimensionality convention)
        assert(len(X.shape) == 2)
        return X.dot(self.W) + self.b


    def sgd(self, X, Y, learning_rate=0.0075, momentum=0.9):
    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        """One step of gradient descent.
        u = momentum term
        n = learning rate
        g(t) = gradient
        theta = generic parameter

        v(t) = u*v(t-1) - n*g(t)

        let theta = T
        T(t) = T(t-1) + v(t), T = {W,b)}
        """

        # make sure X is N x D
        assert(len(X.shape) == 2)

        # the loss values are 2-D
        # normally we would divide by N only
        # but now we divide by N x K
        num_values = np.prod(Y.shape)

        # do one step of gradient descent
        # we multiply by 2 to get the exact gradient
        # (not adjusting the learning rate)
        # i.e. d/dx (x^2) --> 2x
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            Yhat = self.predict(X)
            gW = 2 * X.T.dot(Yhat - Y) / num_values
            gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y)**2)  # Using the mean squared error (This was from the class code, started throwing runtime errors)
        # mse = ((Yhat - Y)**2).mean(axis = None) #still throws run time errors
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)


class SimulatedMarketEnv:
    """
    A multi-asset trading environment.
    For now this has been adopted for only one asset.
    Below shows how to add more.
    The state size and the aciton size throughout the rest of this
    program are linked to this class.

    State: vector of size 7 (n_asset + n_asset*n_indicators)
      - static price of asset 1 (using BTCClose price)
      - associated indicators for each asset
    """

    def __init__(self, data, initial_investment=100):
        # data
        self.asset_data = data
        # print(self.asset_data[0:10]) #good
        self.n_indicators = 3

        # n_step is number of samples, n_stock is number of assets. Assumes to datetimes are included
        self.n_step, self.n_asset = self.asset_data.shape
        # for now this works but will need to be updated when multiple assets are added
        self.n_asset -= self.n_indicators + 1

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.assets_owned = None
        self.asset_prices = None
        self.USD = None

        # Create the attributes to store indicators. This has been implemented to incorporate more information about the past to mitigate the MDP assumption.

        self.min_trade_spacing = 1  # The number of datapoints that must occur between trades
        self.period_since_trade = self.min_trade_spacing

        portfolio_granularity = 1  # smallest fraction of portfolio for investment in single asset
        # The possible portions of the portfolio that could be allocated to a single asset
        possible_vals = [x / 100 for x in list(range(0, 101, int(portfolio_granularity * 100)))]
        # calculate all possible allocations of wealth across the available assets
        self.action_list = []
        permutations_list = list(map(list, itertools.product(possible_vals, repeat=self.n_asset)))
        #Only include values that are possible (can't have more than 100% of the portfolio)
        for i, item in enumerate(permutations_list):
            if sum(item) <= 1:
                self.action_list.append(item)

        #This list is for indexing each of the actions
        self.action_space = np.arange(len(self.action_list))

        # calculate size of state (amount of each asset held, value of each asset, cash in hand, volumes, indicators)
        self.state_dim = self.n_asset*3 + 1 + self.n_indicators*self.n_asset   #State is the static data and the indicators for each asset (currently ignoring volume)

        # self.rewards_hist_len = 10
        # self.rewards_hist = np.ones(self.rewards_hist_len)

        self.last_action = []
        self.reset()

    def reset(self):
        # Resets the environement to the initial state
        self.cur_step = 0  # point to the first datetime in the dataset
        # Own no assets to start with
        self.assets_owned = np.zeros(self.n_asset)
        self.last_action = self.action_list[0] #The action where nothing is owned
        """ the data, for asset_data can be thought of as nested arrays, where indexing the
        highest order array gives a snapshot of all data at a particular time, and information at the point
        in time can be captured by indexing that snapshot."""
        self.asset_prices = self.asset_data[self.cur_step][0:self.n_asset]

        self.USD = self.initial_investment

        # print(self.cur_state)
        # print(self.asset_prices)
        # print(self.n_asset)

        return self._get_state(), self._get_val() # Return the state vector (same as obervation for now)

    def step(self, action):
        # Performs an action in the enviroment, and returns the next state and reward



        if not action in self.action_space:
            #Included for debugging
            print(action)
            print(self.action_space)
            assert action in self.action_space  # Check that a valid action was passed

        # get current value before performing the action
        prev_val = self._get_val()
        prev_price = self.asset_prices[0]
        # update price, i.e. go to the next day
        self.cur_step += 1

        """ the data, for asset_data can be thought of as nested arrays, where indexing the
        highest order array gives a snapshot of all data at a particular time, and information at the point
        in time can be captured by indexing that snapshot."""
        self.asset_prices = self.asset_data[self.cur_step][0:self.n_asset]

        # perform the trade
        self._trade(action)

        # get the new value after taking the action
        cur_val = self._get_val()
        cur_price = self.asset_prices[0]
        # reward is the increase in porfolio value
        delta = (cur_price - prev_price)/prev_price #Percentage change in price
        reward = delta*(self.assets_owned[0]*self.asset_prices[0] - self.USD)/cur_val
        # self.rewards_hist = np.roll(self.rewards_hist,1)
        # self.rewards_hist[0] = step_return
        # reward = sortino_ratio(self.rewards_hist)

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # store the current value of the portfolio here
        info = {'cur_val': cur_val}

        # conform to the Gym API
        #      next state       reward  flag  info dict.
        return self._get_state(), self._get_val(), reward, done, info

    def _get_state(self):
        # Returns the state (for now state, and observation are the same.
        # Note that the state could be a transformation of the observation, or
        # multiple past observations stacked.)
        #state is  (amount of each asset held, value of each asset, cash in hand, volumes, indicators)
        state = np.empty(self.state_dim)  #assets_owned, USD, volume, indicators
        state[0:self.n_asset] = self.last_action    #This is the fraction of the portfolio invested
        state[self.n_asset] = self.USD     #This is set in trade
        # Asset data is amount btc price, usd, volume, indicators


        #Make data static
        full_slice = self.asset_data[self.cur_step]
        # print(full_slice)
        if self.cur_step in [0]:
            slice = full_slice[0:2*self.n_asset]
            last_slice = self.asset_data[self.cur_step + 1][0:2*self.n_asset]

            stationary_slice = np.log(slice) - np.log(last_slice)

            #Set the stationary values
            state_slice = []
            for i, item in enumerate(full_slice):
                if i < 2*self.n_asset:
                    state_slice.append(stationary_slice[i])
                else:
                    state_slice.append(full_slice[i])
        else:
            slice = full_slice[0:2*self.n_asset]
            last_slice = self.asset_data[self.cur_step - 1][0:2*self.n_asset]

            stationary_slice = np.log(slice) - np.log(last_slice)

            #Set the stationary values
            state_slice = []
            for i, item in enumerate(full_slice):
                if i < 2*self.n_asset:
                    state_slice.append(stationary_slice[i])
                else:
                    state_slice.append(full_slice[i])

        # print(full_slice)
        indexs = [self.n_asset + 1, self.n_asset +1 + len(full_slice)]  #good
        state[indexs[0]:indexs[1]] = state_slice   #Taken from data
        # print(state)
        if self.cur_step == 0:
            static_slice = np.zeros(len(self.asset_data[0]))
        else:
            slice = self.asset_data[self.cur_step]
            last_slice = self.asset_data[self.cur_step - 1]
            static_slice = np.log(slice) - np.log(last_slice)

        state[self.n_asset+1:(self.n_asset*2 + 2 + self.n_indicators*self.n_asset)] =  static_slice   #Taken from data
        # state =  self.asset_data[self.cur_step]   #Taken from data
        # last_index = self.cur_step
        # if last_index < 0:
        #     last_index = 0
        # last_price = self.asset_data[last_index][0]
        # last_vol = self.asset_data[last_index][1]
        #
        # state[0] = state[0]/last_price
        # state[1] = state[1]/last_vol
        print(state)
        return state

    def _get_val(self):
        return self.assets_owned.dot(self.asset_prices) + self.USD

    def _trade(self, action):
        # index the action we want to perform
        # action_vec = [(desired amount of stock 1), (desired amount of stock 2), ... (desired amount of stock n)]

        action_vec = self.action_list[action]

        if action_vec != self.last_action and self.period_since_trade >= self.min_trade_spacing:  # if attmepting to change state

            #Calculate the changes needed for each asset
            delta = [s_prime - s for s_prime, s in zip(action_vec, self.last_action)]

            #THIS WILL NEED TO BE MORE COMPLEX IF MORE ASSETS ARE ADDED
            #First fulfill the required USD
            val = self._get_val()

            # for i, d in enumerate(delta):
            # self.assets_owned[0]  += delta[0]*val*self.asset_prices[0]
            # self.USD -= delta[0]*val


            # Sell everything
            for i, a in enumerate(action_vec):
                self.USD += self.assets_owned[i] * self.asset_prices[i]
                self.assets_owned[i] = 0

            # Buy back the right amounts
            for i, a in enumerate(action_vec):
                cash_to_use = a * self.USD
                self.assets_owned[i] = cash_to_use / self.asset_prices[i]
                self.USD -= cash_to_use

            self.last_action = action_vec
            self.period_since_trade = 0
        else:
            self.period_since_trade += 1


class BittrexMarketEnv:
    """
    A multi-asset trading environment.
    For now this has been adopted for only one asset.
    Below shows how to add more.
    The state size and the aciton size throughout the rest of this
    program are linked to this class.

    State: vector of size 7 (n_asset + n_asset*n_indicators)
      - static price of asset 1 (using BTCClose price)
      - associated indicators for each asset
    """

    def __init__(self, initial_investment=10):

        #get my keys
        with open("/Users/biver/Documents/crypto_data/secrets.json") as secrets_file:
            keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
            secrets_file.close()

        self.bittrex_obj_1_1 = Bittrex(keys["key"], keys["secret"], api_version=API_V1_1)
        self.bittrex_obj_2 = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)
        # data
        self.n_indicators = 3

        # n_step is number of samples, n_stock is number of assets. Assumes to datetimes are included
        self.n_asset = 1
        # instance attributes
        self.initial_investment = initial_investment
        self.assets_owned = None #this needs to change
        self.asset_prices = None
        self.asset_volumes = None
        self.USD = None

        self.path_dict = path_dict

        self.markets = ['USD-BTC', 'USD-ETH', 'USD-XMR']    #Alphabetical
        self.data_length = 10  #The maximum length to allow the df to get to (data is kept for ta purposes)
        now = datetime.now()
        self.df = self._fetch_candle_data(self.markets[0], now - timedelta(minutes = 100), now)
        print(self.df.tail())
        self._calculate_indicators()

        self.markets = ['USD-BTC']


        portfolio_granularity = 1  # smallest fraction of portfolio for investment in single asset
        # The possible portions of the portfolio that could be allocated to a single asset
        possible_vals = [x / 100 for x in list(range(0, 101, int(portfolio_granularity * 100)))]
        # calculate all possible allocations of wealth across the available assets
        self.action_list = []
        permutations_list = list(map(list, itertools.product(possible_vals, repeat=self.n_asset)))
        #Only include values that are possible (can't have more than 100% of the portfolio)
        for i, item in enumerate(permutations_list):
            if sum(item) <= 1:
                self.action_list.append(item)

        #This list is for indexing each of the actions
        self.action_space = np.arange(len(self.action_list))

        # calculate size of state (amount of each asset held, cash in hand, value of each asset, volumes, indicators)
        self.state_dim = self.n_asset*3 + 1 + self.n_indicators*self.n_asset   #State is the static data and the indicators for each asset (currently ignoring volume)

        self.last_action = []
        self.reset()


    def reset(self):
        # Resets the environement to the initial state

        self._cancel_all_orders()

        self._get_obs()

        #Put all money into USD
        if self.assets_owned[0] > 0:
            self._trade(-self.assets_owned[0])


        self.assets_owned = np.zeros(self.n_asset)
        self.last_action = self.action_list[0] #The action where nothing is owned
        """ the data, for asset_data can be thought of as nested arrays, where indexing the
        highest order array gives a snapshot of all data at a particular time, and information at the point
        in time can be captured by indexing that snapshot."""

        return self._get_state(), self._return_val() # Return the state vector (same as obervation for now)


    def _calculate_indicators(self):

        def sma(p):
            # p is a number
            price = self.df['BTCClose'].values  # returns an np price

            sma = np.empty_like(price)
            for i, item in enumerate(np.nditer(price)):
                if i == 0:
                    sma[i] = item
                elif i < p:
                    sma[i] = price[0:i].mean()
                else:
                    sma[i] = price[i - p:i].mean()

            self.df['SMA_' + str(p)] = sma  # modifies the input df
        base = 100
        sma(base)
        sma(int(base*8/5))
        sma(int(base*13/5))
        # add the momentum indicators

    def _get_obs(self):

        print('Fetching account balances.')
        self.assets_owned = np.zeros(self.n_asset)
        while True:
            check1 = False
            balance_response = self.bittrex_obj_1_1.get_balance('BTC')
            if balance_response['success']:
                self.assets_owned[0] = balance_response['result']['Balance']

                #Find a more legant way of checking if 'None'
                if not np.isnan(self.assets_owned[0]):
                    pass
                else:
                    self.assets_owned[0] = 0

                    check1 = True

            balance_response = self.bittrex_obj_1_1.get_balance('USD')

            if balance_response['success']:
                self.USD = balance_response['result']['Balance']

                try:
                    if self.USD > 0:
                        pass

                except TypeError: #BTC_balance is none
                        self.USD = 0
                if check1:
                    break


        print('Fetching prices.')
        while True:
            ticker = self.bittrex_obj_1_1.get_ticker('USD-BTC')
            #Change this to make sure that the check went through
            # Check that an order was entered
            if not ticker['success']:
                print('_get_prices failed. Trying again. Ticker message: ')
                print(ticker['message'])
                time.sleep(1)
            else:
                self.asset_prices = [ticker['result']['Last']]

                try:
                    if self.asset_prices[0] > 0:
                        pass
                except TypeError: #BTC_balance is none
                        self.asset_prices[0] = 0
                break

        print('Fetching latest candle')
        candle = self.bittrex_obj_2.get_latest_candle(self.markets[0], 'oneMin')
        candle = process_candle_dict(candle)

        self.df = self.df.append(candle)
        self._calculate_indicators()
        # self.df.drop_duplicates(subset='Date', inplace=True)
        self.df.sort_values(by='Date', inplace=True)   #This was causing a warning about future deprecation/changes to pandas
        self.df = self.df[['Date', 'BTCOpen', 'BTCHigh', 'BTCLow', 'BTCClose', 'BTCVolume']]  #Reorder

        self.df.reset_index(inplace=True, drop=True)

        #Drop some rows
        data_len = len(self.df.Date)
        # self.df = self.df[:-self.data_length]

        self.df.reset_index(inplace=True, drop=True)

        print(self.df.tail())


    def _get_state(self):
        # Returns the state (for now state, and observation are the same.
        # Note that the state could be a transformation of the observation, or
        # multiple past observations stacked.)
        state = np.empty(self.state_dim)  #assets_owned, USD
        # self.cur_state[0:self.n_asset] = self.asset_prices
        state[0:self.n_asset] = self.last_action
        state[self.n_asset] = self.USD     #This is set in trade
        # Asset data is amount btc price, usd, volume, indicators

        #Make data static
        slice = self.df.iloc[[-1]] #This should be close, volumes, indicators
        slice = slice.drop('Date', axis = 1).values[0]


        last_slice = self.df.iloc[[-2]]
        last_slice = last_slice.drop('Date', axis = 1).values[0]
    def _get_prices(self):
        print('Fetching prices.')
        while True:
            ticker = self.bittrex_obj_1_1.get_ticker('USD-BTC')
            #Change this to make sure that the check went through
            # Check that an order was entered
            if not ticker['success']:
                print('_get_prices failed. Trying again. Ticker message: ')
                print(ticker['message'])
                time.sleep(1)
            else:
                break

        self.asset_prices = [ticker['result']['Last']]


    def _get_balances(self):
        print('Fetching account balances.')
        self.assets_owned = np.zeros(self.n_asset)
        while True:
            check1 = False
            balance_response = self.bittrex_obj_1_1.get_balance('BTC')
            if balance_response['success']:
                self.assets_owned[0] = balance_response['result']['Balance']

        stationary_slice = np.log(slice) - np.log(last_slice)

        state[self.n_asset+1:(self.n_asset*2 + 2 + self.n_indicators*self.n_asset)] =  stationary_slice   #Taken from

        return state


    def _return_val(self):
        self._get_obs()
        print(self.asset_prices, self.assets_owned, self.USD)
        return self.assets_owned.dot(self.asset_prices) + self.USD



    def _fetch_candle_data(self, market, start_date, end_date):
        # this function is useful as code is ran for the same period in backtesting several times consecutively,
        # and fetching from original CSV takes longer as it is a much larger file

        print('Fetching historical data...')

        # get the historic data
        path = self.path_dict['updated history']

        df = pd.DataFrame(columns=['Date', 'BTCOpen', 'BTCHigh', 'BTCLow', 'BTCClose', 'BTCVolume'])

        # Fetch candle data from bittrex
        if end_date > datetime.now() - timedelta(days=10):

            attempts = 0
            while True:
                print('Fetching candles from Bittrex...')
                candle_dict = self.bittrex_obj_2.get_candles(
                    market, 'oneMin')

                if candle_dict['success']:
                    candle_df = process_candle_dict(candle_dict)
                    print("Success getting candle data.")
                    break
                else:
                    print("Failed to get candle data.")
                    time.sleep(2*attempts)
                    attempts +=1

                    if attempts == 5:
                        raise(TypeError)
                    print('Retrying...')

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
            orig_df = pd.read_csv(self.path_dict['downloaded history'], usecols=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(Currency)'], parse_dates=[
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


    def _cancel_all_orders(self):
    def cancel_all_orders(self):
        print('Canceling all orders.')
        open_orders = self.bittrex_obj_1_1.get_open_orders('USD-BTC')
        if open_orders['success']:
            if not open_orders['result']:
                print('No open orders.')
            else:
                for order in open_orders['result']:
                    uuid = order['OrderUuid']
                    cancel_result = self.bittrex_obj_1_1.cancel(uuid)['success']
                    if cancel_result == True:  # need to see if im checking if cancel_result exits or if im checking its value
                        print('Cancel status: ', cancel_result, ' for order: ', uuid)

        else:
            print('Failed to get order history.')
            print(result)


    def _is_trade_executed(self, uuid):
        start_time = datetime.now()
        # Loop to see if the order has been filled
        is_open = True
        cancel_result = False
        time_limit = 30 #THIS SHOULD BE CHANGED EVENTAULLY
        while is_open:
            order_data = self.bittrex_obj_1_1.get_order(uuid)
            try:
                is_open = order_data['result']['IsOpen']
            except TypeError:
                print(is_open)
            # print('Order open status: ', is_open)

            #Case: order filled
            if not is_open:
                return True
                break

            time.sleep(1)
            time_elapsed = datetime.now() - start_time

            #Case: time limit reached
            if time_elapsed > timedelta(seconds=time_limit):
                print(f'Order has not gone through in {time_limit} seconds. Cancelling...')
                # Cancel the order
                cancel_result = self.bittrex_obj_1_1.cancel(uuid)['success']
                if cancel_result == True:  # need to see if im checking if cancel_result exits or if im checking its value
                    print(f'Cancel status: {cancel_result} for order: {uuid}.')
                    return False
                    break #Break out of order status loop
                print(cancel_result)


    def _trade(self, amount):
        #amount in USD

        #For now, assuming bitcoin
        #First fulfill the required USD
        self._get_obs()

        # Note that bittrex exchange is based in GMT 8 hours ahead of CA

        start_time = datetime.now()

        # amount = amount / self.asset_prices  # Convert to the amount of BTC

        # Enter a trade into the market.
        # Example result  {'success': True, 'message': '', 'result': {'uuid': '2641035d-4fe5-4099-9e7a-cd52067cde8a'}}
        if amount > 0:  # buy
          trade_result = self.bittrex_obj_1_1.buy_limit(self.markets[0], amount, self.asset_prices[0])
          side = 'buying'
        else:       # Sell
          trade_result = self.bittrex_obj_1_1.sell_limit(self.markets[0], -amount, self.asset_prices[0])
          side = 'selling'

        # Check that an order was entered
        if not trade_result['success']:
          print('Trade attempt failed')
          print(trade_result['message'])
          return False
        else:
            print(f'Order for {side} {amount:.8f} {self.markets[4:6]} at a price of {self.asset_prices[0]:.2f} has been submitted to the market.')
            order_uuid = trade_result['result']['uuid']

            # Loop for a time to see if the order has been filled
            status = self._is_trade_executed(order_uuid)

            if status == True:
                print(f'Order has been filled. Id: {order_uuid}.')
                return True
            else:
                print('Order not filled')
                return False

            dt = datetime.now() - start_time  # note that this include the time to run a small amount of code

            order_data['result']['Order Duration'] = dt
            trade = process_order_data(order_data)
            # print(trade)


    def act(self, action):
        # index the action we want to perform
        # action_vec = [(desired amount of stock 1), (desired amount of stock 2), ... (desired amount of stock n)]

        action_vec = self.action_list[action] #a vectyor like [0.1, 0.5] own 0.1*val of BTC, 0.5*val of ETH etc.

        if action_vec != self.last_action:  # if attmepting to change state

            #THIS WILL NEED TO BE MORE COMPLEX IF MORE ASSETS ARE ADDED
            #Calculate the changes needed for each asset
            delta = [s_prime - s for s_prime, s in zip(action_vec, self.last_action)]


class DQNAgent(object):
    """ Responsible for taking actions, learning from them, and taking actions
    such that they will maximize future rewards
    """

    def __init__(self, state_size, action_size):

        # These two correspond to number of inputs and outputs of the neural network respectively
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # originally .01. The version here is set for training
        self.epsilon_decay = 0.99  # originall .995
        # Get an instance of our model
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        # This is the policy
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)  # Greedy case

        # Take argmax over model predictions to get action with max. Q value.
        # Output of model is batch sized by num of outputs to index by 0
        return np.argmax(act_values[0])  # returns action

    def train(self, state, action, reward, next_state, done):
        # This func. does the learning
        if done:
            target = reward
        else:
            target = reward + self.gamma * \
                np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target

        # Run one training step of gradient descent
        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def play_one_episode(agent, env, scaler, is_train, record=False):
    # note: after transforming states are already 1xD
    log_columns = ['Value']
    log = pd.DataFrame(columns=log_columns)

    state, val  = env.reset()

    if record == True:
        log = log.append(pd.DataFrame.from_records(
            [dict(zip(log_columns, [val]))]), ignore_index=True)

    state = scaler.transform([state])
    done = False

    while not done:

        action = agent.act(state)
        next_state, val, reward, done, info = env.step(action)

        if record == True:
            log = log.append(pd.DataFrame.from_records(
                [dict(zip(log_columns, [val]))]), ignore_index=True)

        next_state = scaler.transform([next_state])
        if is_train in ['train', 'add_train']:
            agent.train(state, action, reward, next_state, done)
        state = next_state

    if record:
        return info['cur_val'], log
    else:
        return info['cur_val']



def run_agent(mode, path_dict, start_date, end_date, episodes, symbols='USDBTC'):
    # Mode should be a string, either train or test or run
    # maybe it would be helpful to run this through command line argv etc

    models_folder = path_dict['models']
    rewards_folder = path_dict['rewards']

    # maybe_make_dir(models_folder)
    # maybe_make_dir(rewards_folder)

    # variable for storing final value of the portfolio (done at end of episode)
    portfolio_value = []

    def return_on_investment(final, initial):
        # Returns the percentage increase/decrease
        return round(final / initial - 1, 4) * 100

    num_episodes = 100
    batch_size = 32  # sampleing from replay memory
    initial_investment = 100
    if mode in ['train', 'test']:
        print('Preparing data...')
        #get my keys
        with open(path_dict['secret']) as secrets_file:
            keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
            secrets_file.close()

        my_bittrex2_0 = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

        market = symbols[0:3] + '-' + symbols[3:6]

        df = fetch_historical_data(path_dict, market, start_date, end_date, my_bittrex2_0)  # oldest date info

        # save_historical_data(path_dict, df)

        df = add_features(df)
        df = strip_open_high_low(df)

        print(df.head())
        save_historical_data(path_dict, df)

        data_to_fit = df.drop('Date', axis = 1).values

        sim_env = SimulatedMarketEnv(data_to_fit, initial_investment)
        state_size = sim_env.state_dim
        action_size = len(sim_env.action_space)
        agent = DQNAgent(state_size, action_size)
        my_scaler = get_scaler(sim_env)

        num_episodes = episodes
        if mode == 'test':
            print('Testing...')
            num_episodes = 10
            # then load the previous scaler
            with open(f'{models_folder}/scaler.pkl', 'rb') as f:
                my_scaler = pickle.load(f)

            # make sure epsilon is not 1!
            # no need to run multiple episodes if epsilon = 0, it's deterministic

            agent.epsilon_min = 0.#00001
            agent.epsilon_min = 0.0005
            agent.epsilon = agent.epsilon_min

            # load trained weights
            agent.load(f'{models_folder}/linear.npz')
        elif mode == 'train':
            test_vals = []

        time_remaining = timedelta(hours=0)
        market_roi = return_on_investment(df.BTCClose.iloc[-1], df.BTCClose.iloc[0])
        print(f'The market changed by {market_roi} % over the designated period.')

        # play the game num_episodes times
        for e in range(num_episodes):

            t0 = datetime.now()

            if e == num_episodes-1 and mode == 'test':
                val, state_log = play_one_episode(agent, sim_env, my_scaler, mode, True)
                print(state_log.head())
            elif mode == 'train' and e % 25 == 0:
                # agent.epsilon = 0.
                val = play_one_episode(agent, sim_env, my_scaler, 'test')
                # agent.epsilon = 1
                # test_vals.append(val)
            else:
                val = play_one_episode(agent, sim_env, my_scaler, mode)

            roi = return_on_investment(val, initial_investment)  # Transform to ROI

            dt = datetime.now() - t0
            # if not end_time in locals():    #initialize with a direct calculation
            #     end_time = dt* (num_episodes - e)
            # else:
            time_remaining -= dt
            time_remaining = time_remaining + \
                (dt * (num_episodes - (e + 1)) - time_remaining) / (e + 1)
            if e % 100 == 0:
                # save the weights
                if mode in ['train', 'add_train']:
                    # save the DQN
                    agent.save(f'{models_folder}/linear.npz')
                    print('DQN saved.')

                    print('Saving scaler...')
                    # save the scaler
                    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
                        pickle.dump(my_scaler, f)
                    print('Scaler saved.')
            # plot losses
            if num_episodes <= 500 or e % 10 == 0:
                print(f"episode: {e + 1}/{num_episodes}, end value: {val:.2f}, episode roi: {roi:.2f}, time remaining: {time_remaining}")
            portfolio_value.append(val)  # append episode end portfolio value

        # save the weights when we are done
        if mode in ['train']:
            # save the DQN
            agent.save(f'{models_folder}/linear.npz')
            print('DQN saved.')


            # save the scaler
            with open(f'{models_folder}/scaler.pkl', 'wb') as f:
                pickle.dump(my_scaler, f)
            print('Scaler saved.')

            # save portfolio value for each episode
            print('Saving rewards...')
            np.save(f'{rewards_folder}/{mode}.npy', portfolio_value)
            print('Rewards saved.')


            # plot losses
            plt.plot(test_vals)
            plt.show()

        # save portfolio value for each episode
        print('Saving rewards...')
        np.save(f'{rewards_folder}/{mode}.npy', portfolio_value)
        print('Rewards saved.')

        if mode == 'test':
            plot_sim_trade_history(df, state_log)
        else:
            plot_data(df)
    else:
        assert(mode == 'run')

        #Prepare the agent
        env = BittrexMarketEnv(path_dict)
        state_size = env.state_dim
        action_size = len(env.action_space)
        agent = DQNAgent(state_size, action_size)
        # load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # make sure epsilon is not 1!
        # Set to 0 for purely deterministic
        agent.epsilon = 0.001

        # Price in USD, price per unit is $/BTC

        log = pd.DataFrame()

        log_columns = ['Value']
        log = pd.DataFrame(columns=log_columns)

        state, val  = env.reset()

        log = log.append(pd.DataFrame.from_records(
                [dict(zip(log_columns, [val]))]), ignore_index=True)
        print(state)
        state = scaler.transform([state])
        done = False

        while True:
            action = agent.act(state)
            next_state, val, reward, done, info = env.act(action)

            if record == True:
                log = log.append(pd.DataFrame.from_records(
                    [dict(zip(log_columns, [val]))]), ignore_index=True)

            next_state = scaler.transform([next_state])
            if mode in ['train', 'add_train']:
                agent.train(state, action, reward, next_state, done)
            state = next_state


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()


    #cryptodatadownload has gaps
    #Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********
    os = 'windows' #linux or windows
    des_granularity = 1 #in minutes
    symbols = 'BTCUSD' #Example: 'BTCUSD'

    #The below should be updated to be simplified to use parent directory? unsure how that works...
    #https://stackoverflow.com/questions/48745333/using-pandas-how-do-i-save-an-exported-csv-file-to-a-folder-relative-to-the-scr?noredirect=1&lq=1

    if os == 'linux':
        paths = {'downloaded history': '/home/bruce/AlgoTrader/BittrexTrader/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
        'updated history': '/home/bruce/AlgoTrader/updated_history_' + symbols + '.csv',
        'secret': "/home/bruce/Documents/crypto_data/secrets.json",
        'rewards': 'agent_rewards',
        'models': 'agent_models',
        'test trade log':  'C:/Python Programs/crypto_trader/historical data/trade_testing' + symbols + '.csv'}

        # TODO: add a loop here that appends the asset folders

    elif os == 'windows':
        paths = {'downloaded history': 'C:/Python Programs/crypto_trader/historical data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv',
        'updated history': 'C:/Python Programs/crypto_trader/historical data/updated_history_' + symbols + '.csv',
        'secret': "/Users/biver/Documents/crypto_data/secrets.json",
        'rewards': 'agent_rewards',
        'models': 'agent_models',
        'test trade log':  'C:/Python Programs/crypto_trader/historical data/trade_testing' + symbols + '.csv'}
    else:
        print('Unknown OS passed when defining the paths')  # this should throw and error


    if mode in ['train']:
        #train
        start = datetime(2018, 11, 1)
        end = datetime(2019, 1, 1)
        # start = datetime(2019,12, 23)
        # end = datetime(2020, 1, 1)

    elif mode == 'run':
        start = datetime(2019,12, 14)
        end = datetime(2019, 12, 29)

    else:
        assert(args.mode == 'test')  #make sure that a proper mode was passed
        #test

        start = datetime(2019,12, 13)
        end = datetime.now()
        # start = datetime(2018, 9, 20)
        # end = datetime(2019, 3, 1)

    run_agent(mode, paths, start, end, 1500)

    # for epochs in [200, 300, 400, 500, 600, 700, 800]:
    #
    #     start = datetime(2019,12, 14)
    #     end = datetime(2019, 12, 29)
    #
    #     run_agent('train', paths, start, end)
    #     start = datetime(2019,1, 1)
    #     end = datetime(2019, 1, 15)
    #     run_agent('test', paths, start, end)
        # start = datetime(2019,12, 14)
        # end = datetime(2019, 12, 28)
