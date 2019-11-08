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
import os


# I had a ton of trouble getting the plots to look right with the dates.
# This link was really helpful http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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
    path = path_dict['downloaded history']

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
    path = path_dict['updated history']

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
    print(df.head())
    return df.values


def overwrite_csv_file(path_dict, array):
    # This function writes the information in the original format to the csv file
    # including new datapoints that have been fetched

    print('Writing data to CSV.')

    path = path_dict['Updated']
    # must create new df as df is passed by reference
    # datetimes to strings
    df = pd.DataFrame({'Date': array[:, 0], 'Close': np.float_(array[:, 1])})
    print(df.head())
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


# need a plot training results
# plot testing results
# plot running results
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
        plt.axvline(x=strt)
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

def get_scaler(env):
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here

    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


class LinearModel:
    """ A linear regression model """

    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []

    def predict(self, X):
        # make sure X is N x D
        assert(len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
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
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y)**2)
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)


class MarketEnv:
    """
    A multi-asset trading environment.
    For now this has been adopted for only one asset.
    Below shows how to add more.

    State: vector of size 7 (n_stock * 2 + 1)
      - # shares of stock 1 owned
      - # shares of stock 2 owned
      - # shares of stock 3 owned
      - price of stock 1 (using daily close price)
      - price of stock 2
      - price of stock 3
      - cash owned (can be used to purchase more stocks)

    Action: categorical variable with 27 (3^3) possibilities
      - for each stock, you can:
      - 0 = sell
      - 1 = hold
      - 2 = buy
    """

    def __init__(self, data, initial_investment=20000):
        # data
        self.asset_price_history = data
        self.n_step, self.n_stock = self.asset_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.asset_owned = None
        self.asset_price = None
        self.cash_in_hand = None

        self.action_space = np.arange(3**self.n_stock)

        # action permutations
        # returns a nested list with elements like:
        # [0,0,0]
        # [0,0,1]
        # [0,0,2]
        # [0,1,0]
        # [0,1,1]
        # etc.
        # 0 = sell
        # 1 = hold
        # 2 = buy
        self.action_list = list(
            map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        # calculate size of state
        self.state_dim = self.n_stock * 2 + 1

        self.reset()

    def reset(self):
        #Resets the environement to the initial state
        self.cur_step = 0
        self.asset_owned = np.zeros(self.n_stock)
        self.asset_price = self.asset_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def step(self, action):
        assert action in self.action_space

        # get current value before performing the action
        prev_val = self._get_val()

        # update price, i.e. go to the next day
        self.cur_step += 1
        self.asset_price = self.asset_price_history[self.cur_step]

        # perform the trade
        self._test_trade(action)

        # get the new value after taking the action
        cur_val = self._get_val()

        # reward is the increase in porfolio value
        reward = cur_val - prev_val

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # store the current value of the portfolio here
        info = {'cur_val': cur_val}

        # conform to the Gym API
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.asset_owned
        obs[self.n_stock:2 * self.n_stock] = self.asset_price
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        return self.asset_owned.dot(self.asset_price) + self.cash_in_hand

    def _test_trade(self, action):
        # index the action we want to perform
        # 0 = sell
        # 1 = hold
        # 2 = buy
        # e.g. [2,1,0] means:
        # buy first stock
        # hold second stock
        # sell third stock
        action_vec = self.action_list[action]

        # determine which stocks to buy or sell
        sell_index = []  # stores index of stocks we want to sell
        buy_index = []  # stores index of stocks we want to buy
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # sell any stocks we want to sell
        # then buy any stocks we want to buy
        if sell_index:
            # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
            for i in sell_index:
                self.cash_in_hand += self.asset_price[i] * self.asset_owned[i]
                self.asset_owned[i] = 0
        if buy_index:
            # NOTE: when buying, we will loop through each stock we want to buy,
            #       and buy one share at a time until we run out of cash
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.asset_price[i]:
                        self.asset_owned[i] += 1  # buy one share
                        self.cash_in_hand -= self.asset_price[i]
                    else:
                        can_buy = False


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * \
                np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target

        # Run one training step
        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def play_one_episode(agent, env, is_train):
    # note: after transforming states are already 1xD
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return info['cur_val']


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

    # Note: can access by index
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


def run_agent(mode, df, bittrex_obj, extrema_filter, path_dict):
    # maybe it would be helpful to run this through command line argv etc

    account_info = bittrex_obj.get_balance('BTC')
    money = account_info['result']['Balance']
    fees = 0.0025  # standard fees are 0.3% per transaction

    while True:
        #Fetch new data
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


def ROI(final, initial):
    # Returns the percentage increase/decrease
    return round(final / initial - 1, 2) * 100
