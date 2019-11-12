from bittrex.bittrex import *

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import math

from datetime import datetime, timedelta
import json
import re
import os
import pickle
import itertools

from sklearn.preprocessing import StandardScaler


# I had a ton of trouble getting the plots to look right with the dates.
# This link was really helpful http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'I made a directory at {directory}')


def process_bittrex_dict(candle_dictionary):
    # Dataframe formatted the same as
    # V and BV refer to volume and base volume
    df = pd.DataFrame(candle_dictionary['result'])
    df.drop(columns=["BV", "V", 'O', 'H', 'L'])
    df = df.rename(columns={'T': "Date", 'C': 'Close'})

    # reorder the columns (defaults to alphabetic)
    df = df[['Date', 'Close']]
    df.reset_index(inplace=True, drop=True)
    # dates into datetimess
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%dT%H:%M:%S")
    return df


def get_candles(bittrex_obj, df, market, granularity):
    print('Fetching candles from Bittrex')
    candle_dict = bittrex_obj.get_candles(
        market, 'oneMin')
    if candle_dict['success']:
        new_df = process_bittrex_dict(candle_dict)
        print("Success getting candle data")

    else:
        print("Failed to get candle data")

    df = df.append(new_df)
    df = df.drop_duplicates(['Date'])
    df = df.sort_values(by='Date')
    df.reset_index(inplace=True, drop=True)
    return df


def original_csv_to_df(path_dict, earliest, end):

    print('Fetching historical data from download CSV.')

    # get the historic data
    path = path_dict['downloaded history']

    def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
    df = pd.read_csv(path, usecols=['Timestamp', 'Close'], parse_dates=[
                     'Timestamp'], date_parser=dateparse)
    df.rename(columns={'Timestamp': 'Date'}, inplace=True)

    # cryptodatadownload def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d %I-%p')
    # df = pd.read_csv(path, header = 1, usecols = ['Date', 'Close'], parse_dates=['Date'], date_parser=dateparse)

    # Format according to my specifications

    df = df[['Date', 'Close']]
    df = df[df['Date'] >= earliest]
    df = df[df['Date'] <= end]

    df = df[df['Close'].notnull()]  # Remove non datapoints from the set
    df.sort_values(by='Date', inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Remove datapoints according to desired des_granularity assume 1 min
    df = df[df['Date'].dt.second == 0]

    return df


def populate_updated_csv(path_dict):

    print('Fetching historical data from updated CSV.')
    # get the historic data
    read_path = path_dict['downloaded history']
    save_path = path_dict['updated history']
    paths = path_dict

    start_date = datetime(2015, 1, 1)
    end_date = datetime.now()
    df = original_csv_to_df(path_dict, start_date, end_date)
    save_historical_data(paths, df)


def fetch_historical_data(path_dict, market, start_date, end_date, bittrex_obj):
    # this function is useful as code is ran for the same period in backtesting several times consecutively,
    # and fetching from original CSV takes longer as it is a much larger file

    print('Fetching historical data.')

    # get the historic data
    path = path_dict['updated history']

    def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
    df = pd.read_csv(path, parse_dates=['Date'], date_parser=dateparse)

    if df.empty:
        populate_updated_csv(path_dict)

        def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
        df = pd.read_csv(path, parse_dates=['Date'], date_parser=dateparse)

    df = df[['Date', 'Close']]
    start_date_in_df = df.Date.min()


    if start_date_in_df > start_date:  # need to fetch from original
        orig_df = original_csv_to_df(
            path_dict, start_date, end_date)
        print(orig_df.head())
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


def save_historical_data(path_dict, df):
    # This function writes the information in the original format to the csv file
    # including new datapoints that have been fetched

    print('Writing data to CSV.')

    path = path_dict['updated history']
    # must create new df as df is passed by reference
    # # datetimes to strings
    # df = pd.DataFrame({'Date': data[:, 0], 'Close': np.float_(data[:, 1])})
    df['Date'] = df['Date'].dt.strftime("%Y-%m-%d %I-%p")
    df.to_csv(path)
    # added this so it doesnt change if passed by object... might be wrong idk
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d %I-%p")


# need a plot training results
# plot testing results
# plot running results
def plot_history(data, return_on_investment=0, market_performance=0):
    # df = pd.DataFrame({'Date': data[:, 0], 'Close': np.float_(data[:, 1])})

    fig, ax = plt.subplots()  # Create the figure

    df.plot(x='Date', y='Close', ax=ax)

    # df.plot(x='Date', y='Account Value', ax=ax)
    bot, top = plt.ylim()
    plt.ylim(0, top)
    fig.autofmt_xdate()
    plt.show()


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
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y)**2)  # Using the mean squared error
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

    State: vector of size 7 (n_asset * 2 + 1)
      - # shares of asset 1 owned
      - # shares of asset 2 owned
      - # shares of asset 3 owned
      - price of asset 1 (using close price)
      - price of asset 2
      - price of asset 3
      - cash owned (can be used to purchase more assets, USD)

    Action: categorical variable with 27 (3^3) possibilities
      - for each stock, you can:
      - 0 = sell
      - 1 = hold
      - 2 = buy
    """

    def __init__(self, data, initial_investment=20000):
        # data
        self.asset_price_history = data

        # n_step is number of samples, n_stock is number of assets. Assumes to datetimes are included
        self.n_step, self.n_asset = self.asset_price_history.shape
        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.asset_owned = None
        self.asset_price = None
        self.cash_in_hand = None

        # initializes as vector [0, 1, 2, ... 3^n_stock - 1]
        self.action_space = np.arange(3**self.n_asset)

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
        """The below line initializes a list of 3*n_stock with nested lists of length
        3 into each of the positions in the action_space, for each possible
        permutation of an action"""
        self.action_list = list(
            map(list, itertools.product([0, 1, 2], repeat=self.n_asset)))

        # calculate size of state
        self.state_dim = self.n_asset * 2 + 1

        self.reset()

    def reset(self):
        # Resets the environement to the initial state
        self.cur_step = 0  # point to the first datetime in the dataset
        # Own no assets to start with
        self.asset_owned = np.zeros(self.n_asset)
        self.asset_price = self.asset_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()  # Return the state vector (same as obervation for now)

    def step(self, action):
        # Performs an action in the enviroment, and returns the next state and reward

        assert action in self.action_space

        # get current value before performing the action
        prev_val = self._get_val()

        # update price, i.e. go to the next day
        self.cur_step += 1
        self.asset_price = self.asset_price_history[self.cur_step]

        # perform the trade
        self._trade(action)

        # get the new value after taking the action
        cur_val = self._get_val()

        # reward is the increase in porfolio value
        reward = cur_val - prev_val

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # store the current value of the portfolio here
        info = {'cur_val': cur_val}

        # conform to the Gym API
        #      next state       reward  flag  info dict.
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # Returns the state (for now state, and observation are the same.
        # Note that the state could be a transformation of the observation, or
        # multiple past observations stacked.)

        obs = np.empty(self.state_dim)
        # How many of each asset are owned
        obs[:self.n_asset] = self.asset_owned
        # Value of each stock
        obs[self.n_asset:2 * self.n_asset] = self.asset_price
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        return self.asset_owned.dot(self.asset_price) + self.cash_in_hand

    def _trade(self, action):
        # index the action we want to perform
        # 0 = sell
        # 1 = hold
        # 2 = buy
        # e.g. [2,1,0] means:
        # buy first stock
        # hold second stock
        # sell third stock
        # So in this case, action is a number that corresponds to one of the possible permutations of actions
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
    """ Responsible for taking actions, learning from them, and taking actions
    such that they will maximize future rewards
    """

    def __init__(self, state_size, action_size):

        # These two correspond to number of inputs and outputs of the neural network respectively
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
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


def play_one_episode(agent, env, scaler, is_train):
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


def run_agent(mode, data, bittrex_obj, path_dict, symbols='USDBTC'):
    # Mode should be a string, either train or test or run
    # maybe it would be helpful to run this through command line argv etc
    models_folder = path_dict['models']
    rewards_folder = path_dict['rewards']

    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    # variable for storing final value of the portfolio (done at end of episode)
    portfolio_value = []


    def return_on_investment(final, initial):
        # Returns the percentage increase/decrease
        return round(final / initial - 1, 2) * 100


    if mode == 'train' or 'test':

        # n_train = n_timesteps // 2 #floor division splitting the data into training and testing

        num_episodes = 500
        batch_size = 32  # sampleing from replay memory
        initial_investment = 1000 * 10000
        fees = 0.0025  # standard fees are 0.3% per transaction

        n_timesteps, n_stocks = data.shape
        sim_env = SimulatedMarketEnv(data, initial_investment)
        state_size = sim_env.state_dim
        action_size = len(sim_env.action_space)
        agent = DQNAgent(state_size, action_size)
        my_scaler = get_scaler(sim_env)

        if mode == 'test':
            # then load the previous scaler
            with open(f'{models_folder}/scaler.pkl', 'rb') as f:
                my_scaler = pickle.load(f)

            # make sure epsilon is not 1!
            # no need to run multiple episodes if epsilon = 0, it's deterministic
            agent.epsilon = 0.01

            # load trained weights
            agent.load(f'{models_folder}/linear.npz')

        # play the game num_episodes times
        for e in range(num_episodes):
            t0 = datetime.now()
            val = play_one_episode(agent, sim_env, my_scaler, mode)
            roi = return_on_investment(val, initial_investment)  # Transform to ROI
            dt = datetime.now() - t0
            print(
                f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, episode roi: {roi:.2f} duration: {dt}")
            portfolio_value.append(val)  # append episode end portfolio value

        # save the weights when we are done
        if mode == 'train':
            # save the DQN
            agent.save(f'{models_folder}/linear.npz')

            # save the scaler
            with open(f'{models_folder}/scaler.pkl', 'wb') as f:
                pickle.dump(my_scaler, f)

            # plot losses
            plt.plot(agent.model.losses)
            plt.show()

        # save portfolio value for each episode
        np.save(f'{rewards_folder}/{mode}.npy', portfolio_value)

    else:
        assert(mode == 'run')

        trade_log = pd.DataFrame(columns=['Date', 'Close', 'Symbol'])
        my_balance = bittrex_obj.get_balance('BTC')

        # load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # make sure epsilon is not 1!
        # Set to 0 for purely deterministic
        agent.epsilon = 0.01

        # load trained weights
        agent.load(f'{models_folder}/linear.npz')

        # Loop to play one episode
        t0 = datetime.now()

        while True:

            # Fetch new data
            candle_dict = bittrex_obj.get_candles('USD-BTC', 'minute')
            if candle_dict['success']:
                new_df = process_bittrex_dict(candle_dict)
            else:
                print("Failed to get candle data")
                continue

            df = df.append(new_df)
            df = df.drop_duplicates(['Date'])
            df = df.sort_values(by='Date')
            df.reset_index(inplace=True, drop=True)

            if first_check:
                market_start = price
                first_check = False

            # def buy_limit(self, market, quantity, rate):
            # """
            # Used to place a buy order in a specific market. Use buylimit to place
            # limit orders Make sure you have the proper permissions set on your
            # API keys for this call to work

            # This allows for accessing by index
            # df.reset_index(inplace=True, drop=True) necessary for plotting

            return_on_investment = return_on_investment(account_value, 100)
            print('Account Return: ', return_on_investment, ' %')
            market_performance = return_on_investment(price, market_start)
            print('Market Performance: ', market_performance, ' %')

            time.sleep(60 * 60)  # run the agent...

def plot_rl_rewards(mode):

    a = np.load(f'linear_rl_trader_rewards/{mode}.npy')

    print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

    plt.hist(a, bins=20)
    plt.title(mode)
    plt.show()

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
    return_on_investment = return_on_investment(account_value, init_money)
    print('Account Return: ', return_on_investment, ' %')

    try:
        market_end = df.loc[df.Date == end, 'Close'].to_numpy()[0]
        market_performance = return_on_investment(market_end, market_start)
        print('Market Performance: ', market_performance, ' %')
    except IndexError:
        print('No date in DF == to end')

    # plot_history(df, start, return_on_investment, market_performance)
    plot_history(df, start)
    return return_on_investment
