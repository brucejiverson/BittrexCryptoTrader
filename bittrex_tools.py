from bittrex.bittrex import *

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from datetime import datetime, timedelta
import json
import re
import os
import pickle
import itertools
from empyrical import sortino_ratio

from statistics import mean

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
    df = df.rename(columns={'T': "Date", 'C': 'BTCUSD'})

    # reorder the columns (defaults to alphabetic)
    df = df[['Date', 'BTCUSD']]
    df.reset_index(inplace=True, drop=True)
    # dates into datetimess
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%dT%H:%M:%S")
    return df


def get_candles(bittrex_obj, df, market, granularity):
    # Tested with 'oneMin', and it returned exactly 10 days worth of data (14400 entries)

    attempts = 0
    while True:
        print('Fetching candles from Bittrex...')
        candle_dict = bittrex_obj.get_candles(
            market, 'oneMin')

        if candle_dict['success']:
            df = process_bittrex_dict(candle_dict)
            print("Success getting candle data.")
            break
        else:
            print("Failed to get candle data.")
            time.sleep(1)
            print('Retrying...')

    return df


def format_df(input_df):
    input_df = input_df[['Date', 'BTCUSD']]
    input_df.drop_duplicates(subset='Date', inplace=True)
    input_df = input_df[input_df['BTCUSD'].notnull()]  # Remove non datapoints from the set
    input_df.sort_values(by='Date', inplace=True)
    input_df.reset_index(inplace=True, drop=True)
    return input_df


def fetch_historical_data(path_dict, market, start_date, end_date, bittrex_obj):
    # this function is useful as code is ran for the same period in backtesting several times consecutively,
    # and fetching from original CSV takes longer as it is a much larger file

    print('Fetching historical data...')

    # get the historic data
    path = path_dict['updated history']

    df = pd.DataFrame(columns=['Date', 'BTCUSD'])

    # Fetch candle data from bittrex
    if end_date > datetime.now() - timedelta(days=10):
        candle_df = get_candles(bittrex_obj, df, market, 'oneMin')
        df = df.append(candle_df)

    if df.empty or df.Date.min() > start_date:  # try to fetch from updated
        print('Fetching data from cumulative data repository.')

        def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p-%M")
        up_df = pd.read_csv(path, usecols=['Date', 'BTCUSD'],
                            parse_dates=['Date'], date_parser=dateparse)

        if up_df.empty:
            print('Cumulative data repository was empty.')
        else:
            print('Success fetching from cumulative data repository.')
            df = df.append(up_df)

    if df.empty or df.Date.min() > start_date:  #Fetch from download file (this is last because its slow)

        print('Fetching data from the download file.')
        # get the historic data

        def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
        orig_df = pd.read_csv(path_dict['downloaded history'], usecols=['Timestamp', 'Close'], parse_dates=[
            'Timestamp'], date_parser=dateparse)
        orig_df.rename(columns={'Timestamp': 'Date', 'Close': 'BTCUSD'}, inplace=True)


        assert not orig_df.empty

        df = df.append(orig_df)

    # Double check that we have a correct date date range. Note: will still be triggered if missing the exact data point
    assert(df.Date.min() <= start_date)

    if df.Date.max() < end_date:
        print('There is a gap between the download data and the data available from Bittrex. Please update download data.')
        assert(df.Date.max() >= end_date)

    df = df[df['Date'] >= start_date]
    df = df[df['Date'] <= end_date]
    df = format_df(df)

    return df


def filter_error_from_download_data(my_df):
    print('Filtering data for errors...')
    for i, row in my_df.iterrows():
        if i > 0 and i < len(my_df.Date) - 2:
            try:
                if my_df.loc[i, 'BTCUSD'] < 0.5 * mean([my_df.loc[i - 1, 'BTCUSD'], my_df.loc[i + 1, 'BTCUSD']]):
                    my_df.drop(i, axis=0, inplace=True)
                    print('Filtered a critical point.')
            except KeyError:
                print(i, len(my_df.Date))
    my_df = format_df(my_df)
    return my_df


def save_historical_data(path_dict, df):
    # This function writes the information in the original format to the csv file
    # including new datapoints that have been fetched

    print('Writing data to CSV.')

    path = path_dict['updated history']
    # must create new df as df is passed by reference
    # # datetimes to strings
    # df = pd.DataFrame({'Date': data[:, 0], 'BTCUSD': np.float_(data[:, 1])})   #convert from numpy array to df

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
    price = df['BTCUSD'].values  # returns an np price

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

    #add the momentum indicators

def make_price_history_static(df):
    # df['Static'] = np.log(df['BTCUSD']) - np.log(df['BTCUSD']).shift(1)
    # df['Static'].iloc[0] = 0
    df['Static'] = df['BTCUSD']
    df = df[['Date', 'BTCUSD', 'Static']]


def ROI(initial, final):
    # Returns the percentage increase/decrease
    return round(final / initial - 1, 4) * 100


def plot_history(df, log, roi=0):
    # df = pd.DataFrame({'Date': data[:, 0], 'BTCUSD': np.float_(data[:, 1])})

    assert not df.empty
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)  # Create the figure

    market_perf = ROI(df.BTCUSD.iloc[0], df.BTCUSD.iloc[-1])
    fig.suptitle('Market performance: ' + str(market_perf), fontsize=14, fontweight='bold')
    df.plot(x='Date', y='BTCUSD', ax=ax1)

    for col in df.columns:
        if col[0:3] == 'SMA':
            df.plot(x='Date', y=col, ax=ax3)

    # df.plot(x='Date', y='Account Value', ax=ax)

    log['Date'] = df.Date
    # log.plot(x='Date', y='BTCUSD', ax=ax)
    log.plot(x='Date', y='Value', ax=ax2)
    # df.plot(x='Date', y='AccounV Value', ax=ax)

    bot, top = plt.ylim()
    cushion = 200
    plt.ylim(bot - cushion, top + cushion)
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
      - static price of asset 1 (using BTCUSD price)
      - associated indicators for each asset
    """

    def __init__(self, data, initial_investment=100):
        # data
        self.asset_data = data
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
        self.cur_state = None
        # self.cur_position = None
        self.cash_in_hand = None

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

        # calculate size of state (amount of each asset held, value of each asset, cash in hand)
        self.state_dim = self.n_asset + self.n_indicators*self.n_asset   #State is the static data and the indicators for each asset (currently ignoring volume)

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
        self.cur_state = self.asset_data[self.cur_step][self.n_asset:]#(self.n_asset + self.n_indicators)]

        self.cur_position = self._update_pos()

        self.cash_in_hand = self.initial_investment

        # print(self.cur_state)
        # print(self.asset_prices)
        # print(self.n_asset)
        # print('reset')
        # time.sleep(100)

        return self.cur_state, self._get_val() # Return the state vector (same as obervation for now)

    def step(self, action):
        # Performs an action in the enviroment, and returns the next state and reward

        if not action in self.action_space:
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
        self.cur_state = self.asset_data[self.cur_step][self.n_asset:]

        # perform the trade
        self._trade(action)
        self._update_pos()

        # get the new value after taking the action
        cur_val = self._get_val()
        cur_price = self.asset_prices[0]
        # reward is the increase in porfolio value
        delta = (cur_price - prev_price)/prev_price #Percentage change in price
        reward = delta*(self.assets_owned[0]*self.asset_prices[0] - self.cash_in_hand)/cur_val
        # self.rewards_hist = np.roll(self.rewards_hist,1)
        # self.rewards_hist[0] = step_return
        # reward = sortino_ratio(self.rewards_hist)

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # store the current value of the portfolio here
        info = {'cur_val': cur_val}

        # conform to the Gym API
        #      next state       reward  flag  info dict.
        return self.cur_state, self._get_val(), reward, done, info

    def _update_pos(self):
        # Returns the state (for now state, and observation are the same.
        # Note that the state could be a transformation of the observation, or
        # multiple past observations stacked.)
        self.cur_position = np.empty(self.n_asset + 1)  #assets_owned, USD
        # self.cur_position[0:self.n_asset] = self.asset_prices
        self.cur_position[0:self.n_asset] = self.assets_owned
        self.cur_position[-1] = self.cash_in_hand


    def _get_val(self):
        return self.assets_owned.dot(self.asset_prices) + self.cash_in_hand

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
            # self.cash_in_hand -= delta[0]*val


            # Sell everything
            for i, a in enumerate(action_vec):
                self.cash_in_hand += self.assets_owned[i] * self.asset_prices[i]
                self.assets_owned[i] = 0

            # Buy back the right amounts
            for i, a in enumerate(action_vec):
                cash_to_use = a * self.cash_in_hand
                self.assets_owned[i] = cash_to_use / self.asset_prices[i]
                self.cash_in_hand -= cash_to_use

            self.last_action = action_vec
            self.period_since_trade = 0
        else:
            self.period_since_trade += 1


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
        self.epsilon_min = 0.15  # originally .01. The version here is set for training
        self.epsilon_decay = 0.9995  # originall .995
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


def run_agent(mode, df, bittrex_obj, path_dict, symbols='USDBTC'):
    # Mode should be a string, either train or test or run
    # maybe it would be helpful to run this through command line argv etc
    models_folder = path_dict['models']
    rewards_folder = path_dict['rewards']

    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    # variable for storing final value of the portfolio (done at end of episode)
    portfolio_value = []

    data_to_fit = df.drop('Date', axis = 1).values

    def return_on_investment(final, initial):
        # Returns the percentage increase/decrease
        return round(final / initial - 1, 4) * 100

    # n_train = n_timesteps // 2 #floor division splitting the data into training and testing

    num_episodes = 50
    batch_size = 32  # sampleing from replay memory
    initial_investment = 100
    # fees = 0.0025  # standard fees are 0.3% per transaction

    if mode in ['train', 'test', 'add_train']:


        sim_env = SimulatedMarketEnv(data_to_fit, initial_investment)
        state_size = sim_env.state_dim
        action_size = len(sim_env.action_space)
        agent = DQNAgent(state_size, action_size)
        my_scaler = get_scaler(sim_env)

        if mode == 'test':
            print('Testing...')
            num_episodes = 2
            # then load the previous scaler
            with open(f'{models_folder}/scaler.pkl', 'rb') as f:
                my_scaler = pickle.load(f)

            # make sure epsilon is not 1!
            # no need to run multiple episodes if epsilon = 0, it's deterministic
            agent.epsilon_min = 0
            agent.epsilon = agent.epsilon_min

            # load trained weights
            agent.load(f'{models_folder}/linear.npz')
        elif mode == 'add_train':
            # then load the previous scaler
            with open(f'{models_folder}/scaler.pkl', 'rb') as f:
                my_scaler = pickle.load(f)

            # load trained weights
            agent.load(f'{models_folder}/linear.npz')
            print('Last scalar and trained weights loaded.')

        time_remaining = timedelta(hours=0)

        # play the game num_episodes times
        for e in range(num_episodes):

            if e == range(num_episodes)[-1]:        #Setting this so that the very last playthrough is purely deterministic
                agent.epsilon_min = 0
                agent.epsilon = agent.epsilon_min

            t0 = datetime.now()

            if e == num_episodes - 1:
                val, state_log = play_one_episode(agent, sim_env, my_scaler, mode, True)
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

            if num_episodes <= 500 or e % 10 == 0:
                print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, episode roi: {roi:.2f}, end time: {(datetime.now() + time_remaining).time()}")
            portfolio_value.append(val)  # append episode end portfolio value

        # save the weights when we are done
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
            plt.plot(agent.model.losses)
            plt.show()

        # save portfolio value for each episode
        print('Saving rewards...')
        np.save(f'{rewards_folder}/{mode}.npy', portfolio_value)
        print('Rewards saved.')

        return state_log

    else:
        assert(mode == 'run')

        my_balance = bittrex_obj.get_balance('BTC')

        # load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # make sure epsilon is not 1!
        # Set to 0 for purely deterministic
        agent.epsilon = agent.epsilon_min

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
