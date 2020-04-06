from bittrex.bittrex import *
from agents import *
from environments import *

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


"""Whats Bruce working on?
-get all plots to show at the same time
-breaking code apart into files, better file management, more official commenting
-better data
-Fixed simulated env trading (compare the old way of doing it and validate that the results are the same)
-change feature engineering to better represent how feature engineering works in real time
-Start regularly scraping data for future training
"""

"""Whats sean working on?

-Plot which currency is held at any given time

"""


"""DESIRED FEATURES
-integrate sentiment with features
-be clear about episode/epoch terminology
-let the agent give two orders, a limit and stop
-Functional, automated trading
-model slippage based on trading volume (need data on each currencies order book to model this). Also maybe non essential
-fabricate simple data to train on to validate learning
-Data for multiple currencies
-understand pass by reference object well, and make sure that I am doing it right. I think this may be why the code is so slow
-Understand plotting losses. Possibly switch to plotting profitablilty over course of training

Big Picture:
-Deep learning?
-Better feature engineering
-multiple currencies (need data, etc)
-Infrastructure :/ this is expensive and maybe impractical
-Trading multiple currencies
"""



def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'I made a directory at {directory}')


def process_candle_dict(candle_dictionary):
    # Dataframe formatted the same as
    # V and BV refer to volume and base volume
    df = pd.DataFrame(candle_dictionary['result'])
    df.drop(columns=["BV"])
    df = df.rename(columns={'T': "Date", 'O': 'BTCOpen', 'H': 'BTCHigh', 'L': 'BTCLow', 'C': 'BTCClose', 'V': 'BTCVolume'})

    df = df[['Date', 'BTCOpen', 'BTCHigh', 'BTCLow', 'BTCClose', 'BTCVolume']]  #Reorder
    df.reset_index(inplace=True, drop=True)
    # dates into datetimess
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%dT%H:%M:%S")
    return df


def ROI(initial, final):
    # Returns the percentage increase/decrease
    return round(final / initial - 1, 4) * 100

#def log_ROI(initial, final):
    # Returns the log rate of return, which accounts for how percent changes "stack" over time
    # For example, a 10% increase followed by a 10% decrease is truly a 1% decrease over time (100 -> 110 -> 99)
    # Arithmetic ROI would show an overall trend of 0%, but log ROI properly computes this to be -1%
    #return round(np.log(final/initial), 4) *100


def format_df(input_df):
    #Note that this should only be used before high low open are stripped from the data
    # input_df = input_df[['Date', 'BTCClose']]
    formatted_df = input_df.drop_duplicates(subset='Date', inplace=False)
    formatted_df = formatted_df[formatted_df['BTCClose'].notnull()]  # Remove non datapoints from the set
    formatted_df.sort_values(by='Date', inplace=True)   #This was causing a warning about future deprecation/changes to pandas
    formatted_df.reset_index(inplace=True, drop=True)
    formatted_df = input_df[['Date', 'BTCOpen', 'BTCHigh', 'BTCLow', 'BTCClose', 'BTCVolume']]  #Reorder

    return formatted_df

def fetch_sentiment(limit):
    """This function pulls sentiment data from the crypto fear and greed index. That data is updated daily.
    This is the link to the website: https://alternative.me/crypto/fear-and-greed-index/#fng-history
    The 'limit' argument is the number of data points to fetch (one for each day).
    The given value is on a scale of 0 - 100, with 0 being extreme fear and 100 being extreme greed."""

    url = "https://api.alternative.me/fng/?limit="+ str(limit) +"&date_format=us"

    data = requests.get(url).json()["data"] #returns a list of dictionaries

    sentiment_df = pd.DataFrame(data)

    #Drop unnecessaary columns
    sentiment_df.drop(columns = ["time_until_update", "value_classification"], inplace = True)
    #Rename the columns
    sentiment_df.rename(columns={'timestamp': 'Date', 'value': 'Value'}, inplace = True)
    #Format the dates
    sentiment_df['Date'] = pd.to_datetime(sentiment_df["Date"], format = "%m-%d-%Y")
    #Convert value to int, and center the sentiment value at 0
    sentiment_df["Value"] = sentiment_df['Value'].apply(int)
    sentiment_df['Value'] = sentiment_df['Value'] - 50

    for i, row in sentiment_df.iterrows():
        for minute in range(1, 60):
            new_row = {'Date': , "Value": float('nan')}

    # print(sentiment_df)
    return sentiment_df

def fetch_historical_data(path_dict, markets, start_date, end_date, bittrex_obj):
    """This function pulls data from the exchange, the cumulative repository, and the data download in that order
    depending on the input date range."""

    print('Fetching historical data...')

    cols = ["Date"]
    for market in markets:
        for item in ['Open', 'High', 'Low', 'Close', 'Volume']:
            cols.append(market[0:3] + item)

    df = pd.DataFrame(columns=cols)

    # Fetch candle data from bittrex
    if end_date > datetime.now() - timedelta(days=9):
        for market in markets:
            print('Market: ' + market)
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
                    print(candle_dict)
                    time.sleep(2*attempts)
                    attempts +=1

                    if attempts == 5:
                        raise(TypeError)
                    print('Retrying...')

        df = df.append(candle_df)

    path = path_dict['updated history']

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
    return input_df #same


def change_df_granulaty(input_df, gran):
    """This function looks at the Date columns of the df and modifies the df according to the input granularity (in minutes)"""
    print('Changing data granularity from 1 minute to '+ str(gran) + ' minutes.')

    if input_df['Date'].iloc[1] - input_df['Date'].iloc[0] == timedelta(minutes = 1): #verified this works
        print('here')
        return input_df.iloc[::gran, :]
    else:
        print('Granularity of df input to change_df_granularity was not 1 minute.')
        raise(ValueError)


def strip_df_open_high_low(input_df):

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


def add_features(input_df): #This didnt exist in this file, added it from the broken
    """ If you change the number of indicators in this function, be sure to also change the expected number in the enviroment"""

    def add_sma_as_column(mydata, p):
        # p is a number
        price = mydata['BTCClose'].values  # returns an np price, faster

        sma = np.empty_like(price)
        for i, item in enumerate(np.nditer(price)):
            if i == 0:
                sma[i] = item
            elif i < p:
                sma[i] = price[0:i].mean()
            else:
                sma[i] = price[(i - p):i].mean()

        # subtract
        indicator = np.empty_like(sma)
        for i, item in enumerate(np.nditer(price)):
            indicator[i] = sma[i] - price[i]

        mydata['SMA_' + str(p)] = indicator  # modifies the input df

    def add_renko(mydata, blocksize):
        #reference for how bricks are calculated https://www.tradingview.com/wiki/Renko_Charts
        # p is a number
        prices = mydata['BTCClose'].values  # returns an np price, faster

        renko = np.empty_like(prices)

        indicator_val = 0
        #Loop to calculate the renko value at each data point
        for i, price in enumerate(np.nditer(prices)):
            if i == 0:
                upper_thresh = price + blocksize
                lower_thresh = price - blocksize
            elif price <= lower_thresh: #create a down block

                indicator_val -= blocksize #continuing a downtrend
                lower_thresh -= blocksize
                upper_thresh = lower_thresh + 3*blocksize

            elif price >= upper_thresh: #create an up block

                indicator_val += blocksize #continuing an uptrend
                upper_thresh += blocksize
                lower_thresh = upper_thresh - 3*blocksize

            renko[i] = indicator_val

        period = 2
        indicator = np.empty_like(prices)

        #Loop to interpret the renko to be more useful
        for i, item in enumerate(renko):
            if i == 0:
                indicator[i] = item
            elif i < period:
                indicator[i] = renko[0:i].mean()
            else:
                indicator[i] = renko[(i - period):i].mean() - renko[i-period]


        mydata['Renko'] = renko

    # base = 50
    # add_sma_as_column(input_df, base)
    # add_sma_as_column(input_df, int(base*8/5))
    # add_sma_as_column(input_df, int(base*13/5))
    add_renko(input_df, 70)

    input_df['BTCMACD'] = ta.trend.macd_diff(input_df['BTCClose'], fillna  = True)
    input_df['BTCRSI'] = ta.momentum.rsi(input_df['BTCClose'], fillna  = True)
    input_df['BTCRSI'] = input_df['BTCRSI'] - 50 #center at 0
    # input_df['BTCOBV'] = ta.volume.on_balance_volume(input_df['BTCClose'], input_df['BTCVolume'], fillna  = True)


    input_df = format_df(input_df)


def plot_data(df):


    # I had a ton of trouble getting the plots to look right with the dates.
    # This link was really helpful http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html
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


def plot_sim_trade_history(df, log, roi=0): #updated based on the "broken file."
    # df = pd.DataFrame({'Date': data[:, 0], 'BTCClose': np.float_(data[:, 1])})

    assert not df.empty
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)  # Create the figure

    market_perf = ROI(df.BTCClose.iloc[0], df.BTCClose.iloc[-1])
    fig.suptitle('Market performance: ' + str(market_perf), fontsize=14, fontweight='bold')
    df.plot(x='Date', y='BTCClose', ax=ax1)

    for col in df.columns:
        if not col[3:] in ['Open', 'High', 'Low', 'Close'] and not col in ['Date', 'BTCVolume']:
            df.plot(x='Date', y=col, ax=ax3)

    # df.plot(x='Date', y='Account Value', ax=ax)

    # log['Date'] = df.Date
    # my_roi = ROI(log.Value.iloc[0], log.Value.iloc[-1])
    #
    # log.plot(x='Date', y='Value', ax=ax2)

    my_roi = ROI(log.Value.iloc[0], log.Value.iloc[-1])
    sharpe = my_roi/log.Value.std()
    print(f'Sharpe Ratio: {sharpe}') #one or better is good
    log.plot(y='Value', ax=ax2)


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
        # print(action)
        next_state, val, reward, done, info = env.step(action)

        if record == True:
            log = log.append(pd.DataFrame.from_records(
                [dict(zip(log_columns, [val]))]), ignore_index=True)

        next_state = scaler.transform([next_state])
        if is_train in ['train']:
            agent.train(state, action, reward, next_state, done)
        state = next_state

    if record:
        return info['cur_val'], log
    else:
        return info['cur_val']


def run_agent_sim(mode, path_dict, start_date, end_date, num_episodes, symbols=['USDBTC']):
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

    batch_size = 32  # sampleing from replay memory
    initial_investment = 100

    print('Preparing data...')
    #get my keys
    with open(path_dict['secret']) as secrets_file:
        keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
        secrets_file.close()

    my_bittrex2_0 = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

    markets = [sym[3:6] + '-' + sym[0:3] for sym in symbols]

    df = fetch_historical_data(path_dict, markets, start_date, end_date, my_bittrex2_0)  # oldest date info
    # save_historical_data(path_dict, df)

    print("ORIGINAL DATA: ")
    print(df.head())
    df = change_df_granulaty(df, 5)
    add_features(df)
    df = strip_df_open_high_low(df)
    print("DATA TO RUN ON: ")
    print(df.head())

    data_to_fit = df.drop('Date', axis = 1).values

    sim_env = SimulatedCryptoExchange(data_to_fit, initial_investment)
    state_size = sim_env.state_dim
    action_size = len(sim_env.action_space)
    dqn_agent = DQNAgent(state_size, action_size)
    my_scaler = get_scaler(sim_env)
    if mode == 'test':
        print('Testing...')
        num_episodes = 5
        # then load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            my_scaler = pickle.load(f)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        dqn_agent.epsilon_min = 0.0005
        dqn_agent.epsilon = dqn_agent.epsilon_min

        # load trained weights
        dqn_agent.load(f'{models_folder}/linear.npz')

    time_remaining = timedelta(hours=0)
    market_roi = return_on_investment(df.BTCClose.iloc[-1], df.BTCClose.iloc[0])
    print(f'The market changed by {market_roi} % over the designated period.')

    # play the game num_episodes times
    for e in range(num_episodes):

        t0 = datetime.now()

        if e == num_episodes - 1:
            val, state_log = play_one_episode(dqn_agent, sim_env, my_scaler, mode, True)
        else:
            val = play_one_episode(dqn_agent, sim_env, my_scaler, mode)

        roi = return_on_investment(val, initial_investment)  # Transform to ROI
        dt = datetime.now() - t0

        time_remaining -= dt
        time_remaining = time_remaining + \
            (dt * (num_episodes - (e + 1)) - time_remaining) / (e + 1)
        if e % 100 == 0 and mode in ['train']: # save the weights when we are done
            # save the DQN
            dqn_agent.save(f'{models_folder}/linear.npz')
            print('DQN saved.')

            print('Saving scaler...')
            # save the scaler
            with open(f'{models_folder}/scaler.pkl', 'wb') as f:
                pickle.dump(my_scaler, f)
            print('Scaler saved.')

        print(f"episode: {e + 1}/{num_episodes}, end value: {val:.2f}, episode roi: {roi:.2f}, time remaining: {time_remaining}")
        portfolio_value.append(val)  # append episode end portfolio value

    # save the weights when we are done
    if mode in ['train']:
        # save the DQN
        dqn_agent.save(f'{models_folder}/linear.npz')
        print('DQN saved.')

        # save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(my_scaler, f)
        print('Scaler saved.')
        # plot losses
        plt.plot(dqn_agent.model.losses) #this plots the index on the x axis and he loss on the y

    # save portfolio value for each episode
    print('Saving rewards...')
    np.save(f'{rewards_folder}/{mode}.npy', portfolio_value)
    print('Rewards saved.')

    plot_sim_trade_history(df, state_log)


def run_agents_live(mode, path_dict, start_date, end_date, num_episodes, symbols=['USDBTC']):
    assert(mode == 'run')

    #Prepare the agent
    # load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # make sure epsilon is not 1!
    # Set to 0 for purely deterministic
    agent.epsilon = agent.epsilon_min

    # Price in USD, price per unit is $/BTC

    is_USD = True

    log = pd.DataFrame()

    # Note that bittrex exchange is based in GMT 8 hours ahead of CA
    trade_incomplete = True

    # Enter a trade into the market.
    # Example result  {'success': True, 'message': '', 'result': {'uuid': '2641035d-4fe5-4099-9e7a-cd52067cde8a'}}
    ticker = self.bittrex_obj_1_1.get_ticker(market)
    price = ticker['result']['Last']
    amount = amount / price  # Convert to the amount of BTC

    if is_USD:  # buy
        trade_result = self.bittrex_obj_1_1.buy_limit(market, amount, round(price*1.0001, 3))
        side = 'buying'
    else:       # Sell
        trade_result = my_bittrex1_1.sell_limit(market, amount, round(price*0.9999, 3) )
        side = 'selling'

    # Check that an order was entered
    if not trade_result['success']:
        print('Trade attempt failed')
        print(trade_result['message'])
        #Start again?

    print(f'Order for {side} {amount:.8f} {symbols[0:3]} at a price of {price:.3f} has been submitted to the market.')
    order_uuid = trade_result['result']['uuid']

    # Loop to see if the order has been filled
    status = trade_is_executed(order_uuid, order_data, my_bittrex)

    if status == True:
        is_USD = not is_USD
        print(f'Order has been filled. Id: {order_uuid}.')


    print(f'Attempt was not filled. Attempting to order again.')

    dt = datetime.now() - start_time  # note that this include the time to run a small amount of code

    try: #Some weird error here that I have not been able to recreate. Added print statements for debugging if it occurs again
        order_data['result']['Order Duration'] = dt.total_seconds()
    except TypeError:
        print(dt)
        print(dt.total_seconds())
    trade = process_order_data(order_data)
    log = log.append(trade, ignore_index=True)
    # log.reset_index(inplace=True)

    # load trained weights
    agent.load(f'{models_folder}/linear.npz')

    # Loop to play one episode
    t0 = datetime.now()

    while True:
        # Fetch new data
        candle_dict = my_bittrex1_1.get_candles('USD-BTC', 'minute')
        if candle_dict['success']:
            new_df = process_candle_dict(candle_dict)
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

        # This allows for accessing by index
        # df.reset_index(inplace=True, drop=True) necessary for plotting

        return_on_investment = return_on_investment(account_value, 100)
        print('Account Return: ', return_on_investment, ' %')
        market_performance = return_on_investment(price, market_start)
        print('Market Performance: ', market_performance, ' %')

#cryptodatadownload has gaps
#Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********
os = 'windows' #linux or windows
symbols = ['BTCUSD'] #Example: 'BTCUSD'

#The below should be updated to be simplified to use parent directory? unsure how that works...
#https://stackoverflow.com/questions/48745333/using-pandas-how-do-i-save-an-exported-csv-file-to-a-folder-relative-to-the-scr?noredirect=1&lq=1

if os == 'linux':
    paths = {'downloaded history': '/home/bruce/AlgoTrader/BittrexTrader/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
    'updated history': '/home/bruce/AlgoTrader/updated_history_' + symbols[0] + '.csv',
    'secret': "/home/bruce/Documents/crypto_data/secrets.json",
    'rewards': 'agent_rewards',
    'models': 'agent_models',
    'test trade log':  'C:/Python Programs/crypto_trader/historical data/trade_testing' + symbols[0] + '.csv'}

    # TODO: add a loop here that appends the asset folders

elif os == 'windows':
    paths = {'downloaded history': 'C:/Python Programs/crypto_trader/historical data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv',
    'updated history': 'C:/Python Programs/crypto_trader/historical data/updated_history_' + symbols[0] + '.csv',
    'secret': "/Users/biver/Documents/crypto_data/secrets.json",
    'rewards': 'agent_rewards',
    'models': 'agent_models',
    'test trade log':  'C:/Python Programs/crypto_trader/historical data/trade_testing' + symbols[0] + '.csv'}
else:
    print('Unknown OS passed when defining the paths')  # this should throw and error
    assert(os in ['windows', 'linux'])



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()
    mode = args.mode
    assert mode in ["train", "test", "run"]

    if mode in ['train']:
        #train
        # start = datetime(2019,1, 1)
        # end = datetime(2019, 2, 1)
        start = datetime(2020, 3, 27)
        end = datetime(2020, 4, 5)
        # end = datetime.now() - timedelta(hours = 6)

    elif mode == 'test':
        # start = datetime(2019,12, 14)
        # end = datetime(2019, 12, 28)

        # start = datetime(2020, 3, 27)
        # end = datetime(2020, 4, 5)

        start = datetime(2018, 3, 1)
        end = datetime(2018, 4, 1)


    run_agent_sim(mode, paths, start, end, 800, symbols)
