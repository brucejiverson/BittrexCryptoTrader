from bittrex.bittrex import *
from agents import *
from environments import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

from datetime import datetime, timedelta
import re
import os
import pickle
import math
import urllib.request

from statistics import mean
from sklearn.preprocessing import StandardScaler

"""Whats Bruce working on?
    -fix remove_data
    -make live df state pull from last price and not candlestick
    -search features for optimal parameters
    -train an agent!
    -make data scraping work better
        -cum data shouldnt have download data
        -enable data scraping to handle multiple currencies -- see dataframe.join method or maybe merge
"""

"""Whats sean working on?
    -make state reflect the percentage change instead of amount of change in price
    -mirror act logic in simulation, mirror bid/ask
    -Plot which currency is held at any given time. Needs to be scalable for multiple assests? May need to reevaluate how an agents performance is evaluated during testing.
    -Better feature engineering (more features, different features, more, less, DERIVATIVES OF FEATURES OF MULTIPLE ORDERS)
        -read up on technical analysis, 'indicators'
    -look up hedge fund stucturing
"""

"""OTHER FEATURES
    -agent logging
    -make granularity work with live
    -make state reflect the percentage change instead of amount of change in price

    -incorporate delayed trading (std dev?) (unnecessary if granularity is sufficiently large, say 10 min)
    -be clear about episode/epoch terminology
    -let the agent give two orders, a limit and stop
    -model slippage based on trading volume (need data on each currencies order book to model this). Also maybe non essential

Big Picture:
    -Deep learning?
    -Infrastructure :/ this is expensive and maybe impractical
    -Trading multiple currencies
"""

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'I made a directory at {directory}')


def ROI(initial, final):
    # Returns the percentage increase/decrease
    return round(100* final / initial - 1, 4)


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


def get_scaler(env):
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here
    """From video:
    Need some data --> play an episode randomly
    Running for multiple episodes will make more accurate
    """

    #Create a list of all of the states to gather mean/variance data on
    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, val, reward, done = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)

    return scaler


def play_one_episode(agent, env, scaler, is_train):
    # note: after transforming states are already 1xD

    state, val  = env.reset()
    try:
        if agent.name == 'dqn': state = scaler.transform([state])
    except ValueError:
        print('State size has changed. Revert or retrain.')

    done = False

    while not done:
        # print(state[2:])
        action = agent.act(state)
        # print(action)
        next_state, val, reward, done = env.step(action)
        if agent.name == 'dqn': next_state = scaler.transform([next_state])

        if is_train in ['train'] and agent.name == 'dqn':
            agent.train(state, action, reward, next_state, done)

        state = next_state

    return val


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()
    mode = args.mode
    assert mode in ["train", "test", "run"]

    models_folder = paths['models']
    rewards_folder = paths['rewards']

    if mode in ['train']:
        #train

        # start = datetime(2018, 3, 15)
        # end = datetime(2018, 4, 1)
        start = datetime.now() - timedelta(days = 9)
        end = datetime.now() - timedelta(days = 4)
        num_episodes = 500

    elif mode == 'test':
        print('Testing...')
        # start = datetime(2019,12, 14)
        # end = datetime(2019, 12, 28)

        end = datetime.now()
        start = end - timedelta(days = 9)
        num_episodes = 1
        # start = datetime(2018, 3, 1)
        # end = datetime(2018, 4, 1)

    # maybe_make_dir(models_folder)
    # maybe_make_dir(rewards_folder)

    # variable for storing final value of the portfolio (done at end of episode)
    portfolio_value = []

    def return_on_investment(final, initial):
        # Returns the percentage increase/decrease
        return round(final / initial - 1, 4) * 100

    batch_size = 32  # sampleing from replay memory
    initial_investment = 100
    sim_env = SimulatedCryptoExchange(start, end, initial_investment = initial_investment)
    # sim_env.save_data()
    # print(sim_env.df.head())

    state_size = sim_env.state_dim
    action_size = len(sim_env.action_space)

    agent = MeanReversionAgent()
    # agent = EMAReversion(state_size, action_size)
    # agent = RegressionAgent(state_size, action_size)
    # agent = DQNAgent(state_size, action_size)
    # agent = SimpleAgent(state_size, action_size)
    my_scaler = get_scaler(sim_env)

    if agent.name == 'dqn' and mode == 'test':
        # then load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            my_scaler = pickle.load(f)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon_min = 0.00#1
        agent.epsilon = agent.epsilon_min

        # load trained weights
        agent.load(f'{models_folder}/linear.npz')

    time_remaining = timedelta(hours=0)
    #Print out some info about the assets
    for market in sim_env.markets:
        token = market[4:7]
        market_roi = return_on_investment(sim_env.df[token + 'Close'].iloc[-1], sim_env.df[token + 'Close'].iloc[0])
        print(f'The {market} pair changed by {market_roi} % over the designated period.')

    # play the game num_episodes times
    for e in range(num_episodes):

        t0 = datetime.now()

        #If you're playing the last episode, create a log
        if e == num_episodes - 1:
            sim_env.should_log = True
        else: #In case for some reason we record a log and then switch back
            sim_env.should_log = False

        val = play_one_episode(agent, sim_env, my_scaler, mode)

        roi = return_on_investment(val, initial_investment)  # Transform to ROI
        dt = datetime.now() - t0

        time_remaining -= dt
        time_remaining = time_remaining + \
            (dt * (num_episodes - (e + 1)) - time_remaining) / (e + 1)

        print(f"episode: {e + 1}/{num_episodes}, ", end = ' ')
        print(f"end value: {val:.2f}, episode roi: {roi:.2f}%, ", end = ' ')
        print(f"time remaining: {time_remaining + timedelta(seconds = 5)}")
        portfolio_value.append(val)  # append episode end portfolio value

    sim_env.plot_market_data()
    sim_env.plot_agent_history()
    # save the weights when we are done
    if mode in ['train']:
        # save the DQN
        agent.save(f'{models_folder}/linear.npz')
        print('DQN saved.')

        # save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(my_scaler, f)
        print('Scaler saved.')
        # plot losses
        plt.plot(agent.model.losses) #this plots the index on the x axis and he loss on the y

    # save portfolio value for each episode
    print('Saving rewards...')
    np.save(f'{rewards_folder}/{mode}.npy', portfolio_value)
    print('Rewards saved.')

    plt.show()
