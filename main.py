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
import math
import requests
import urllib.request

from statistics import mean
from sklearn.preprocessing import StandardScaler

"""Whats Bruce working on?
-why are all agents giving inconsistent test results?
-mirror the datastructure of the sim env in the real env
-MAKE SURE THAT ohl IS GETTING stripped at the appropriate time/way so it doesnt affect
-Functional, automated trading
    -calculate the actual 'last_action' on inits
    -agent logging
    -test act
    -make a 'live' function, I figure it should go for a day


-Fixed simulated env trading (compare the old way of doing it and validate that the results are the same)
-change feature engineering to better represent how feature engineering works in real time
"""

"""Whats sean working on?
-play one episode should be a part of sim env?

-Plot which currency is held at any given time. Needs to be scalable for multiple assests? May need to reevaluate how an agents performance is evaluated during testing.

-enable data scraping to handle multiple currencies -- see dataframe.join method or maybe merge

-Better feature engineering (more features, different features, more, less, DERIVATIVES OF FEATURES OF MULTIPLE ORDERS)
    -read up on technical analysis, 'indicators'

-look up hedge fund stucturing

-incorporate features into the environment classes (we want features to auto update when running in real time)
"""


"""OTHER FEATURES
-make granularity work with live
-make state reflect the percentage change instead of amount of change in price
-understand pass by reference object well, and make sure that I am doing it right. I think this may be why the code is so slow
-make _add_features also update the self.n_features
-fix an error in renko

-incorporate delayed trading (std dev?) (unnecessary if granularity is sufficiently large, say 10 min)
-be clear about episode/epoch terminology
-let the agent give two orders, a limit and stop
-model slippage based on trading volume (need data on each currencies order book to model this). Also maybe non essential
-fabricate simple data to train on to validate learning (why tho)

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
    return round(final / initial - 1, 4) * 100


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
    print(state) #validated type is numpy.ndarray()
    if agent.name == 'dqn': state = scaler.transform([state])
    print(state)

    done = False

    while not done:
        action = agent.act(state)
        # print(action)
        next_state, val, reward, done = env.step(action)

        if agent.name == 'dqn': next_state = scaler.transform([next_state])
        if is_train in ['train']:
            agent.train(state, action, reward, next_state, done)

        state = next_state

    return val


def run_agent_sim(mode, path_dict, start_date, end_date, num_episodes):
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

    sim_env = SimulatedCryptoExchange(start_date, end_date, initial_investment)
    # sim_env.save_data(path_dict)
    # print(sim_env.df.head())

    state_size = sim_env.state_dim
    action_size = len(sim_env.action_space)

    agent = DQNAgent(state_size, action_size)
    # agent = SimpleAgent(state_size, action_size)
    my_scaler = get_scaler(sim_env)

    if mode == 'test':
        print('Testing...')
        num_episodes = 5
        if agent.name == 'dqn':
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
        print(f"time remaining: {time_remaining + timedelta(seconds = 3)}")
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


def run_agents_live(mode, start_date, end_date):
    assert(mode == 'run')

    #Prepare the agent

    # make sure epsilon is not 1!
    # Set to 0 for purely deterministic
    if agent.name == 'dqn':
        agent.epsilon = 0
        # load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # load trained weights
        agent.load(f'{models_folder}/linear.npz')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()
    mode = args.mode
    assert mode in ["train", "test", "run"]

    if mode in ['train']:
        #train

        # start = datetime(2018, 3, 15)
        # end = datetime(2018, 4, 1)
        start = datetime.now() - timedelta(days = 10)
        end = datetime.now()

    elif mode == 'test':
        # start = datetime(2019,12, 14)
        # end = datetime(2019, 12, 28)

        end = datetime.now()
        start = end - timedelta(days = 9)

        # start = datetime(2018, 3, 1)
        # end = datetime(2018, 4, 1)


    run_agent_sim(mode, paths, start, end, 400)
