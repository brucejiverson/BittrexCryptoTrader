from agents.agents import *
from environments.environments import *
from tools.tools import maybe_make_dir, printProgressBar, f_paths, ROI, percent_change_column
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
import itertools

from statistics import mean

def play_one_episode(agent, env, is_train):

    state, val  = env.reset()
    done = False
    # Total number of steps in the simulation
    n_step = env.asset_data.shape[0]
    
    while not done:
        if env.cur_step%10 == 0:
            printProgressBar(env.cur_step, n_step, prefix='Backtest progress:', suffix='Complete')
        action = agent.act(state)
        next_state, val, reward, done = env.step(action)

        if is_train in ['train'] and agent.name == 'dqn':
            agent.train(state, action, reward, next_state, done)

        state = next_state
    printProgressBar(n_step, n_step, prefix='Backtest progress:', suffix='Complete')
    print('')
    return val


def test(features):
    # end = datetime(year=2020, month=7, day=8)
    # start = end - timedelta(hours = 1, days = 1, weeks=4)

    start = datetime(year=2020, month=1, day=1)
    end = datetime(year=2020, month=4, day=1)

    print(f'Backtesting from {start} to {end}.')
    initial_investment = 100
    print(features)
    sim_env = SimulatedCryptoExchange(start, end, granularity=5, feature_dict=features)

    feature_list = sim_env.df.columns
    feature_map = {}
    for i, item in enumerate(feature_list):
        feature_map[item] = i
    action_size = len(sim_env.action_space)
    hyperparams = {'ind base': 5,
                'high thresh': .5,  # these are fraction base on BBInd (custom indicator)
                'low thresh': .5,
                'order': .1,                      # these are percentages for trailing orders
                'loss': 1,
                'stop loss': 7}

    agents = [
            biteBack(feature_map, action_size, hyperparams),
            # knnAgent(feature_map, action_size),
            # MeanReversionAgent(feature_map, action_size),
            # bollingerAgent(feature_map, action_size),
            # doubleReverse(feature_map, action_size),
            # bollingerStateAgent(feature_map, action_size),
            # EMAReversion(feature_map, action_size),
            # RegressionAgent(feature_map, action_size),
            # DQNAgent(feature_map, action_size),
            # SimpleAgent(feature_map, action_size),
            # BuyAndHold(feature_map, action_size)
            ]

    # Load agent models
    for agent in agents:
        if agent.name == 'dqn':
            # make sure epsilon is not 1!
            # no need to run multiple episodes if epsilon = 0, it's deterministic
            agent.epsilon_min = 0.0001
            agent.epsilon = agent.epsilon_min

            # load trained weights
            agent.load(f'{models_folder}/linear.npz')

    #Print out some info about the assets
    for market in sim_env.markets:
        token = market[4:7]
        market_roi = ROI(sim_env.df[token + 'Close'].iloc[0], sim_env.df[token + 'Close'].iloc[-1])
        print(f'The {market} pair changed by {market_roi} % over the designated period.')

    # create a log
    sim_env.should_log = True
    for agent in agents:
        val = play_one_episode(agent, sim_env, mode)
        roi = ROI(initial_investment, val)  # Transform to ROI
        print(f"Agent {agent.name} end value: {val:.2f}, episode roi: {roi:.2f}%, ")
        sim_env.log.to_dataframe()
        s = scorer(sim_env.log.df)
        s.print_scores()
        sim_env.log.plot(sim_env.df, agent.name)

    # sim_env.plot_market_data()
    
    plt.show()


def grid_search(features):

    parameters = {  #'granularity': [30, 60, 120, 240],
                'agent': {'ind base': [2, 3, 4],
                        'high thresh': [0, 0.1, .2, .3, .4, .5],        # these are fraction base on BBInd (custom indicator)
                        'low thresh': [.3, .35, .4, .45, .5],
                        'order': [.1, .25, .5, 1, 1.5],                   # these are percentages for trailing orders
                        'loss':  [.1, .25, .5, 1, 1.5],
                        'stop loss': [None, 6, 8]}
                }
    
    start = datetime(year=2020, month=1, day=1)
    end = datetime(year=2020, month=3, day=1)

    print(f'Backtesting from {start} to {end}.')
    initial_investment = 100
    
    sim_env = SimulatedCryptoExchange(start, end, granularity=5, feature_dict=features)
    sim_env.should_log = True
    feature_list = sim_env.df.columns
    feature_map = {}
    for i, item in enumerate(feature_list):
        feature_map[item] = i
    action_size = len(sim_env.action_space)

    #Print out some info about the assets
    for market in sim_env.markets:
        token = market[4:7]
        market_roi = ROI(sim_env.df[token + 'Close'].iloc[0], sim_env.df[token + 'Close'].iloc[-1])
        print(f'The {market} pair changed by {market_roi} % over the designated period.')

    # create all of the permutations
    params = parameters['agent']

    best_score = -np.inf
    best_params = None
    score_log = []
    keys, values = zip(*params.items())
    for experiment in itertools.product(*values):
        exp_dict = dict(zip(keys, experiment))
        print(f'Preprocessing data for hyperparameters: {exp_dict}')

        # Initialize the agent
        agent = biteBack(feature_map, action_size, hyperparams=exp_dict)
        # agent = MeanReversionAgent(feature_map, action_size)
        # agent = bollingerAgent(feature_map, action_size)
        # agent = doubleReverse(feature_map, action_size)
        val = play_one_episode(agent, sim_env, mode)
        sim_env.log.to_dataframe()
        s = scorer(sim_env.log.df)
        s.print_scores()
        score_log.append([*s.get_scores(), *list(exp_dict.values())])
        if s.custom_score > best_score:
            best_score, best_params = s.custom_score, exp_dict
    
    print(f'Best params: {best_params}')
    agent = biteBack(feature_map, action_size, hyperparams=best_params)
    # create a log
    val = play_one_episode(agent, sim_env, mode)
    roi = ROI(initial_investment, val)  # Transform to ROI
    print(f"Agent {agent.name}")
    sim_env.log.to_dataframe()
    s = scorer(sim_env.log.df)
    s.print_scores()
    sim_env.log.plot(sim_env.df, agent.name)
    score_df = pd.DataFrame(columns=['ROI', 'Sharp', 'Score', *list(best_params.keys())], data=score_log)
    score_df.to_pickle(f_paths['score log'])
    plt.show()


def gradient_search(features):

    parameters = {  #'granularity': [30, 60, 120, 240],
                'agent': {'ind base': [3],
                        'high thresh': [0, 0.1, .2, .3, .4, .5],        # these are fraction base on BBInd (custom indicator)
                        'low thresh': [.3, .4, .5],
                        'order': [.1, .5, 1, 1.5, 2],                   # these are percentages for trailing orders
                        'loss': [.1, .5, 1, 1.5, 2],
                        'stop loss': [None]}
                }
    
    start = datetime(year=2020, month=7, day=22)
    end = datetime(year=2020, month=8, day=2)

    print(f'Backtesting from {start} to {end}.')
    initial_investment = 100
    
    sim_env = SimulatedCryptoExchange(start, end, granularity=5, feature_dict=features)
    sim_env.should_log = True
    feature_list = sim_env.df.columns
    feature_map = {}
    for i, item in enumerate(feature_list):
        feature_map[item] = i
    action_size = len(sim_env.action_space)

    #Print out some info about the assets
    for market in sim_env.markets:
        token = market[4:7]
        market_roi = ROI(sim_env.df[token + 'Close'].iloc[0], sim_env.df[token + 'Close'].iloc[-1])
        print(f'The {market} pair changed by {market_roi} % over the designated period.')

    # create all of the permutations
    params = parameters['agent']

    best_score = -np.inf
    best_params = None
    score_log = []
    keys, values = zip(*params.items())
    for experiment in itertools.product(*values):
        exp_dict = dict(zip(keys, experiment))
        print(f'Preprocessing data for hyperparameters: {exp_dict}')

        # Initialize the agent
        agent = biteBack(feature_map, action_size, hyperparams=exp_dict)
        # agent = MeanReversionAgent(feature_map, action_size)
        # agent = bollingerAgent(feature_map, action_size)
        # agent = doubleReverse(feature_map, action_size)
        val = play_one_episode(agent, sim_env, mode)
        sim_env.log.to_dataframe()
        s = scorer(sim_env.log.df)
        s.print_scores()
        score_log.append([*s.get_scores(), *list(exp_dict.values())])
        if s.custom_score > best_score:
            best_score, best_params = s.custom_score, exp_dict
    
    print(f'Best params: {best_params}')
    agent = biteBack(feature_map, action_size, hyperparams=best_params)
    # create a log
    val = play_one_episode(agent, sim_env, mode)
    roi = ROI(initial_investment, val)  # Transform to ROI
    print(f"Agent {agent.name}")
    sim_env.log.to_dataframe()
    s = scorer(sim_env.log.df)
    s.print_scores()
    sim_env.log.plot(sim_env.df, agent.name)
    score_df = pd.DataFrame(columns=['ROI', 'Sharp', 'Score', *list(best_params.keys())], data=score_log)
    score_df.to_pickle(f_paths['score log'])
    plt.show()


def train(features):
    
    start = datetime.now() - timedelta(days = 9)
    end = datetime.now() - timedelta(days = 4)
    num_episodes = 500
    models_folder = f_paths['models']
    rewards_folder = f_paths['rewards']
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    # variable for storing final value of the portfolio (done at end of episode)
    portfolio_value = []

    batch_size = 32  # sampleing from replay memory
    initial_investment = 100
    sim_env = SimulatedCryptoExchange(start, end, granularity=1, feature_dict=features)

    state_size = sim_env.state_dim
    action_size = len(sim_env.action_space)

    agents = [
            # ClassificationAgent(state_size, action_size),
            # MeanReversionAgent(state_size, action_size)
            # EMAReversion(state_size, action_size)
            # RegressionAgent(state_size, action_size)
            # DQNAgent(state_size, action_size)
            # SimpleAgent(state_size, action_size)
                ]

    time_remaining = timedelta(hours=0)

    #Print out some info about the assets
    for market in sim_env.markets:
        token = market[4:7]
        market_roi = ROI(sim_env.df[token + 'Close'].iloc[0], sim_env.df[token + 'Close'].iloc[-1])
        print(f'The {market} pair changed by {market_roi} % over the designated period.')

    # play the game num_episodes times
    for e in range(num_episodes):
        printProgressBar(e, num_episodes, prefix='Progress:', suffix='Complete')
        t0 = datetime.now()

        #If you're playing the last episode, create a log
        if e == num_episodes - 1:
            sim_env.should_log = True
        else: #In case for some reason we record a log and then switch back
            sim_env.should_log = False

        val = play_one_episode(agent, sim_env, mode)

        roi = ROI(initial_investment, val)  # Transform to ROI
        dt = datetime.now() - t0

        # time_remaining -= dt
        # time_remaining = time_remaining + \
        #     (dt * (num_episodes - (e + 1)) - time_remaining) / (e + 1)

        print(f"episode: {e + 1}/{num_episodes}, ", end = ' ')
        print(f"end value: {val:.2f}, episode roi: {roi:.2f}%, ")#, end = ' ')
        # print(f"time remaining: {time_remaining + timedelta(seconds = 5)}")
        portfolio_value.append(val)  # append episode end portfolio value

    sim_env.plot_market_data()
    sim_env.log.plot(sim_env.df)
    # save the weights when we are done
    if mode in ['train']:
        # save the DQN
        agent.save(f'{models_folder}/linear.npz')
        print('DQN saved.')

        # plot losses
        plt.plot(agent.model.losses) #this plots the index on the x axis and he loss on the y

    # save portfolio value for each episode
    print('Saving rewards...')
    np.save(f'{rewards_folder}/{mode}.npy', portfolio_value)
    print('Rewards saved.')

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        help='either "train", "gridsearch", or "test"')
    args = parser.parse_args()
    mode = args.mode
    assert mode in ["train", "test", 'gridsearch']

    # Leave this intact when running consecutive backtests and switching between modes
    features = {  # 'sign': ['Close', 'Volume'],
        # 'EMA': [50, 80, 130],
        'OBV': [],
        'RSI': [],
        # 'high': [],
        # 'low': [],
        'BollingerBands': [1, 2, 3, 4, 5, 6],
        'BBInd': [],
        # 'time of day': [],
        # 'knn':[]
        # 'stack': [0]
        }

    if mode == 'test':
        test(features)
    
    elif mode == 'gridsearch':
        grid_search(features)

    elif mode == 'train':
        train(features)