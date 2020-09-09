
from environments.environments import BittrexExchange
from agents.agents import *
import pandas as pd
from datetime import datetime, timedelta
import time
import pickle
import json

def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + timedelta(0,rounding-seconds,-dt.microsecond)
 
 # Leave this intact when running consecutive backtests and switching between modes
features = {  # 'sign': ['Close', 'Volume'],
        # 'EMA': [50, 80, 130],
        'OBV': [],
        'RSI': [],
        # 'high': [],
        # 'low': [],
        'BollingerBands': [3],
        'BBInd': [],
        'BBWidth': [],
        'discrete_derivative': ['BBWidth3'],
        # 'time of day': [],
        # 'stack': [2],
        # 'rolling probability': ['BBInd3', 'BBWidth3']
        'probability': ['BBInd3', 'BBWidth3']
        # 'knn':[],
        }

env = BittrexExchange(granularity=5, feature_dict=features, window_size=25, verbose=3)
state = env.reset()

feature_list = env.df.columns
feature_map = {}
for i, item in enumerate(feature_list):
    feature_map[item] = i
action_size = len(env.action_space)

# Bite Back
hyperparams = {'ind base': 15,
                'high thresh': .2,  # these are fraction base on BBInd (custom indicator)
                'low thresh': .5,
                'order': 3,                      # these are percentages for trailing orders
                'loss': 0,
                'stop loss': 6}
hyperparams = {'buy thresh': .55,
                'sell thresh': .5,
                'stop loss': .98}

# agent = biteBack(feature_map, action_size, hyperparams)
agent = probabilityAgent(feature_map, action_size, hyperparams)

# Load agent models
if agent.name == 'dqn':
    # make sure epsilon is not 1!
    # no need to run multiple episodes if epsilon = 0, it's deterministic
    agent.epsilon_min = 0.0001
    agent.epsilon = agent.epsilon_min

    # load trained weights
    agent.load(f'{models_folder}/linear.npz')


print('\n Oohh wee, here I go trading again! \n')

start_time = datetime.now()
loop_frequency = 60*env.granularity/2 #seconds

while True: # datetime.now() < start_time + timedelta(hours = .5):
    loop_start = datetime.now()
    bittrex_time = roundTime(datetime.now() + timedelta(hours=7))

    # print(f'It is now {bittrex_time} on the Bittrex Servers.')
    state = env.update() #This fetches data and preapres it, and also gets
    
    action = agent.act(state)
    env.act(action)
    env.log.save()

    # use async instead?
    # or just fix to run right after the candle gets made (15 seconds?)
    sleep_time = (timedelta(seconds = loop_frequency) - (datetime.now() - loop_start)).seconds #in seconds
    if sleep_time > 1 and sleep_time < loop_frequency - 5:
        print(f'Sleeping for {sleep_time} seconds.')
        time.sleep(sleep_time)