
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
        # 'OBV': [],
        # 'RSI': [],
        # 'high': [],
        # 'low': [],
        'BollingerBands': [4],
        'BBInd': [],
        'BBWidth': [],
        'discrete_derivative': ['BBWidth4'],
        # 'time of day': [],
        # 'stack': [2],
        # 'rolling probability': ['BBInd3', 'BBWidth3']
        'probability': ['BBInd4', 'BBWidth4']
        # 'knn':[],
        }

env = BittrexExchange(granularity=1, feature_dict=features, window_size=15, verbose=3)
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
hyperparams = {'buy thresh': .5,
                'sell thresh': .42,
                'stop loss': None}

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
loop_frequency = 60*env.granularity #seconds
# The next time to run at (rounded to nearest minute)
run_time = roundTime(datetime.now() + timedelta(hours=7))   # To the nearest minute

while True: # datetime.now() < start_time + timedelta(hours = .5):
    now = datetime.now() + timedelta(hours=7)

    # print(f'It is now {run_time} on the Bittrex Servers.')
    if now >= run_time:
        state = env.update() #This fetches data and preapres it, and also gets
        
        action = agent.act(state)
        env.act(action)
        env.log.save()
        # The next time to run at (rounded to nearest minute)
        run_time += timedelta(seconds= loop_frequency)   # To the nearest minute
    
    delta_time = run_time - (now)
    sleep_time =  delta_time.seconds            # in seconds
    # print(f'Run time:   {run_time}')
    # print(f'Now:        {now}')
    # print(f'Deltatime:  {delta_time}')
    # print(f'Sleep time: {sleep_time}')
    if delta_time > timedelta(seconds=0):
        print(f'Sleeping for {sleep_time} seconds.')
        time.sleep(sleep_time)