import os
import pandas as pd 
import numpy as np

project_path = 'C:/Python Programs/crypto_trader'
f_paths = {'downloaded csv': project_path + '/data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv',
           'cum data pickle': project_path + '/data/cum_data_gran',
           'secret': "/Users/biver/Documents/crypto_data/secrets.json",
           'rewards': '../agent/agent-rewards',
           'models': '../agent/agent-models',
           'feature_models': project_path + '/bittrex_trader/features/models',
           'order log': project_path + '/bittrex_trader/logs/order_log.pkl',
           'test trade log':  project_path + '/bittrex_trader/logs/trade_testingBTCUSD.pkl',
           'live log': project_path + '/bittrex_trader/logs/live_account_log.pkl',
           'paper log': project_path + '/bittrex_trader/logs/paper_account_log.pkl',
           'score log': project_path + '/bittrex_trader/logs/gridsearch_score_log.pkl'}

project_path = '/home/bruce/AlgoTrader'
linux_paths = {'downloaded csv': project_path + '/data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv',
               'cum data pickle': project_path + '/data/cum_data_gran_',
               'secret': "/home/bruce/Documents/Crypto/secrets.json",
               'rewards': '../agents/agent-rewards',
               'models': '../agents/agent-models',
               'order log': project_path + '/BittrexTrader/agent_logging/order_log.csv',
               'test trade log': project_path + '/BittrexTrader/agent_logging/trade_testingBTCUSD.csv',
               'logging': project_path + '/BittrexTrader/agent_logging/live_account_log.csv'}

# f_paths = linux_paths

def ROI(initial, final):
    # Returns the percentage increase/decrease
    return round(final / initial - 1, 4)*100


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    # if iteration == 0:
        # print()
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'I made a directory at {directory}')


def percent_change_column(col_name, input_df, shift_val=1):
    """If shift_val > 0, converts the column to the percentage change.
    if < 0, adds a new column with the future change"""
    df = input_df.copy()
    if shift_val > 0:
        df[col_name] = df[col_name]/df[col_name].shift(shift_val, fill_value=0) - 1
        df[col_name] = 100*df[col_name].fillna(0)
        df.replace([np.inf, -np.inf], np.nan)
        df.dropna(inplace=True)
        return df
    elif shift_val < 0:
        name = '% Change ' + str(-shift_val) + ' steps in future'
        df[name] = (df[col_name].shift(shift_val, fill_value=None) - df[col_name])/df[col_name]
        df[name] = 100*df[name]
        df.replace([np.inf, -np.inf], np.nan)
        df.dropna(inplace=True)
        return df, name
    else:
        raise(ValueError)
    