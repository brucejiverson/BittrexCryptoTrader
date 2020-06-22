f_paths = {'downloaded csv': 'C:/Python Programs/crypto_trader/data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv',
           'cum data pickle': 'C:/Python Programs/crypto_trader/data/cum_data.pkl',
           'secret': "/Users/biver/Documents/crypto_data/secrets.json",
           'rewards': 'agent-rewards',
           'models': 'agent-models',
           'order log': 'C:/Python Programs/crypto_trader/agent_logging/order_log.csv',
           'test trade log':  'C:/Python Programs/crypto_trader/agent_logging/trade_testingBTCUSD.csv',
           'logging': 'C:/Python Programs/crypto_trader/agent_logging/live_account_log.csv'}

linux_paths = {'downloaded csv': '/home/bruce/AlgoTrader/data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv',
               'cum data pickle': '/home/bruce/AlgoTrader/data/cum_data_gran_',
               'secret': "/home/bruce/Documents/Crypto/secrets.json",
               'rewards': '../agents/agent-rewards',
               'models': '../agents/agent-models',
               'order log': '/home/bruce/AlgoTrader/BittrexTrader/agent_logging/order_log.csv',
               'test trade log': '/home/bruce/AlgoTrader/BittrexTrader/agent_logging/trade_testingBTCUSD.csv',
               'logging': '/home/bruce/AlgoTrader/BittrexTrader/agent_logging/live_account_log.csv'}

f_paths = linux_paths


# Print iterations progress
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
