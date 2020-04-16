from main import *
from environments import *
from datetime import datetime, timedelta
import pandas as pd

#cryptodatadownload has gaps
#Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********


# symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD'] #Example: 'BTCUSD'
# markets = [sym[3:6] + '-' + sym[0:3] for sym in symbols]

start = datetime(2016, 1, 1)
end = datetime.now()


sim_env = SimulatedCryptoExchange(paths, start, end)
sim_env.plot_market_data()
sim_env.plot_stationary_data()
