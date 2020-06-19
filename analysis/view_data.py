# from main import *
from environments import *
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

#cryptodatadownload has gaps
#Place to download: https://www.kaggle.com/jessevent/all-crypto-currencies iSinkInWater, brucejamesiverson@gmail.com, I**********


# symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD'] #Example: 'BTCUSD'
# markets = [sym[3:6] + '-' + sym[0:3] for sym in symbols]

start = datetime(2019, 1, 1)
end = datetime.now()


sim_env = SimulatedCryptoExchange(start, end)
# sim_env.save_candle_data()
sim_env.plot_market_data()
# sim_env.plot_stationary_data()
plt.show()
