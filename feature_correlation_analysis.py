from environments import *
from datetime import datetime, timedelta
import pandas as pd

end = datetime.now() - timedelta(days = 5)
start = end - timedelta(hours = 40)

last_time_scraped = datetime.now() - timedelta(days = 3)

sim_env = SimulatedCryptoExchange(start, end)
# sim_env.plot_
df = sim_env.df.copy()
df['Future Percent Change'] = 100*(df['BTCClose'].shift(1) - df['BTCClose'])/df['BTCClose']
df['Last Percent Change'] = df['Future Percent Change'].shift(-1)
print(df.tail())
fig, ax = plt.subplots(1, 1)  # Create the figure
feat1_name = 'SMA_80'
feat2_name = 'BTCRSI'
df[df['Future Percent Change'] > .05].plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'b')
df[df['Future Percent Change'] < .05].plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'r')
# for param in [30]:
#
#     assert not sim_env.df.empty
#     for market in sim_env.markets:
#         token = market[4:7]
#         sim_env.df['Combo'] = sim_env.df[token + 'MACD']*sim_env.df[token + 'RSI']
#
#         features = [token + 'MACD', token + 'RSI', token + 'OBV', 'Combo']
#
#         for feature in features:
#             fig, ax = plt.subplots(1, 1)  # Create the figure
#
#             fig.suptitle(feature + ' Feature and Future Price Correlation For' + token, fontsize=14, fontweight='bold')
#             sim_env.df.plot(x=feature, y='Future Price', ax=ax, kind = 'scatter')


plt.show()
