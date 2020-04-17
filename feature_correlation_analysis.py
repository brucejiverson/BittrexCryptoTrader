from environments import *
from datetime import datetime, timedelta
import pandas as pd

start = datetime.now() - timedelta(days = 10)
end = datetime.now()

last_time_scraped = datetime.now() - timedelta(days = 20)

sim_env = SimulatedCryptoExchange(paths, start, end)

for param in [30]:

    for market in sim_env.markets:
        token = market[4:7]
        sim_env.df['Future Price'] = sim_env.df[token + 'Close'].shift(1) - sim_env.df[token + 'Close']

        assert not sim_env.df.empty

        features = [token + 'MACD', token + 'RSI']

        for feature in features:
            fig, ax = plt.subplots(1, 1)  # Create the figure

            fig.suptitle(feature + ' Feature and Future Price Correlation For' + token, fontsize=14, fontweight='bold')
            sim_env.df.plot(x=feature, y='Future Price', ax=ax, kind = 'scatter')

plt.show()
