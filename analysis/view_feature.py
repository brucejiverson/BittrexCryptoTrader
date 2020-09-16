from environments.environments import SimulatedCryptoExchange
from tools.tools import percent_change_column, ROI
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

features_to_view = [
                    # 'bb_bbm1', 'bb_bbh1', 'bb_bbl1', 'bb_bbm5', 'bb_bbh5', 'bb_bbl5', # 'bb_bbm10', 'bb_bbh10', 'bb_bbl10', 
                    # 'bb_bbli', 'bb_bbhi', 
                    # 'BBInd4',# 'BBInd5', 'BBInd10',
                    
                    # 'BBWidth4', #'BBWidth5', 'BBWidth10',
                    'Likelihood of buy given x/Likelihood of sell given x',
                    # 'discrete_derivative': ['BBWidth3'],
                    # 'BTCOBV', 
                    # 'BTCRSI'
                    ] # This is the feature to visualize
                    
start = datetime(2020, 8, 25)
end = datetime(2020, 9, 16)

last_time_scraped = datetime.now() - timedelta(days = .25)

features = {    # 'sign': ['Close', 'Volume'],
    # 'EMA': [50, 80, 130],
    'BollingerBands': [4, 5, 10],
    'BBInd': [],
    'BBWidth': [],
    # 'discrete_derivative': ['BBWidth4'],
    # 'OBV': [],
    # 'RSI': [],
    # 'time of day': [],
    'rolling probability': ['BBInd4', 'BBWidth4'],
    # 'probability': ['BBInd3', 'BBWidth3']
    # 'knn':[],
    'divide': ['Likelihood of buy given x', 'Likelihood of sell given x']
    # 'stack': [1]
    }
sim_env = SimulatedCryptoExchange(start, end, granularity=5, feature_dict=features, train_amount=14)

df = sim_env.df.copy()
df, new_name = percent_change_column('BTCClose', df, -1)
df = percent_change_column('BTCClose', df, 1)
df = percent_change_column('BTCVolume', df, 1)
# df['BTCOBV'] 

# Look for small scale (0 - 1) features
signal_features = []
known_small_scale = ['BTCOBV', 'BTCRSI', 'BBInd1', 'BBInd5', 'BBInd10', 'BBWidth1', 'BBWidth3', 'BBWidth5', 'BBWidth10',
                    'Likelihood of buy given x/Likelihood of sell given x']
for f in features_to_view:
    series = df[f]
    high, low = max(series), min(series)

    if high < 10 and low > -10 or f in known_small_scale:
        signal_features.append(f)

if not signal_features:
    fig, ax = plt.subplots(1, 1)  # Create the figure
else:
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)  # Create the figure
        
for market in sim_env.markets:
    token = market[4:7]

    market_perf = ROI(df[token + 'Close'].iloc[0], df[token + 'Close'].iloc[-1])
    # fig.suptitle(f'Market performance: {market_perf}%', fontsize=14, fontweight='bold')
    sim_env.df.plot( y=token +'Close', ax=ax)

for f in features_to_view:
    if f in signal_features:
        df.plot(y=f, ax=ax2)
    else: 
        df.plot(y=f, ax=ax)

fig.suptitle(f'Feature overlay: {features_to_view}', fontsize=14, fontweight='bold')

fig.autofmt_xdate()
plt.show()

