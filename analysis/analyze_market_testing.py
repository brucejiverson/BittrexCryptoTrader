import pandas as pd
import statistics
from datetime import datetime
from tools.tools import f_paths


symbols = 'BTCUSD'  # Example: 'BTCUSD'
market = symbols[3:6] + '-' + symbols[0:3]

path = f_paths['test trade log']

df = pd.read_pickle(path)

# Convert datetimes to string?? why is this here?
df['Closed'] = df['Closed'].dt.strftime("%Y-%m-%d %I-%p-%M")
df['Opened'] = df['Opened'].dt.strftime("%Y-%m-%d %I-%p-%M")

#Count how many attempted orders there were
n_attempts = df.shape[0]

#Count how many attempted trades Failed
filled_orders = df[df['IsOpen'] == False]
n_unfilled = df[df['IsOpen'] == True].shape[0]

#Count how many trades succeeded
n_filled = filled_orders.shape[0]

#Out of the trades that succeeded, what was the mean and the std dev of the trade time
mean_fill_time = filled_orders['Order Duration'].mean()
std_fill_time = filled_orders['Order Duration'].std()

print(f'Out of {n_attempts} attempts, {n_filled} were filled ({100*n_filled/n_attempts:.1f} %).')
print(f'The mean time a successful order was filled was {mean_fill_time:.2f}, +- {2*std_fill_time:.2f} (95% confidence).')
