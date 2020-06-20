from bittrex.bittrex import *
from main import *
from environments import BittrexExchange
import pandas as pd
from datetime import datetime, timedelta
import json


exchange = BittrexExchange(paths)

exchange.get_balances()

print(f"BTC: {exchange.assets_owned[0]}")
print(f"USD: {exchange.USD}")
