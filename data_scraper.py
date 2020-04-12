from main import *
from datetime import datetime, timedelta
import pandas as pd


symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD'] #Example: 'BTCUSD'
markets = [sym[3:6] + '-' + sym[0:3] for sym in symbols]


#get my keys
with open(paths['secret']) as secrets_file:
    keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

while True:
    start = datetime.now() - timedelta(days = 10)
    end = datetime.now()
    df = fetch_historical_data(paths, markets, start, end, my_bittrex)  #gets all data
    print(df.head())
    save_historical_data(paths, df)
    print('Waiting...')
    time.sleep(60*60*24*5)
