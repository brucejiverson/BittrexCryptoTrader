from main import *
from datetime import datetime, timedelta
import pandas as pd

start = datetime.now() - timedelta(days = 10)
end = datetime.now()

last_time_scraped = datetime.now() - timedelta(days = 20)

symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD'] #Example: 'BTCUSD'
markets = [sym[3:6] + '-' + sym[0:3] for sym in symbols]


#get my keys
with open(paths['secret']) as secrets_file:
    keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

while True:
    df = fetch_historical_data(paths, markets, start, end, my_bittrex)  #gets all data
    save_historical_data(path_dict, df)
    time.sleep(60*60*24*5)
