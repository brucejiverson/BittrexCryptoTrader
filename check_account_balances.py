from bittrex.bittrex import *
from bittrex_tools import *
import pandas as pd
from datetime import datetime, timedelta
import json


symbols = 'BTCUSD'  # Example: 'BTCUSD'
market = symbols[3:6] + '-' + symbols[0:3]


paths = {'downloaded history': 'C:/Python Programs/crypto_trader/historical data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv',
         'updated history': 'C:/Python Programs/crypto_trader/historical data/updated_history_' + symbols + '.csv',
         'secret': "/Users/biver/Documents/crypto_data/secrets.json",
         'rewards': 'agent_rewards',
         'models': 'agent_models',
         'test trade log':  'C:/Python Programs/crypto_trader/historical data/trade_testing' + symbols + '.csv'}

# get my keys
with open(paths['secret']) as secrets_file:
    keys = json.load(secrets_file)  # loads the keys as a dictionary with 'key' and 'secret'
    secrets_file.close()

my_bittrex = Bittrex(keys["key"], keys["secret"], api_version=API_V1_1)

cancel_all_orders(my_bittrex, market)

#Loop forever to get the balances for USD and BTC
while True:
    check1 = False
    balance_response = my_bittrex.get_balance('BTC')
    if balance_response['success']:
        BTC_balance = balance_response['result']['Balance']
        check1 = True

    balance_response = my_bittrex.get_balance('USD')
    if balance_response['success']:
        USD_balance = balance_response['result']['Balance']
        if check1:
            break

print(f"BTC: {BTC_balance}")
print(f"USD: {USD_balance}")

try:
    if BTC_balance > 0:
        need_to_sell = True
except TypeError: #BTC_balance is none
        need_to_sell = False

while need_to_sell:

    ticker = my_bittrex.get_ticker(market)
    price = ticker['result']['Last']
    amount = BTC_balance

    #Sell any bitcoin
    trade_result = my_bittrex.sell_limit(market, amount, round(price*0.9999, 3) )
    order_uuid = trade_result['result']['uuid']
    order_data = my_bittrex.get_order(order_uuid)

    side = 'selling'

    # Check that an order was entered
    if not trade_result['success']:
        print('Trade attempt failed')
        print(trade_result['message'])
        continue

    print(f'Order for {side} {amount:.8f} {symbols[0:3]} at a price of {price:.3f} has been submitted to the market.')

    status = trade_is_executed(order_uuid, order_data, my_bittrex)

    if status == True:
        # is_USD = not is_USD
        print(f'Order has been filled. Id: {uuid}.')
        break
    print(f'Attempt was not filled. Attempting to order again.')
