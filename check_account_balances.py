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


"""Preserving the below for use in developing BittrexExchange environment"""
# try:
#     if BTC_balance > 0:
#         need_to_sell = True
# except TypeError: #BTC_balance is none
#         need_to_sell = False

# while need_to_sell:
#
#     #Sell any bitcoin
#     trade_result = my_bittrex.sell_limit(market, amount, round(price*0.9999, 3) )
#     order_uuid = trade_result['result']['uuid']
#     order_data = my_bittrex.get_order(order_uuid)
#
#     side = 'selling'
#
#     # Check that an order was entered
#     if not trade_result['success']:
#         print('Trade attempt failed')
#         print(trade_result['message'])
#         continue
#
#     print(f'Order for {side} {amount:.8f} {symbols[0:3]} at a price of {price:.3f} has been submitted to the market.')
#
#     status = trade_is_executed(order_uuid, order_data, my_bittrex)
#
#     if status == True:
#         # is_USD = not is_USD
#         print(f'Order has been filled. Id: {uuid}.')
#         break
#     print(f'Attempt was not filled. Attempting to order again.')
