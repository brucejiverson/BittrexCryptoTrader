from bittrex.bittrex import *
import pandas as pd
from datetime import datetime, timedelta
import json


def process_trade_history(dict):
    # Example input: {'success': True, 'message': '', 'result': [{'OrderUuid': '3d87588d-70d6-4b40-a723-11248aaaff8b', 'Exchange': 'USD-BTC', 'TimeStamp': '2019-11-19T07:42:48.85', 'OrderType': 'LIMIT_SELL', 'Limit': 1.3, 'Quantity': 0.00123173, 'QuantityRemaining': 0.0, 'Commission': 0.02498345, 'Price': 9.99338392, 'PricePerUnit': 8113.29099722, 'IsConditional': False, 'Condition': '', 'ConditionTarget': 0.0, 'ImmediateOrCancel': False, 'Closed': '2019-11-19T07:42:48.85'}]}

    trade_df = pd.DataFrame(dict['result'])
    trade_df.drop(columns=['IsConditional', 'Condition', 'ConditionTarget',
                           'ImmediateOrCancel', 'Closed'], inplace=True)

    trade_df.reset_index(inplace=True, drop=True)
    # dates into datetimess
    trade_df.TimeStamp = pd.to_datetime(trade_df.TimeStamp, format="%Y-%m-%dT%H:%M:%S")
    # trade_df.Closed = pd.to_datetime(trade_df.Closed, format="%Y-%m-%dT%H:%M:%S")
    return trade_df


def process_order_data(dict):
    # Example input: {'success': True, 'message': '', 'result': {'AccountId': None, 'OrderUuid': '3d87588d-70d6-4b40-a723-11248aaaff8b', 'Exchange': 'USD-BTC', 'Type': 'LIMIT_SELL', 'Quantity': 0.00123173, 'QuantityRemaining': 0.0, 'Limit': 1.3, 'Reserved': None, 'ReserveRemaining': None, 'CommissionReserved': None, 'CommissionReserveRemaining': None, 'CommissionPaid': 0.02498345, 'Price': 9.99338392, 'PricePerUnit': 8113.29099722, 'Opened': '2019-11-19T07:42:48.85', 'Closed': '2019-11-19T07:42:48.85', 'IsOpen': False, 'Sentinel': None, 'CancelInitiated': False, 'ImmediateOrCancel': False, 'IsConditional': False, 'Condition': 'NONE', 'ConditionTarget': 0.0}}

    # in order to construct a df, the values of the dict cannot be scalars, must be lists, so convert to lists
    results = {}
    for key in dict['result']:
        results[key] = [dict['result'][key]]
    order_df = pd.DataFrame(results)

    order_df.drop(columns=['AccountId', 'Reserved', 'ReserveRemaining', 'CommissionReserved', 'CommissionReserveRemaining',
                           'Sentinel', 'IsConditional', 'Condition', 'ConditionTarget', 'ImmediateOrCancel', 'CancelInitiated'], inplace=True)

    order_df.reset_index(inplace=True, drop=True)
    # dates into datetimess
    order_df.Closed = pd.to_datetime(order_df.Closed, format="%Y-%m-%dT%H:%M:%S")
    order_df.Opened = pd.to_datetime(order_df.Opened, format="%Y-%m-%dT%H:%M:%S")

    return order_df


def save_trade_data(trade_df, path_dict):
    save_path = path_dict['test trade log']

    try:
        def dateparse(x):
            try:
                return pd.datetime.strptime(x, "%Y-%m-%d %I-%p-%M")
            except ValueError:  #handles cases for incomplete trades where 'Closed' is NaT
                return datetime(year = 2000, month = 1, day = 1)

        old_df = pd.read_csv(save_path, parse_dates=['Opened', 'Closed'], date_parser=dateparse)

        trade_df = trade_df.append(old_df)
        trade_df.sort_values(by='Opened', inplace=True)
        trade_df.reset_index(inplace=True, drop=True)

        trade_df['Closed'] = trade_df['Closed'].dt.strftime("%Y-%m-%d %I-%p-%M")
        trade_df['Opened'] = trade_df['Opened'].dt.strftime("%Y-%m-%d %I-%p-%M")

        trade_df.to_csv(save_path, index=False)
        print('Data written to test trade log.')

    except KeyError:
        print('Order log is empty.')


def cancel_all_orders(bittrex_obj, markit):
    open_orders = bittrex_obj.get_open_orders(markit)
    if open_orders['success']:
        if not open_orders['result']:
            print('No open orders.')
        else:
            for order in open_orders['result']:
                uuid = order['OrderUuid']
                cancel_result = my_bittrex.cancel(uuid)['success']
                if cancel_result == True:  # need to see if im checking if cancel_result exits or if im checking its value
                    print('Cancel status: ', cancel_result, ' for order: ', uuid)

    else:
        print('Failed to get order history.')
        print(result)


def order(amount, side):
  # Note that bittrex exchange is based in GMT 8 hours ahead of CA
  trade_incomplete = True
  retries = 5
  while trade_incomplete:
      if retries <= 0:
          print('Failed to order 5 times.')
          break

      start_time = datetime.now()

      # Enter a trade into the market.
      # Example result  {'success': True, 'message': '', 'result': {'uuid': '2641035d-4fe5-4099-9e7a-cd52067cde8a'}}
      ticker = my_bittrex.get_ticker(market)
      price = ticker['result']['Last']
      amount = amount / price  # Convert to the amount of BTC
      if is_USD:  # buy
          trade_result = my_bittrex.buy_limit(market, amount, price)
          side = 'buying'
      else:       # Sell
          trade_result = my_bittrex.sell_limit(market, amount, price)
          side = 'selling'

      # Check that an order was entered
      if not trade_result['success']:
          print('Trade attempt failed')
          print(trade_result['message'])
          continue

      print(f'Order for {side} {amount:.8f} {symbols[0:3]} at a price of {price:.2f} has been submitted to the market.')
      order_uuid = trade_result['result']['uuid']

      # Loop to see if the order has been filled
      is_open = True
      cancel_result = False
      while is_open:
          order_data = my_bittrex.get_order(order_uuid)
          is_open = order_data['result']['IsOpen']
          # print('Order open status: ', is_open)

          if not is_open:
              print(f'Order number {n+1} out of {num_trades} has been filled. Id: {order_uuid}.')
              break

          time.sleep(0.5)
          time_elapsed = datetime.now() - start_time

          if time_elapsed > timedelta(seconds=10):
              print('Order has not gone through in 10 seconds. Cancelling...')
              # Cancel the order
              cancel_result = my_bittrex.cancel(order_uuid)['success']
              if cancel_result == True:  # need to see if im checking if cancel_result exits or if im checking its value
                  print(f'Cancel status: {cancel_result} for order: {order_uuid}.')
                  retries -= 1
                  break

      if cancel_result == True:
          # if the order was cancelled, try again
          print(f'Attempt number {3 - retries} was not filled. Attempting to order again. ')
          continue

      dt = datetime.now() - start_time  # note that this include the time to run a small amount of code

      order_data['result']['Order Duration'] = dt
      trade = process_order_data(order_data)
      return trade


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

balance_response = my_bittrex.get_balance('BTC')

if balance_response['success']:
    BTC_balance = balance_response['result']['Balance']

USD = my_bittrex.get_balance('USD')['result']['Balance']

num_trades = 10

# Price in USD, price per unit is $/BTC

# Figure out where the money is in the account and how much there is
is_USD = True

log = pd.DataFrame()

for n in list(range(num_trades)):

    # Note that bittrex exchange is based in GMT 8 hours ahead of CA
    trade_incomplete = True
    # retries = 5
    while trade_incomplete:
        # if retries <= 0:
        #     print('Failed to order 5 times.')
        #     break

        start_time = datetime.now()
        amount = 5  # in USD

        # Enter a trade into the market.
        # Example result  {'success': True, 'message': '', 'result': {'uuid': '2641035d-4fe5-4099-9e7a-cd52067cde8a'}}
        ticker = my_bittrex.get_ticker(market)
        price = ticker['result']['Last']
        amount = amount / price  # Convert to the amount of BTC

        if is_USD:  # buy
            trade_result = my_bittrex.buy_limit(market, amount, round(price*1.0001, 3))
            side = 'buying'
        else:       # Sell
            trade_result = my_bittrex.sell_limit(market, amount, round(price*0.9999, 3) )
            side = 'selling'

        # Check that an order was entered
        if not trade_result['success']:
            print('Trade attempt failed')
            print(trade_result['message'])
            continue

        print(f'Order for {side} {amount:.8f} {symbols[0:3]} at a price of {price:.3f} has been submitted to the market.')
        order_uuid = trade_result['result']['uuid']

        # Loop to see if the order has been filled
        is_open = True
        cancel_result = False
        while is_open:
            order_data = my_bittrex.get_order(order_uuid)
            try:
                is_open = order_data['result']['IsOpen']
            except TypeError:
                print(is_open)
            # print('Order open status: ', is_open)

            if not is_open:
                print(f'Order number {n+1} out of {num_trades} has been filled. Id: {order_uuid}.')
                is_USD = not is_USD
                trade_incomplete = False
                break

            time.sleep(0.5)
            time_elapsed = datetime.now() - start_time

            time_limit = 30
            if time_elapsed > timedelta(seconds=time_limit):
                print(f'Order has not gone through in {time_limit} seconds. Cancelling...')
                # Cancel the order
                cancel_result = my_bittrex.cancel(order_uuid)['success']
                if cancel_result == True:  # need to see if im checking if cancel_result exits or if im checking its value
                    print(f'Cancel status: {cancel_result} for order: {order_uuid}.')
                    # retries -= 1
                    break

        if cancel_result == True:
            # if the order was cancelled, try again
            # print(f'Attempt number {3 - retries} was not filled. Attempting to order again. ')
            print(f'Attempt was not filled. Attempting to order again.')
            # continue

        dt = datetime.now() - start_time  # note that this include the time to run a small amount of code

        try: #Some weird error here that I have not been able to recreate. Added print statements for debugging if it occurs again
            order_data['result']['Order Duration'] = dt.total_seconds()
        except TypeError:
            print(dt)
            print(dt.total_seconds())
        trade = process_order_data(order_data)
        log = log.append(trade, ignore_index=True)
        # log.reset_index(inplace=True)


save_trade_data(log, paths)

# def cancel(self, uuid):
#     """
#     Used to cancel a buy or sell order
#     Endpoint:
#     1.1 /market/cancel
#     2.0 /key/market/tradecancel
#     :param uuid: uuid of buy or sell order
#     :type uuid: str
#     :return:
#     :rtype : dict
#     """
#     return self._api_query(path_dict={
#         API_V1_1: '/market/cancel',
#         API_V2_0: '/key/market/tradecancel'
#     }, options={'uuid': uuid, 'orderid': uuid}, protection=PROTECTION_PRV)
#
# def get_ticker(self, market):
#         """
#         Used to get the current tick values for a market.
#         Endpoints:
#         1.1 /public/getticker
#         2.0 NO EQUIVALENT -- but get_latest_candle gives comparable data
#         :param market: String literal for the market (ex: BTC-LTC)
#         :type market: str
#         :return: Current values for given market in JSON
#         :rtype : dict
#         """
#         return self._api_query(path_dict={
#             API_V1_1: '/public/getticker',
#         }, options={'market': market}, protection=PROTECTION_PUB)

# def get_open_orders(self, market=None):
#     """
#     Get all orders that you currently have opened.
#     A specific market can be requested.
#     Endpoint:
#     1.1 /market/getopenorders
#     2.0 /key/market/getopenorders
#     :param market: String literal for the market (ie. BTC-LTC)
#     :type market: str
#     :return: Open orders info in JSON
#     :rtype : dict
#     """
#     return self._api_query(path_dict={
#         API_V1_1: '/market/getopenorders',
#         API_V2_0: '/key/market/getopenorders'
#     }, options={'market': market, 'marketname': market} if market else None, protection=PROTECTION_PRV)

# def sell_limit(self, market, quantity, rate):
#         """
#         Used to place a sell order in a specific market. Use selllimit to place
#         limit orders Make sure you have the proper permissions set on your
#         API keys for this call to work
#         Endpoint:
#         1.1 /market/selllimit
#         2.0 NO Direct equivalent.  Use trade_sell for LIMIT and MARKET sells
#         :param market: String literal for the market (ex: BTC-LTC)
#         :type market: str
#         :param quantity: The amount to sell
#         :type quantity: float
#         :param rate: The rate at which to place the order.
#             This is not needed for market orders
#         :type rate: float
#         :return:
#         :rtype : dict
#         """
#         return self._api_query(path_dict={
#             API_V1_1: '/market/selllimit',
#         }, options={'market': market,
#                     'quantity': quantity,
#                     'rate': rate}, protection=PROTECTION_PRV)
#
# def get_order(self, uuid):
#         """
#         Used to get details of buy or sell order
#         Endpoint:
#         1.1 /account/getorder
#         2.0 /key/orders/getorder
#         :param uuid: uuid of buy or sell order
#         :type uuid: str
#         :return:
#         :rtype : dict
#         """
#         return self._api_query(path_dict={
#             API_V1_1: '/account/getorder',
#             API_V2_0: '/key/orders/getorder'
#         }, options={'uuid': uuid, 'orderid': uuid}, protection=PROTECTION_PRV)
