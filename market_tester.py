
from environments import *
from agents import *
import pandas as pd
from datetime import datetime, timedelta
import json

env = BittrexExchange(money_to_use = 5)
# print(env.action_list)
env._act(0)
# env._trade('USD-BTC', -5)
# env.view_order_data()
# env.get_latest_candle(env.markets[0])
# time.sleep(60)


print('Initializing agent...', end = ' ')
agent = DQNAgent(state_size, action_size)
# agent = SimpleAgent(state_size, action_size)
my_scaler = get_scaler(sim_env)

if agent.name == 'dqn':
    # then load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
        my_scaler = pickle.load(f)

    # make sure epsilon is not 1!
    # no need to run multiple episodes if epsilon = 0, it's deterministic
    agent.epsilon_min = 0.00#1
    agent.epsilon = agent.epsilon_min

    # load trained weights
    agent.load(f'{models_folder}/linear.npz')
print('Testing...')
start_time = datetime.now()
while datetime.now() < start_time + timedelta(hours =1):
    print(f'It is now {datetime.now() + timedelta(hours = 7)} on the Bittrex Servers.')
    state = env.update() #This fetches data and preapres it, and also gets
    print(state)
    if agent.name == 'dqn':next_state = scaler.transform([next_state])
    action = agent.act(state)
    print('Sleeping')
    time.sleep(60)


state = next_state







#BELOW IS REFERENCE FROM BITREX.BITTREX LIBRARY
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
# def get_market_history(self, market):
#     """
#     Used to retrieve the latest trades that have occurred for a
#     specific market.
#     Endpoint:
#     1.1 /market/getmarkethistory
#     2.0 NO Equivalent
#     Example ::
#         {'success': True,
#         'message': '',
#         'result': [ {'Id': 5625015,
#                      'TimeStamp': '2017-08-31T01:29:50.427',
#                      'Quantity': 7.31008193,
#                      'Price': 0.00177639,
#                      'Total': 0.01298555,
#                      'FillType': 'FILL',
#                      'OrderType': 'BUY'},
#                      ...
#                    ]
#         }
#     :param market: String literal for the market (ex: BTC-LTC)
#     :type market: str
#     :return: Market history in JSON
#     :rtype : dict
#     """
#     return self._api_query(path_dict={
#         API_V1_1: '/public/getmarkethistory',
#     }, options={'market': market, 'marketname': market}, protection=PROTECTION_PUB)
