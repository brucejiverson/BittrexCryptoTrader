from bittrex.bittrex import *
from main import *

import pandas as pd
import math
from datetime import datetime, timedelta
import itertools
import numpy as np
import json

class ExchangeEnvironment:
    """All other environment classes inherit from this class. This is done to ensure similar architecture
    between different classes (eg simulated vs real exchange), and to make it easy to change that
    architecture by having change be in a single place."""

    def __init__(self):
        self.markets = ['USD-BTC']#, 'USD-ETH', 'USD-LTC']    #Alphabetical
        self.n_asset = len(self.markets)

        self.n_indicators = 3 #This HAS to match the number of features that have been created in the add features thing

        portfolio_granularity = 1  # smallest fraction of portfolio for investment in single asset (.01 to 1)
        # The possible portions of the portfolio that could be allocated to a single asset
        possible_vals = [x / 100 for x in list(range(0, 101, int(portfolio_granularity * 100)))]
        # calculate all possible allocations of wealth across the available assets
        self.action_list = []
        permutations_list = list(map(list, itertools.product(possible_vals, repeat=self.n_asset)))
        #Only include values that are possible (can't have more than 100% of the portfolio)
        for i, item in enumerate(permutations_list):
            if sum(item) <= 1:
                self.action_list.append(item)

        #This list is for indexing each of the actions
        self.action_space = np.arange(len(self.action_list))

        # calculate size of state (amount of each asset held, value of each asset, volumes, USD/cash, indicators for each asset)
        self.state_dim = self.n_asset*3 + 1 + self.n_indicators*self.n_asset

        self.last_action = []

        # self.rewards_hist_len = 10
        # self.rewards_hist = np.ones(self.rewards_hist_len)


    def _add_features():
        pass

class SimulatedCryptoExchange(ExchangeEnvironment):
    """
    A multi-asset trading environment.
    For now this has been adopted for only one asset.
    Below shows how to add more.
    The state size and the aciton size throughout the rest of this
    program are linked to this class.
    State: vector of size 7 (n_asset + n_asset*n_indicators)
      - stationary price of asset 1 (using BTCClose price)
      - associated indicators for each asset
    """

    def __init__(self, data, initial_investment=100):
        ExchangeEnvironment.__init__(self)

        # data
        self.asset_data = data

        # n_step is number of samples, n_stock is number of assets. Assumes to datetimes are included
        self.n_step, self.n_asset = self.asset_data.shape
        # for now this works but will need to be updated when multiple assets are added
        self.n_asset -= self.n_indicators + 1

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.assets_owned = None
        self.asset_prices = None
        self.USD = None
        self.mean_spread = .0001 #Fraction of asset value typical for the spread

        self.reset()

    def reset(self):
        # Resets the environement to the initial state
        self.cur_step = 0  # point to the first datetime in the dataset
        # Own no assets to start with
        self.assets_owned = np.zeros(self.n_asset)
        self.last_action = self.action_list[0] #The action where nothing is owned
        """ the data, for asset_data can be thought of as nested arrays, where indexing the
        highest order array gives a snapshot of all data at a particular time, and information at the point
        in time can be captured by indexing that snapshot."""
        self.asset_prices = self.asset_data[self.cur_step][0:self.n_asset]

        self.USD = self.initial_investment

        # print(self.cur_state)
        # print(self.asset_prices)
        # print(self.n_asset)

        return self._get_state(), self._get_val() # Return the state vector (same as obervation for now)

    def step(self, action):
        # Performs an action in the enviroment, and returns the next state and reward

        if not action in self.action_space:
            #Included for debugging
            # print(action)
            print(self.action_space)
            assert action in self.action_space  # Check that a valid action was passed

        prev_val = self._get_val()

        # perform the trade
        self._trade(action)

        # update price, i.e. go to the next minute
        self.cur_step += 1

        """ the data, for asset_data can be thought of as nested arrays, where indexing the
        highest order array gives a snapshot of all data at a particular time, and information at the point
        in time can be captured by indexing that snapshot."""
        self.asset_prices = self.asset_data[self.cur_step][0:self.n_asset]


        # store the current value of the portfolio here
        cur_val = self._get_val()
        info = {'cur_val': cur_val}

        reward = (cur_val - prev_val) #this used to be more complicated

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # conform to the Gym API
        #      next state       reward  flag  info dict.
        return self._get_state(), self._get_val(), reward, done, info

    def _get_state(self):
        # Returns the state (for now state, and observation are the same.
        # Note that the state could be a transformation of the observation, or
        # multiple past observations stacked.)
        #state is (amount of each asset held, value of each asset, cash in hand, volumes, indicators)

        state = np.empty(self.state_dim)  #assets_owned, USD, volume, indicators
        state[0:self.n_asset] = self.last_action   #This is set in trade
        state[self.n_asset] = self.USD     #This is set in trade
        # Asset data is amount btc price, usd, volume, indicators

        #Instituted a try catch here to help with debugging and potentially as a solution to handling invalid/inf values in log
        try:

            if self.cur_step == 0:
                stationary_slice = np.zeros(len(self.asset_data[0]))

            else:   #Make data stationary
                slice = self.asset_data[self.cur_step]
                last_slice = self.asset_data[self.cur_step - 1]

                stationary_slice = np.empty(len(slice))

                #Comment out one of the two below, the difference is statinary data. NEEDS AN UPGRADE FOR MULTIPLE ASSETS
                # stationary_slice[0:2] = slice[0:2]
                stationary_slice[0:2] = np.log(slice[0:2]) - np.log(last_slice[0:2]) #this is price and volume

                stationary_slice[2:6] = slice[2:6] #these are indicators.
                # print(stationary_slice)
                # print(state)

        except ValueError:  #Print shit out to help with debugging then throw error
            # print(slice)
            # print(stationary_slice)
            print("Error in simulated market class, _get_state method.")
            print(state)
            raise(ValueError)

        state[self.n_asset+1:self.state_dim] =  stationary_slice   #Taken from data
        # print(state)

        return state

    def _get_val(self):
        return self.assets_owned.dot(self.asset_prices) + self.USD

    def _trade(self, action):
        # index the action we want to perform
        # action_vec = [(desired amount of stock 1), (desired amount of stock 2), ... (desired amount of stock n)]

        # get current value before performing the action
        prev_price = self.asset_prices[0]

        action_vec = self.action_list[action]

        cur_price = self.asset_prices[0]
        bid_price = cur_price*(1 - self.mean_spread/2)
        ask_price = cur_price*(1 + self.mean_spread/2)

        cur_val = self._get_val()


        if action_vec != self.last_action:  # if attmepting to change state
            #Calculate the changes needed for each asset
            # delta = [s_prime - s for s_prime, s in zip(action_vec, self.last_action) #not using this now, but how it should be done


            #currently set up for only bitcoin
            """for i, a in enumerate(action_vec): #for each asset
                fractional_change_needed = a - (1 - self.USD/cur_val) #desired fraction of portfolio to have in asset - fraction held

                if abs(fractional_change_needed) > .05: #Porfolio granulatiryt will change with asset price movement. This sets a threshhold for updating position
                    # print("Frac change: " + str(fractional_change_needed))

                    trade_amount = fractional_change_needed*cur_val #in USD
                    # print("Trade amount: " + str(trade_amount))
                    if trade_amount > 0:    #buy
                        self.assets_owned[0] += trade_amount/bid_price
                    else:   #sell
                        self.assets_owned[0] += trade_amount/ask_price

                    self.USD -= trade_amount"""

            #Below this is the old way
             # Sell everything
            for i, a in enumerate(action_vec):
                self.USD += self.assets_owned[i] * self.asset_prices[i]
                self.assets_owned[i] = 0

            # Buy back the right amounts
            for i, a in enumerate(action_vec):
                cash_to_use = a * self.USD
                self.assets_owned[i] = cash_to_use / self.asset_prices[i]
                self.USD -= cash_to_use


            # print("Initial val: " + str(cur_val) + ". Post trade val:" + str(self._get_val()))
            self.last_action = action_vec
            self.period_since_trade = 0

class BittrexExchange(ExchangeEnvironment):
    """This class provides an interface with the Bittrex exchange for any and all operations. It inherites from the 'ExchangeEnvironment
    class, which ensures similar architecture between different environments. Methods for this class include executing trades,
    logging account value over time, displaying account value over time, retrieving information on prices, balances, and orders from
    Bittrex, and uses a similar 'act' method to interface with agents."""

    def __init__(self, path_dict, money_to_use=10):
        ExchangeEnvironment.__init__(self)
        #get my keys
        with open("/Users/biver/Documents/crypto_data/secrets.json") as secrets_file:
            keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
            secrets_file.close()

        #Need both versions of the interface as they each provide certain useful functions
        self.bittrex_obj_1_1 = Bittrex(keys["key"], keys["secret"], api_version=API_V1_1)
        self.bittrex_obj_2 = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

        # instance attributes
        self.initial_investment = money_to_use
        self.assets_owned = None #this needs to change
        self.asset_prices = [0]*self.n_asset
        self.asset_volumes = None
        self.USD = None

        # self.state, self.cur_val = self.reset()

        self.get_account_health()


    def reset(self):
        # Resets the environement to the initial state

        self.cancel_all_orders()
        self.get_prices(self.markets[0])
        self.get_all_balances()

        # #Put all money into USD
        # if self.assets_owned[0] > 0:
        #     sucess = False
        #     while not success:
        #         success = self._trade(-self.assets_owned[0])

        self.assets_owned = np.zeros(self.n_asset)
        self.last_action = self.action_list[0] #The action where nothing is owned
        """ the data, for asset_data can be thought of as nested arrays, where indexing the
        highest order array gives a snapshot of all data at a particular time, and information at the point
        in time can be captured by indexing that snapshot."""

        return self._get_state(), self._return_val() # Return the state vector (same as obervation for now)


    def get_prices(self, currency_pair):
        """This method retrieves up to date price information from the exchange.
        To Do: make this get multiple currencies."""

        print('Fetching prices... ', end = ' ')
        count = 3
        while True:
            ticker = self.bittrex_obj_1_1.get_ticker(currency_pair)
            #Change this to make sure that the check went through
            # Check that an order was entered
            if not ticker['success']:
                print('_get_prices failed. Ticker message: ', end = ' ')
                print(ticker['message'])

                if count == 0:
                    print('get_prices has failed. exiting the method...')
                    break
                else:
                    count -= 1
                    time.sleep(1)
                    print('Retrying...')
            else: #success
                # print(ticker['result'])
                print('Sucess.')
                self.asset_prices[0] = ticker['result']['Last']
                break


    def get_all_balances(self):
        """This method retrieves the account balanes for each currency including USD from the exchange."""

        print('Fetching account balances...', end = ' ')
        self.assets_owned = np.zeros(self.n_asset)
        while True:
            check1 = False
            balance_response = self.bittrex_obj_1_1.get_balance('BTC')
            if balance_response['success']:
                print('BTC balance fetched.', end = ' ')
                amount = balance_response['result']['Balance']
                #Find a more legant way of checking if 'None'
                if amount is None:
                    self.assets_owned[0] = 0
                else: self.assets_owned[0] = amount

                # try:
                #     if self.assets_owned[0] > 0:
                #         pass
                # except TypeError: #BTC_balance is none
                check1 = True


            balance_response = self.bittrex_obj_1_1.get_balance('USD')
            if balance_response['success']:
                print('USD balance fetched.')
                self.USD = balance_response['result']['Balance']

                #Find a more legant way of checking if 'None'
                try:
                    if self.USD > 0:
                        pass
                except TypeError: #BTC_balance is none
                        self.USD = 0
                if check1:
                    break


    def get_account_health(self):
        """This method prints out a variety of information about the account. In the future this should print a dataframe
        for cleaner formatting (if mutliple currency trading strategies are implemented)"""

        self.get_all_balances()
        self.get_prices('USD-BTC')
        cur_val = self._return_val()

        print('')
        print('ACCOUNT SUMMARY: ')
        print('There is $' + str(self.USD) + ' USD in your account.')
        print('There is ' + str(round(self.assets_owned[0],2)) + ' BTC in your account.')
        print('Current value of the accout is: $' + str(cur_val) + '.')
        print('The current value of 1 BTC is $' + str(self.asset_prices[0]))


    def _calculate_indicators(self):
        """This method calculates all of the indicators defined in the initialization up to the current price point."""
        pass


    def _get_state(self):
        # Returns the state (for now state, and observation are the same.
        # Note that the state could be a transformation of the observation, or
        # multiple past observations stacked.)
        state = np.empty(self.state_dim)  #assets_owned, USD
        # self.cur_state[0:self.n_asset] = self.asset_prices
        state[0:self.n_asset] = self.asset_prices
        state[0:self.n_asset] = self.assets_owned   #This is set in trade
        state[self.n_asset] = self.USD     #This is set in trade
        # Asset data is amount btc price, usd, volume, indicators
        # state[self.n_asset+1:(self.n_asset*2 + 2 + self.n_indicators*self.n_asset)] =  self.asset_data[self.cur_step]   #Taken from data
        return state


    def _return_val(self):
        """ This method returns the current value of the account that the object is tied to in USD."""
        try:
            return self.assets_owned.dot(self.asset_prices) + self.USD
        except TypeError:
            print('TypeError calculating account value.')
            print('Assets owned: ' + str(self.assets_owned))
            print('Asset prices: ' + str(self.asset_prices))
            return 0


    def _fetch_candle_data(self, market, start_date, end_date):
        # this function is useful as code is ran for the same period in backtesting several times consecutively,
        # and fetching from original CSV takes longer as it is a much larger file

        print('Fetching historical data...')

        # get the historic data
        path = self.path_dict['updated history']

        df = pd.DataFrame(columns=['Date', 'BTCOpen', 'BTCHigh', 'BTCLow', 'BTCClose', 'BTCVolume'])

        # Fetch candle data from bittrex
        if end_date > datetime.now() - timedelta(days=10):

            attempts = 0
            while True:
                print('Fetching candles from Bittrex...')
                candle_dict = bittrex_obj.get_candles(
                    market, 'oneMin')
                print(market)

                if candle_dict['success']:
                    candle_df = process_candle_dict(candle_dict)
                    print("Success getting candle data.")
                    break
                else:
                    print("Failed to get candle data.")
                    time.sleep(2*attempts)
                    attempts +=1

                    if attempts == 5:
                        raise(TypeError)
                    print('Retrying...')

            df = df.append(candle_df)


        if df.empty or df.Date.min() > start_date:  # try to fetch from updated
            print('Fetching data from cumulative data repository.')

            def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p-%M")
            up_df = pd.read_csv(path, parse_dates=['Date'], date_parser=dateparse)

            if up_df.empty:
                print('Cumulative data repository was empty.')
            else:
                print('Success fetching from cumulative data repository.')
                df = df.append(up_df)

        if df.empty or df.Date.min() > start_date:  # Fetch from download file (this is last because its slow)

            print('Fetching data from the download file.')
            # get the historic data. Columns are Timestamp	Open	High	Low	Close	Volume_(BTC)	Volume_(Currency)	Weighted_Price

            def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
            orig_df = pd.read_csv(self.path_dict['downloaded history'], usecols=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(Currency)'], parse_dates=[
                'Timestamp'], date_parser=dateparse)
            orig_df.rename(columns={'Timestamp': 'Date', 'O': 'BTCOpen', 'H': 'BTCHigh',
                                    'L': 'BTCLow', 'C': 'BTCClose', 'V': 'BTCVolume'}, inplace=True)

            assert not orig_df.empty

            df = df.append(orig_df)

        # Double check that we have a correct date date range. Note: will still be triggered if missing the exact data point
        assert(df.Date.min() <= start_date)

        # if df.Date.max() < end_date:
        #     print('There is a gap between the download data and the data available from Bittrex. Please update download data.')
        #     assert(df.Date.max() >= end_date)

        df = df[df['Date'] >= start_date]
        df = df[df['Date'] <= end_date]
        df = format_df(df)

        return df


    def cancel_all_orders(self):
        """This method looks for any open orders associated with the account, and cancels those orders."""

        print('Canceling any open orders...')
        open_orders = self.bittrex_obj_1_1.get_open_orders('USD-BTC')
        if open_orders['success']:
            if not open_orders['result']:
                print('No open orders.')
            else:
                for order in open_orders['result']:
                    uuid = order['OrderUuid']
                    cancel_result = self.bittrex_obj_1_1.cancel(uuid)['success']
                    if cancel_result == True:  # need to see if im checking if cancel_result exits or if im checking its value
                        print('Cancel status: ', cancel_result, ' for order: ', uuid)

        else:
            print('Failed to get order history.')
            print(result)


    def _trade(self, currency_pair, amount):
        """This method will execute a limit order trade for the 'amount' in USD passed. The limit is set at a price
        similar to the mean of the order book. THe method accepts positive or negative values in the 'amount' field.
        A positive value indicates buying, and a negative value indicates selling. """

        # Note that bittrex exchange is based in GMT 8 hours ahead of CA

        self.get_prices(currency_pair)


        start_time = datetime.now() #this is for tracking how long the order has been open

        # Enter a trade into the market.
        #The bittrex.bittrex buy_limit method takes 4 arguments: market, amount, rate
        # Example result  {'success': True, 'message': '', 'result': {'uuid': '2641035d-4fe5-4099-9e7a-cd52067cde8a'}}
        spread = .0003
        if amount > 0:  # buy
            rate = round(self.asset_prices[0]*(1 - spread/2), 3)
            amount_currency = round(amount/rate, 4)
            trade_result = self.bittrex_obj_1_1.buy_limit(self.markets[0], amount_currency, rate)
            side = 'buying'
        else:       # Sell
            rate = round(self.asset_prices[0]*(1 + spread/2), 3)
            amount_currency = round(-amount/rate, 4)
            trade_result = self.bittrex_obj_1_1.sell_limit(self.markets[0], amount_currency, rate)
            side = 'selling'

        # Check that an order was entered
        if not trade_result['success']:
            print('Trade attempt failed: ', end = ' ')
            print(trade_result['message'])
            return False
        else:
            print(f'Order for {side} {amount_currency:.8f} {self.markets[0][4:7]} at a price of ${self.asset_prices[0]:.2f} has been submitted to the market.')
            order_uuid = trade_result['result']['uuid']

            # Loop for a time to see if the order has been filled
            status = self._monitor_order_status(order_uuid)

            if status == True:
                print(f'Order has been filled. Id: {order_uuid}.')
                return True
            else:
                # print('Order not filled')
                return False

                dt = datetime.now() - start_time  # note that this includes the time to run a small amount of code

                order_data['result']['Order Duration'] = dt
                trade = _process_order_data(order_data)
                # print(trade)


    def _monitor_order_status(self, uuid, time_limit = 30):
        """This method loops for a maximum duration of timelimit seconds, checking the status of the open order uuid that is passed.
        If the timelimit is reached, the order is cancelled. If the """

        start_time = datetime.now()
        # Loop to see if the order has been filled
        is_open = True
        cancel_result = False
        while is_open:
            order_data = self.bittrex_obj_1_1.get_order(uuid)
            try:
                is_open = order_data['result']['IsOpen']
            except TypeError:
                print('TypeError getting open status on order. Open status: ' + is_open)
                # print('Order open status: ', is_open)

            #Case: order filled
            if not is_open:
                return True
                break

            time_elapsed = datetime.now() - start_time

            #Case: time limit reached
            if time_elapsed >= timedelta(seconds=time_limit):
                print(f'Order has not gone through in {time_limit} seconds. Cancelling...')
                # Cancel the order POTENTIALLY NEED TO THIS TO LOOP TO MAKE SURE THE ORDER IS CANCELLED
                cancel_result = self.bittrex_obj_1_1.cancel(uuid)['success']
                if cancel_result == True:  # need to see if im checking if cancel_result exits or if im checking its value
                    print(f'Cancel status {cancel_result} for order: {uuid}.')
                    return False
                    break #Break out of order status loop
            time.sleep(1)


    def _process_order_data(self, dict):
        """Needs to be updated"""
        # Example input: {'success': True, 'message': '',
        #'result': {'AccountId': None, 'OrderUuid': '3d87588d-70d6-4b40-a723-11248aaaff8b', 'Exchange': 'USD-BTC', 'Type': 'LIMIT_SELL', 'Quantity': 0.00123173, 'QuantityRemaining': 0.0, 'Limit': 1.3, 'Reserved': None, 'ReserveRemaining': None, 'CommissionReserved': None, 'CommissionReserveRemaining': None, 'CommissionPaid': 0.02498345, 'Price': 9.99338392, 'PricePerUnit': 8113.29099722, 'Opened': '2019-11-19T07:42:48.85', 'Closed': '2019-11-19T07:42:48.85', 'IsOpen': False, 'Sentinel': None, 'CancelInitiated': False, 'ImmediateOrCancel': False, 'IsConditional': False, 'Condition': 'NONE', 'ConditionTarget': 0.0}}

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


    def get_all_order_history(self):
        """This method retrieves trade history for all currency pairs from the exchange, creates a dataframe with the orders,
        and then appends them to the CSV trade log. Trade history is stored locally since the bittrex API only returns trades
        that happened within a recent timeframe. I am not sure what that time frame is."""

        #This section fetches order data from the exchange
        for i, currency_pair in enumerate(self.markets):
            order_history_dict = self.bittrex_obj_1_1.get_order_history(market = currency_pair)
            # print(order_history_dict)
            order_df = pd.DataFrame(order_history_dict['result'])
            order_df.drop(columns=['IsConditional', 'Condition', 'ConditionTarget',
                                   'ImmediateOrCancel', 'Closed'], inplace=True)

            order_df.reset_index(inplace=True, drop=True)
            # dates into datetimess
            order_df.TimeStamp = pd.to_datetime(order_df.TimeStamp, format="%Y-%m-%dT%H:%M:%S")
            order_df.set_index('TimeStamp', drop = True, inplace = True)


            if i == 0: df = order_df
            else: df = df.append(order_df)

        #Should check right here if there is actually new data

        #This section reads in the order log, and appends any new data
        save_path = path_dict['order log']

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
        print(df)


    def _act(self, action):
        # index the action we want to perform
        # action_vec = [(desired amount of stock 1), (desired amount of stock 2), ... (desired amount of stock n)]

        action_vec = self.action_list[action] #a vectyor like [0.1, 0.5] own 0.1*val of BTC, 0.5*val of ETH etc.

        if action_vec != self.last_action:  # if attmepting to change state

            #THIS WILL NEED TO BE MORE COMPLEX IF MORE ASSETS ARE ADDED
            #Calculate the changes needed for each asset
            delta = [s_prime - s for s_prime, s in zip(action_vec, self.last_action)]
