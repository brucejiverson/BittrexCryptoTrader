# from bittrex.bittrex import *
import mplfinance as mpf
import plotly.graph_objects as go

from features.feature_constructor import featuresHandler
from tools.tools import f_paths, maybe_make_dir, printProgressBar, ROI
from environments.environment_types import ExchangeEnvironment, AccountLog
import pandas as pd
import math
from datetime import datetime, timedelta
# import itertools
import numpy as np
# import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller

# cryptodatadownload has gaps
# Place to download: kaggle  iSinkInWater, brucejamesiverson@gmail.com, I**********
# This has data for all currencies, 10 GB, too big for now https://www.kaggle.com/jorijnsmit/binance-full-history

# The below should be updated to be simplified to use parent directory? unsure how that works...
# https://stackoverflow.com/questions/48745333/using-pandas-how-do-i-save-an-exported-csv-file-to-a-folder-relative-to-the-scr?noredirect=1&lq=1
 

class SimulatedCryptoExchange(ExchangeEnvironment):
    """A multi-asset trading environment.
    For now this has been adopted for only one asset.
    Below shows how to add more.
    The state size and the aciton size throughout the rest of this
    program are linked to this class.
    State: vector of size 7 (n_asset + n_asset*n_indicators)
      - stationary price of asset 1 (using BTCClose price)
      - associated indicators for each asset
    """

    def __init__(self, 
                start = datetime.now() - timedelta(days = 30), 
                end = datetime.now(),
                train_amount=30,
                initial_investment=100,
                granularity=1,   # in minutes
                feature_dict=None):
                
        super().__init__(granularity, feature_dict)
        # print(feature_dict)
        """The data, for asset_data can be thought of as nested arrays, where indexing the
        highest order array gives a snapshot of all data at a particular time, and information at the point
        in time can be captured by indexing that snapshot."""

        self._fetch_candle_data(start, end)
        print("PRICE HISTORY DATA: ")
        print(self.candle_df.head())
        print(self.candle_df.tail())

        #Convention here is string key, list of hyperparams typically for multiple of the feature type
        # This is how many of the previous states to include
        
        self.featuresHandler = featuresHandler(feature_dict=feature_dict) # This fills in the asset_data array
        self.df = self.featuresHandler.build_basic_features(self.candle_df, self.markets)
        test_df, train_df = self.featuresHandler.train_predictors(self.df, self.markets, train_amount=train_amount)
        self.df = test_df.copy()

        self.asset_data = self.df.values
        print('PREPARED DATA:')
        print(self.df.head())
        print(self.df.tail())
        # n_step is number of samples, n_stock is number of assets. Assumes to datetimes are included
        self.n_step = self.asset_data.shape[0]

        # instance attributes
        self.initial_investment = initial_investment
        self.mean_spread = .00003 #fraction of the price to use as spread when placing limit orders
        self.cur_step = None

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
        self.asset_prices = self.asset_data[self.cur_step][0:self.n_asset] #assumes data is asset prices and then volums

        self.USD = self.initial_investment

        self.log = AccountLog()

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
        btc_amt = self.assets_owned[0]*self.asset_prices[0]

        if self.should_log:
            # new_info = {'$ of BTC':btc_amt, 'Total Value':cur_val, 'Action':action, 'Timestamp':self.df.iloc[self.cur_step].name}
            new_info = [btc_amt, cur_val, action, self.df.iloc[self.cur_step].name]
            self.log.update(new_info)

        def log_ROI(initial, final):
            """ Returns the log rate of return, which accounts for how percent changes "stack" over time
            For example, a 10% increase followed by a 10% decrease is truly a 1% decrease over time (100 -> 110 -> 99)
            Arithmetic ROI would show an overall trend of 0%, but log ROI properly computes this to be -1%"""
            return round(np.log(final/initial), 4) *100

        reward = log_ROI(prev_val, cur_val)#(cur_val - prev_val) #this used to be more complicated

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # conform to the Gym API
        #      next state       reward  flag  info dict.
        return self._get_state(), self._get_val(), reward, done


    def _get_val(self):
        return self.assets_owned.dot(self.asset_prices) + self.USD


    def _get_state(self):
        """This method returns the state, which is an observation that has been transformed to be stationary.
        Note that the state could later be expanded to be a stack of the current state and previous states
        The structure of the state WAS PREVIOUSLY [amount of each asset held, value of each asset, cash in hand, volumes, indicators]
        The structure of the state IS NOW [value of each asset, volumes, indicators]
        For reference if reconstructing the state, the data is ordered in self.asset_data as [asset prices, asset volumes, asset indicators]"""

        #assets_owned, USD, volume, indicators
        state = np.empty(self.state_dim)

        #Instituted a try catch here to help with debugging and potentially as a solution to handling invalid/inf values in log
        state = self.asset_data[self.cur_step]
        return state


    def _trade(self, action):
        # index the action we want to perform
        # action_vec = [(desired amount of stock 1), (desired amount of stock 2), ... (desired amount of stock n)]

        # get current value before performing the action
        action_vec = self.action_list[action]

        cur_price = self.asset_prices[0]
        bid_price = cur_price*(1 - self.mean_spread/2)
        ask_price = cur_price*(1 + self.mean_spread/2)

        cur_val = self._get_val()

        if action_vec != self.last_action:  # if attmepting to change state

            #Calculate the changes needed for each asset
            # delta = [s_prime - s for s_prime, s in zip(action_vec, self.last_action) #not using this now, but how it should be done
            fees = .002                 # Bittrex rate for both maker and taker for 30 day volume < $50k

            #Below this is the old way
             # Sell everything
            for i, a in enumerate(action_vec):
                self.USD += self.assets_owned[i] * self.asset_prices[i] * (1 - fees)
                self.assets_owned[i] = 0

            # Buy back the right amounts
            for i, a in enumerate(action_vec):
                cash_to_use = a * self.USD
                self.assets_owned[i] = cash_to_use / ask_price
                self.USD -= cash_to_use


            # print("Initial val: " + str(cur_val) + ". Post trade val:" + str(self._get_val()))
            self.last_action = action_vec
            self.period_since_trade = 0
            """
            # print('Evaluating whether to buy or sell...')
            for i, a in enumerate(action_vec): #for selling assets (must happen first)

                current_holding = (self.assets_owned[i]*self.asset_prices[i])/cur_val       #amount of coin currently held as fraction of total portfolio value, between 0 and 1
                currency_pair = self.markets[i]                         #which currency pair is being evaluated
                decimal_diff = a - current_holding                      #want minus have
                threshhold = 0.05                                       #trades only executed if difference between want and have is sufficiently high enough


                if -decimal_diff > threshhold:                          #sell if decimal_diff is sufficiently negative
                    # print("Aw jeez, I've got " + str(round(-decimal_diff*100,2)) + "% too much of my portfolio in " + str(currency_pair[4:]))
                    trade_amount = decimal_diff * cur_val               #amount to sell of coin in USD, formatted to be neg for _trade logic
                    self.USD -+ trade_amount
                    self.assets_owned[i] += trade_amount/self.asset_prices[i] #bid_price

            for i, a in enumerate(action_vec): #for buying assets
                current_holding = (self.assets_owned[i]*self.asset_prices[i])/cur_val       #amount of coin currently held as fraction of total portfolio value, between 0 and 1
                currency_pair = self.markets[i]                         #which currency pair is being evaluated
                decimal_diff = a - current_holding                      #want minus have
                threshhold = 0.05                                       #trades only executed if difference between want and have is sufficiently high enough

                if decimal_diff > threshhold:                         #buy if decimal_diff is sufficiently positive
                    # print("Oh boy, time to spend " + str(round(decimal_diff*100,2)) + "% of my portfolio on " + str(currency_pair[4:]))
                    trade_amount = decimal_diff * cur_val               #amount to buy of coin in USD, formatted to be pos for _trade logic
                    self.USD -+ trade_amount
                    self.assets_owned[i] += trade_amount/self.asset_prices[i] #ask_price"""


class BittrexExchange(ExchangeEnvironment):
    """This class provides an interface with the Bittrex exchange for any and all operations. It inherites from the 'ExchangeEnvironment
    class, which ensures similar architecture between different environments. Methods for this class include executing trades,
    logging account value over time, displaying account value over time, retrieving information on prices, balances, and orders from
    Bittrex, and uses a similar 'act' method to interface with agents."""

    def __init__(self, 
                train_amount=30,   # in days
                granularity=5,      # in minutes
                feature_dict=None,
                money_to_use=20,
                window_size=30,
                verbose=1):         # from 0 (no messages) to 3 (lots of messages)

        super().__init__(granularity, feature_dict)
        # Note that in this class, there is the candle_df, which has the candle data 
        # At the proper granularity, the 'df', which has the dataframe with all features 
        # also at the the proper gran, and additionally the candle_1min_df
        self.verbose = verbose
        # instance attributes
        self.initial_investment = money_to_use
        self.feature_dict = feature_dict
        self.asset_volumes = None
        self.window_size = window_size
        
        self.featuresHandler = featuresHandler(feature_dict=self.feature_dict) # This fills in the asset_data array
        

    def reset(self):
        # Resets the environement to the initial state
        print('Resetting the environment object')
        end = datetime.now()
        start = end - timedelta(self.window_size)

        self._fetch_candle_data(start, end)
        print('CANDLE DATA:')
        print(self.candle_df.tail())

        if self.verbose >=3:
            print('building basic features')

        self.df = self.featuresHandler.build_basic_features(self.candle_df, self.markets)
        if self.verbose >=3:
            print('training predictors')
        test_df, train_df = self.featuresHandler.train_predictors(self.df, self.markets, train_amount=14)   # this trains the predictor
        self.df = test_df.copy()
        if self.verbose >=3:
            print('predicting')
        self.df = self.featuresHandler.predict(self.df)

        self.asset_data = self.df.values
        print('PREPARED DATA:')
        print(self.df.tail())

        self.cancel_all_orders()
        self.get_all_balances()
        if self.verbose >= 1:
            self.print_account_health()

        # #Put all money into USD
        # if self.assets_owned[0] > 0:
        #     success = False
        #     while not success:
        #         success = self._trade(-self.assets_owned[0])

        return self._get_state() #, self._get_val()


    def update(self):
        end = datetime.now()
        start = datetime.now() - timedelta(days=self.window_size)

        self._fetch_candle_data(start, end)

        self.df = self.featuresHandler.build_basic_features(self.candle_df, self.markets)
        self.df = self.featuresHandler.predict(self.df)

        self.asset_data = self.df.values
        self.get_all_balances()
    
        return self._get_state()


    def _get_state(self):
        # Returns the state (for now state, and observation are the same.)
        # Note that the state could be a transformation of the observation, or
        # multiple past observations stacked.)
        state = np.empty(self.state_dim)  #assets_owned, USD

        penult_row = self.df.iloc[-2].values
        ult_row = self.df.iloc[-1].values                #last row in slice_df an array

        # for i, a in enumerate(self.markets):
        # ult_row[0+2*i] -= penult_row[0+2*i]                # correcting each market's close to be a delta rather than its value
        # ult_row[1+2*i] -= penult_row[1+2*i]                # correcting each market's volume to be a delta

        state = ult_row

        if self.verbose >= 3:
            print(self.df.iloc[-1])
        return state


    def act(self, action):
        """
        action_vec = [(desired amount of stock 1), (desired amount of stock 2), ... (desired amount of stock n)]
        action_vec = self.action_list[action] #a vector like [0.1, 0.5] own 0.1*val of BTC, 0.5*val of ETH etc.
        """
        #currently set up for only bitcoin
        # index the action we want to perform
        # action_vec = [(desired amount of stock 1), (desired amount of stock 2), ... (desired amount of stock n)]
            
        # get current value before performing the action
        action_vec = self.action_list[action]

        cur_val = self._get_val()
        if self.verbose >= 3:
            print(f'Action given to exchange: {action}')        
            print(f'action_list: {self.action_list}')
            print(f'Action_vec: {action_vec}')
            print(f'Current account value: {cur_val}')


        if action_vec != self.last_action:  # if attmepting to change state
            #Calculate the changes needed for each asset
            # delta = [s_prime - s for s_prime, s in zip(action_vec, self.last_action) #not using this now, but how it should be done
            print('Evaluating whether to buy or sell...')
            for i, a in enumerate(action_vec): #for selling assets (must happen first)

                """
                4/18 - This is currently structured to work with a simple USD-BTC pairing only. Eventually, this will need to have the following logic:

                -cycle through every element in action_vec and see if you need to sell any of that coin
                    -execute all sells as they come up to stock up on usd
                -cycle through every element in action_vec and see if you need to buy any of that coin
                    -<should probably find a way to write this such that it only cycles through the elements where selling did not occur for efficiency>
                    -execute all buys
                """

                current_holding = (self.assets_owned[i]*self.asset_prices[i])/cur_val       #amount of coin currently held as fraction of total portfolio value, between 0 and 1
                currency_pair = self.markets[i]                         #which currency pair is being evaluated
                decimal_diff = a - current_holding                      #want minus have
                threshhold = 0.05                                       #trades only executed if difference between want and have is sufficiently high enough

                if self.verbose >= 3:
                    print(f'Current fraction in BTC: {current_holding}.')
                    print(f'Difference between current fraction and action: {decimal_diff}')

                if -decimal_diff > threshhold:                          #sell if decimal_diff is sufficiently negative
                    if self.verbose >= 1:
                        print(f"Aw jeez, I've got {str(round(-decimal_diff*100,2))} % too much of my portfolio in {str(currency_pair[4:])}")

                    trade_amount = min(decimal_diff * cur_val, self.assets_owned[i]*self.asset_prices[i]*.99)               #amount to sell of coin in USD, formatted to be neg for _trade logic
                    if self.verbose >= 2:
                        print(f'Trade amount give to _trade: {trade_amount}')
                    self._trade(currency_pair, trade_amount)            #pass command to sell trade @ trade_amount

                elif decimal_diff > threshhold:                         #buy if decimal_diff is sufficiently positive
                    if self.verbose >= 1:
                        print(f'Oh boy, time to spend {str(round(decimal_diff*100,2))} % of my portfolio on {str(currency_pair[4:])}')

                    # trade_amount = min(decimal_diff * cur_val, self.assets_owned[i]*self.asset_prices[i]*.99)               #amount to sell of coin in USD, formatted to be neg for _trade logic
                    trade_amount = decimal_diff*cur_val
                    if self.verbose >= 2:
                        print(f'Trade amount give to _trade: {trade_amount}')
                    self._trade(currency_pair, trade_amount)            #pass command to sell trade @ trade_amount in USD

        btc_amt = self.assets_owned[0]*self.asset_prices[0]                              # !!! only stores BTC and USD for now
        cur_val = btc_amt + self.USD
        # new_info = {'$ of BTC':btc_amt, 
        #             'Total Value':cur_val,
        #             'Action': action,
        #             'Timestamp':self.df.iloc[-1].name}
        new_info = [btc_amt, cur_val, action, self.df.iloc[-1].name]
        self.log.update(new_info)
        if self.verbose >= 2:
            print('Log has been updated.')

    def _get_val(self):
        """ This method returns the current value of the account that the object
        is tied to in USD. VALIDTED"""
        try:
            return self.assets_owned.dot(self.asset_prices) + self.USD
        except TypeError:
            print('TypeError calculating account value.')
            print('Assets owned: ' + str(self.assets_owned))
            print('Asset prices: ' + str(self.asset_prices))
            return 0


    def _get_current_prices(self):
        """This method retrieves up to date price information from the exchange.
        VALIDATED"""

        for i, market in enumerate(self.markets):
            token = market[4:7]
            print('Fetching ' + token + ' price... ', end = ' ')
            attempts_left = 3
            while True:
                ticker = self.bittrex_obj_1_1.get_ticker(market)
                #Change this to make sure that the check went through
                # Check that an order was entered
                if not ticker['success']:
                    print('_get_prices failed. Ticker message: ', end = ' ')
                    print(ticker['message'])

                    if attempts_left == 0:
                        print('_get_current_prices has failed. exiting the method...')
                        return None, None
                    else:
                        attempts_left -= 1
                        time.sleep(1)
                        print('Retrying...')
                else: #success
                    # print(ticker['result'])
                    print('success.')
                    self.asset_prices[i] = ticker['result']['Last']
                    return ticker['result']['Bid'], ticker['result']['Ask']


    def get_latest_candle(self, currency_pair):
        """This method fetches recent candle data on a specific market.
        currency_pair should be a string, 'USD-BTC' """

        for i, market in enumerate(self.markets):
            print('Fetching last candle for ' + market + ' from the exchange.')
            attempts_left = 3
            while True:
                print('Fetching candles from Bittrex...', end = " ")
                candle_dict = self.bittrex_obj_2.get_latest_candle(market, 'oneMin')

                if candle_dict['success']:
                    candle_df = self._parse_candle_dict(candle_dict, market)
                    print("Success.")
                    print(candle_df)
                    break
                else: #If there is an error getting the proper data
                    print("Failed to get candle data. Candle dict: ", end = ' ')
                    print(candle_dict)
                    time.sleep(2*attempts)
                    attempts -= 1

                    if attempts == 0:
                        print('Exceeded maximum number of attempts.')
                        raise(TypeError)
                    print('Retrying...')


    def get_all_balances(self):
        """This method retrieves the account balances for each currency including
         USD from the exchange."""

        print('Fetching account balances...', end = ' ')

        self.assets_owned = np.zeros(self.n_asset)
        for i, currency_pair in enumerate(self.markets):
            token = currency_pair[4:7]
            attempts_left = 3
            while attempts_left >= 0:
                balance_response = self.bittrex_obj_1_1.get_balance(token)

                if balance_response['success']:
                    print(token + ' balance fetched.', end = ' ')
                    amount = balance_response['result']['Balance']

                    if amount is None:
                        self.assets_owned[i] = 0
                    else: self.assets_owned[i] = amount
                    break
                else:
                    print('Error fetching balances.')
                    print(balance_response)

                attempts_left -= 1
        #Get USD
        attempts_left = 3
        while attempts_left >= 0:
            balance_response = self.bittrex_obj_1_1.get_balance('USD')
            if balance_response['success']:
                print('USD balance fetched.')
                amount = balance_response['result']['Balance']

                if amount is None:
                    self.USD = 0
                else: #Balance is 0
                    self.USD = amount
                break
            else:
                print('Error fetching balances.')
                print(balance_response)
            attempts_left -= 1


    def print_account_health(self):
        """This method prints out a variety of information about the account. In the future this should print a dataframe
        for cleaner formatting (if mutliple currency trading strategies are implemented)"""

        self.get_all_balances()
        self._get_current_prices()

        index = ['USD', *[x[4:7] for x in self.markets]]\
        
        try:
            dict = {'Amount of currency': [round(self.USD, 2), *self.assets_owned], 'Value in USD':  [round(self.USD, 2), *self.assets_owned*self.asset_prices]}
        except TypeError:
            print('Some value was not initialized:')
            print(f'USD          {self.USD}')
            print(f'Assets owned {self.assets_owned}')
            print(f'Asset prices {self.asset_prices}')
            raise(TypeError)
            # print(f'USD {self.USD}')
            # print(f'USD {self.USD}')
        df = pd.DataFrame(dict, index = index)

        print('\nCURRENT ACCOUNT INFO:')
        print(f'Total Account Value: {round(float(self._get_val()),2)}')
        print(df)
        print(' ')


    def cancel_all_orders(self):
        """This method looks for any open orders associated with the account,
        and cancels those orders. VALIDATED"""

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
            print(open_orders)


    def _trade(self, currency_pair, amount):
        """This method will execute a limit order trade for the 'amount' in USD passed. The limit is set at a price
        similar to the mean of the order book. THe method accepts positive or negative values in the 'amount' field.
        A positive value indicates buying, and a negative value indicates selling. VALIDATED"""

        # Note that bittrex exchange is based in GMT 8 hours ahead of CA

        bid, ask = self._get_current_prices()

        # Enter a trade into the market.
        #The bittrex.bittrex buy_limit method takes 4 arguments: market, amount, rate
        # Example result  {'success': True, 'message': '', 'result': {'uuid': '2641035d-4fe5-4099-9e7a-cd52067cde8a'}}
        
        if amount > 0:  # buy
            rate = ask

            amount_currency = round(amount/rate, 6)

            most_possible = round(self.USD/rate * .997, 6)

            if self.verbose >= 3:
                print(f'Amount of BTC: {amount_currency}. Calculated most possible: {most_possible}')
                print(f'Price: {rate}')

            if self.verbose >= 1:
                print(f'Buying ${amount_currency} of pair {currency_pair}')

            if amount_currency > most_possible:
                amount_currency = most_possible
            
            if self.verbose >= 2:
                print(f'Trade amount entered to exchange: {amount_currency}')
            coin_index = self.markets.index(currency_pair)          #index of currency pair in market list to correlate to trade amounts
            order_entry_status = self.bittrex_obj_1_1.buy_limit(currency_pair, amount_currency, rate)
            side = 'buying'

        else:       # Sell
            # cur_price is last price, meaning the last that was traded on the exchange
            rate = bid      # round(self.asset_prices[0]*(1-self.mean_spread/2), 3)

            amount_currency = round(-amount/rate, 6)
            most_possible = round(self.assets_owned[0], 6)

            if self.verbose >= 3:
                print(f'Amount of BTC: {amount_currency}. Calculated most possible: {most_possible}')

            if amount_currency > most_possible:
                amount_currency = most_possible

            if self.verbose >= 1:
                print(f'Buying ${amount_currency} of pair {currency_pair}')

            coin_index = self.markets.index(currency_pair)          #index of currency pair in market list to correlate to trade amounts
            order_entry_status = self.bittrex_obj_1_1.sell_limit(currency_pair, amount_currency, rate)
            side = 'selling'

        # Check that an order was entered
        if not order_entry_status['success']:
            if self.verbose >= 1:
                print('Trade attempt for ' + side + ' failed: ', end = ' ')
                print(order_entry_status['message'])
                if order_entry_status['message'] == 'INSUFFICIENT_FUNDS':
                    print('Amount: ' + str(amount))
            return False
        else: #order has been successfully entered to the exchange
            if self.verbose >= 1:
                print(f'Order for {side} {amount_currency:.8f} {currency_pair[4:]} at a price of ${rate:.2f} has been submitted to the market.')
            
            uuid = order_entry_status['result']['uuid']
            # Loop for a time to see if the order has been filled
            order_is_filled = self._monitor_order_status(uuid) #True if order is filled

            if order_is_filled == True:
                self.get_all_balances()        #updating with new amount of coin
                if self.verbose >=1:
                    print(f'Order has been filled. uuid: {uuid}.')
                    self.print_account_health()

            #this saves the information regardless of if the trade was successful or not
            self._get_and_save_order_data(uuid)


    def _monitor_order_status(self, uuid, time_limit = 30):
        """This method loops for a maximum duration of timelimit seconds, checking the status of the open order uuid that is passed.
        If the timelimit is reached, the order is cancelled. VALIDATED"""

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


    def _get_and_save_order_data(self, uuid):
        """This method fetches information on a specific order from the exchange.
        # Example dictionary from Bittrex API is:
        {'success': True, 'message': '', 'result': {'AccountId': None, 'OrderUuid': '3d87588d-70d6-4b40-a723-11248aaaff8b', 'Exchange': 'USD-BTC',
        'Type': 'LIMIT_SELL', 'Quantity': 0.00123173, 'QuantityRemaining': 0.0, 'Limit': 1.3, 'Reserved': None, 'ReserveRemaining': None,
        'CommissionReserved': None, 'CommissionReserveRemaining': None, 'CommissionPaid': 0.02498345, 'Price': 9.99338392, 'PricePerUnit': 8113.29099722,
        'Opened': '2019-11-19T07:42:48.85', 'Closed': '2019-11-19T07:42:48.85', 'IsOpen': False, 'Sentinel': None, 'CancelInitiated': False,
        'ImmediateOrCancel': False, 'IsConditional': False, 'Condition': 'NONE', 'ConditionTarget': 0.0}}"""


        print('Fetching order data...', end = ' ')
        date_format = "%Y-%m-%dT%H:%M:%S"
        path = f_paths['order log']

        dictionary = self.bittrex_obj_1_1.get_order(uuid)

        # in order to construct a df, the values of the dictionary cannot be scalars, must be lists, so convert to lists
        results = {}
        for key in dictionary['result']:
            results[key] = [dictionary['result'][key]]
        order_df = pd.DataFrame(results)

        order_df.drop(columns=['AccountId', 'Reserved', 'ReserveRemaining', 'CommissionReserved', 'CommissionReserveRemaining',
                                'Sentinel', 'IsConditional', 'Condition', 'ConditionTarget', 'ImmediateOrCancel', 'CancelInitiated',
                                'ImmediateOrCancel', 'IsConditional', 'Condition', 'ConditionTarget'], inplace=True)
        # order_df = order_df.rename(columns={'CommissionPaid': 'Commission'})

        # date strings into datetimes
        order_df.Closed = pd.to_datetime(order_df.Closed, format=date_format)
        order_df.Opened = pd.to_datetime(order_df.Opened, format=date_format)
        order_df.set_index('Opened', inplace = True, drop = True)
        print('fetched.')

        # print('ORDER INFO:')
        # print(order_df.columns)

        # Load the trade log from csv
        # Note that the dateformat for this is different than for the price history.
        # The format is the same in csv as the bittrex API returns for order data
        df = pd.read_pickle(path)
        # df.set_index('Opened', inplace = True, drop = True)
        df = df.append(order_df, sort = False)

        # Format and save the df
        df.sort_index(inplace = True)
        df.to_pickle(path)
        print('Order log binary file has been updated.')


    def view_order_data(self):

        date_format = "%Y-%m-%dT%H:%M:%S"
        def dateparse(x):
            try:
                return pd.datetime.strptime(x, date_format)
            except ValueError:  #handles cases for incomplete trades where 'Closed' is NaT
                return x
        # df = pd.read_csv(f_paths['order log'], parse_dates = ['Opened', 'Closed'], date_parser=dateparse)
        # df.set_index('Opened', inplace = True, drop = True)
        df = pd.read_pickle(f_path['order log'])
        print(' ')
        print('ALL ORDER INFORMATION:')
        print(df)
        print(' ')


    def get_and_save_order_history(self):
        """FOR NOW I AM LEAVING THIS INCOMPLETE. THE GET_ORDER METHOD RETRIEVE MORE INFORMATION ON
        EACH ORDER, AND IS A MORE COMPLETE METHOD OF LOGGING ORDER INFO.
        This method retrieves trade history for all currency pairs from the exchange, creates a dataframe with the orders,
        and then appends them to the CSV trade log. Trade history is stored locally since the bittrex API only returns trades
        that happened within a recent timeframe. I am not sure what that time frame is."""

        if self.verbose >= 1:
            print('Fetching all recent order data...', end = ' ')
        date_format = "%Y-%m-%dT%H:%M:%S"
        path = f_paths['order log']

        #This section fetches order data from the exchange for each relevant currency pair.
        #This will eventually need to be updated to accomodate for inter asset trading
        maybe_make_dir(path[:-14])
        for i, currency_pair in enumerate(self.markets):
            order_history_dict = self.bittrex_obj_1_1.get_order_history(market = currency_pair)
            order_df = pd.DataFrame(order_history_dict['result'])
            # print(order_df)
            order_df.drop(columns=['IsConditional', 'Condition', 'ConditionTarget',
                                   'ImmediateOrCancel'], inplace=True)
            # print(order_df.columns)
            
            # dates into datetimes
            order_df.TimeStamp = pd.to_datetime(order_df.TimeStamp, format=date_format)
            order_df.Closed = pd.to_datetime(order_df.Closed, format=date_format)
            order_df.set_index('Closed', drop = True, inplace = True)
            order_df.rename({'Commission':'CommisionPaid'})
            #Create or append to the df
            if i == 0: df = order_df
            else: df = df.append(order_df, sort = False)

        if self.verbose >= 1:
            print('fetched.')
        #This section reads in the order log from csv, and appends any new data
        try:
            old_df = pd.read_pickle(path)
            df = df.append(old_df, sort = False)
            df.sort_values(by='Closed', inplace=True)
            df.drop_duplicates('OrderUuid',inplace=True)
            # print(df['Closed'])
            print(df)
            filled_orders = df.loc[~df['Closed'].is_null()]
            print(filled_orders)
        except KeyError:
            print('Order log is empty.')
        if self.verbose >= 1:
            print('Data written to test trade log.')
            # print(df)
            print(df.columns)


    def save_candle_data(self):
        # This function writes the information in the original format to the csv file
        # including new datapoints that have been fetched

        print("Writing data to file.")
        path = f_paths['cum data pickle']
        old_df = pd.read_pickle(path)
        df_to_save = self.candle_df.append(old_df)
        df_to_save.to_pickle(path)


if __name__ == '__main__':
    # This will do data scraping and test sim class
    print("Creating simulation environment.")
    sim_env = SimulatedCryptoExchange()

    # This will create a live class, and get account health
    exchange = BittrexExchange(granularity=1)

    exchange.print_account_health()