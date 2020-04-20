from bittrex.bittrex import *

import pandas as pd
import math
from datetime import datetime, timedelta
import itertools
import numpy as np
import json
import ta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# HEY brucejamesiverson
#cryptodatadownload has gaps
#Place to download: kaggle  iSinkInWater, brucejamesiverson@gmail.com, I**********
#This has data for all currencies, 10 GB, too big for now https://www.kaggle.com/jorijnsmit/binance-full-history
#

#The below should be updated to be simplified to use parent directory? unsure how that works...
#https://stackoverflow.com/questions/48745333/using-pandas-how-do-i-save-an-exported-csv-file-to-a-folder-relative-to-the-scr?noredirect=1&lq=1

paths = {'downloaded csv': 'C:/Python Programs/crypto_trader/historical data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv',
'cum data csv': 'C:/Python Programs/crypto_trader/historical data/cumulative_data_all_currencies.csv',
'secret': "/Users/biver/Documents/crypto_data/secrets.json",
'rewards': 'agent_rewards',
'models': 'agent_models',
'order log': 'C:/Python Programs/crypto_trader/account logs/order_log.csv',
'test trade log':  'C:/Python Programs/crypto_trader/account logs/trade_testingBTCUSD.csv',
'account log': 'C:/Python Programs/crypto_trader/account logs/account_log.csv'}


def ROI(initial, final):
    # Returns the percentage increase/decrease
    return round(final / initial - 1, 2)*100


class ExchangeEnvironment:
    """All other environment classes inherit from this class. This is done to ensure similar architecture
    between different classes (eg simulated vs real exchange), and to make it easy to change that
    architecture by having change be in a single place."""

    def __init__(self):

        #get my keys
        with open("/Users/biver/Documents/crypto_data/secrets.json") as secrets_file:
            keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
            secrets_file.close()

        #Need both versions of the interface as they each provide certain useful functions
        self.bittrex_obj_1_1 = Bittrex(keys["key"], keys["secret"], api_version=API_V1_1)
        self.bittrex_obj_2 = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)

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

        #BELOW IS THE OLD WAY THAT INCLUDES THE CURRENT ASSETS HELD
        # calculate size of state (amount of each asset held, value of each asset, volumes, USD/cash, indicators for each asset)
        # self.state_dim = self.n_asset*3 + 1 + self.n_indicators*self.n_asset
        self.state_dim = self.n_asset*2 + self.n_asset*self.n_indicators #+ self.n_asset #price and volume, and then the indicators, then last state
        self.last_action = [] #The action where nothing is owned


        self.assets_owned = [0]*self.n_asset
        self.asset_prices = [0]*self.n_asset

        self.USD = None
        self.candle_df = None
        self.df = None
        self.transformed_df = None
        self.mean_spread = .0003 #fraction of the price to use as spread when placing limit orders

        # log_columns = [*[x for x in self.markets], 'Value']
        self.should_log = False
        log_columns = ['BTC', 'Total Value']
        self.log = pd.DataFrame(columns=log_columns)


    def _process_candle_dict(self, candle_dictionary, market):
        # Dataframe formatted the same as in other functions
        # V and BV refer to volume and base volume
        ticker = market[4:7]
        df = pd.DataFrame(candle_dictionary['result'])
        df['T'] = pd.to_datetime(df['T'], format="%Y-%m-%dT%H:%M:%S")
        df = df.rename(columns={'T': 'TimeStamp', 'O': ticker + 'Open', 'H': ticker +'High', 'L': ticker +'Low', 'C': ticker +'Close', 'V': ticker +'Volume'})
        df.set_index('TimeStamp', drop = True, inplace = True)
        df.drop(columns=["BV"])

        #Reorder the columns
        df = df[[ticker +'Open', ticker +'High', ticker +'Low', ticker +'Close', ticker +'Volume']]
        # print(df.head())
        return df


    def fetch_data(self, start_date, end_date):
        """This function pulls data from the exchange, the cumulative repository, and the data download in that order
        depending on the input date range."""

        print('Fetching historical data...')

        #construct the column names to use in the output df
        cols = []
        for market in self.markets:
            for item in ['Open', 'High', 'Low', 'Close', 'Volume']:
                cols.append(market[4:7] + item)

        df = pd.DataFrame(columns=cols)

        # Fetch candle data from bittrex for each market
        if end_date > datetime.now() - timedelta(days=9): #this may need to be shifted to account for time offset
            for i, market in enumerate(self.markets):
                print('Fetching ' + market + ' historical data from the exchange.')
                attempts = 0
                while True:
                    print('Fetching candles from Bittrex...', end = " ")
                    candle_dict = self.bittrex_obj_2.get_candles(market, 'oneMin')

                    if candle_dict['success']:
                        candle_df = self._process_candle_dict(candle_dict, market)
                        print("Success.")
                        # print(candle_df.head())
                        break
                    else: #If there is an error getting the proper data
                        print("Failed to get candle data. Candle dict: ", end = ' ')
                        print(candle_dict)
                        time.sleep(2*attempts)
                        attempts += 1

                        if attempts == 5:
                            print('Exceeded maximum number of attempts.')
                            raise(TypeError)
                            print('Retrying...')
                            #The below logic is to handle joinging data from multiple currencies
                if i == 0: test = df.append(candle_df, sort = True)
                else: test = df.join(candle_df)
                # print('CANDLE DF:')
                # print(candle_df.tail())
                df = df.append(candle_df, sort = True) #ok this works
                # print(df.tail())

        path = paths['cum data csv']

        if df.empty or df.index.min() > start_date:  # try to fetch from updated
            print('Fetching data from cumulative data repository.')

            def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%d %I-%p-%M")
            up_df = pd.read_csv(path, index_col = 'TimeStamp', parse_dates = True, date_parser=dateparse)

            assert not up_df.empty
            print('Success fetching from cumulative data repository.')
            up_df.set_index('TimeStamp', drop = True, inplace = True)
            # print(df.tail())
            # print('CUM DATA DF:')
            # print(up_df.head())
            # print(up_df.tail())
            df = df.append(up_df, sort = True)
            # print('DATA INCL CUMULATIVE DATA REPOSITORY:')
            # print(df.head())
            # print(df.tail())

        if df.empty or df.index.min() > start_date:  # Fetch from download file (this is last because its slow)

            print('Fetching data from the download file.')
            # get the historic data. Columns are TimeStamp	Open	High	Low	Close	Volume_(BTC)	Volume_(Currency)	Weighted_Price

            def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
            cols_to_use = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(Currency)']
            orig_df = pd.read_csv(paths['downloaded csv'], usecols=cols_to_use, parse_dates = ['Timestamp'], date_parser=dateparse)

            name_map = {'Open': 'BTCOpen', 'High': 'BTCHigh', 'Low': 'BTCLow', 'Close': 'BTCClose', 'Volume_(Currency)': 'BTCVolume'}
            orig_df.rename(columns= name_map, inplace=True)
            orig_df.set_index('Timestamp', inplace = True, drop = True)

            assert not orig_df.empty

            df = df.append(orig_df, sort = True)
            df.sort_index(inplace = True) #apparently this is necessary otherwise the orig_df is in reverse order

        # Double check that we have a correct date date range. Note: will still be triggered if missing the exact data point
        # print(df.index.min())
        # print(start_date)

        assert(df.index.min() <= start_date)

        #This was a different way that was unsuccessful. leaving for reference
        # df = df.index.indexer_between_time(start_date, end_date)

        #Drop undesired currencies (that may have been carried from datascraping csv)
        for col in df.columns:
            market = 'USD-' + col[0:3] #should result in 'USD-ETH' or similar
            #drop column if necessary
            if not market in self.markets: df.drop(columns=[col], inplace = True)

        df = df.loc[df.index > start_date + timedelta(hours = 7)]
        df = df.loc[df.index < end_date + timedelta(hours = 7)]

        self.candle_df = self._format_df(df) #This completely resets the candle dataframe


    def _format_df(self, df):
        """This function formats the dataframe according to the assets that are in it. Needss to be updated to handle multiple assets.
        Note that this should only be used before high low open are stripped from the data."""
        # input_df = input_df[['Date', 'BTCClose']]
        formatted_df = df.copy()
        formatted_df = formatted_df.loc[~formatted_df.index.duplicated(keep = 'first')] #this is intended to remove duplicates. ~ flips bits

        # cols = formatted_df.columns #started writing this cause it doesnt work for mulitple currencies
        # bool_series = df['BTCClose'].notnull()
        # formatted_df = formatted_df[bool_series.values]  # Remove non datapoints from the set

        formatted_df.sort_index(inplace = True)
        # formatted_df = input_df[['Date', 'BTCOpen', 'BTCHigh', 'BTCLow', 'BTCClose', 'BTCVolume']]  #Reorder

        return formatted_df.dropna()


    def prepare_data(self):
        """This method takes the raw candle data and constructs features, changes granularity, etc."""

        print("FETCHED DATA: ")
        print(self.df.head())
        print(self.df.tail())
        # df = self._change_df_granularity(10)

        self._add_features_to_df()

        stripped_df = self.df.copy()
        df_cols = stripped_df.columns

        # Strip out open high low close
        for market in self.markets:
            token = market[4:7]
            for col in df_cols:
                if col in [token + 'Open', token + 'High', token + 'Low']:
                    stripped_df.drop(columns=[col], inplace = True)

        #This is here before the stripped_df get made to be stationary
        self.asset_data = stripped_df.copy().values #This is used in simulation and not Excahnge class
        self.df = stripped_df() #this is used in exchange class and not simulated
        print('PREPARED DATA:')
        print(stripped_df.head())
        print(stripped_df.tail())


    def _change_df_granulaty(self, gran):
        """This function looks at the Date columns of the df and modifies the df according to the input granularity (in minutes).
        This could possibly be imporved in the future with the ".resample()" method."""

        print('Changing data granularity from 1 minute to '+ str(gran) + ' minutes.')

        # if input_df.index[1] - input_df.index[0] == timedelta(minutes = 1): #verified this works
        self.candle_df =  self.candle_df.iloc[::gran, :]
        self.candle_df = self._format_df(self.candle_df)
        # print(input_df.head())
        # else:
        #     print('Granularity of df input to change_df_granularity was not 1 minute.')
        #     raise(ValueError)
        #     return input_df

    def _add_features_to_df(self, renko_block = 40):
        """ If you change the number of indicators in this function, be sure to also change the expected number in the enviroment"""

        print('Constructing features...')

        def add_sma_as_column(mydata, token, p):
            # p is a number
            price = mydata[token + 'Close'].values  # returns an np price, faster

            sma = np.empty_like(price)
            for i, item in enumerate(np.nditer(price)):
                if i == 0:
                    sma[i] = item
                elif i < p:
                    sma[i] = price[0:i].mean()
                else:
                    sma[i] = price[(i - p):i].mean()

            # subtract
            indicator = np.empty_like(sma)
            for i, item in enumerate(np.nditer(price)):
                indicator[i] = sma[i] - price[i]

            mydata['SMA_' + str(p)] = indicator  # modifies the input df


        def add_renko(blocksize, token):
            #reference for how bricks are calculated https://www.tradingview.com/wiki/Renko_Charts
            # p is a number
            prices = self.df[token + 'Close'].values  # returns an np price, faster

            renko = np.empty_like(prices)

            indicator_val = 0
            #Loop to calculate the renko value at each data point
            for i, price in enumerate(np.nditer(prices)):
                if i == 0:
                    upper_thresh = price + blocksize
                    lower_thresh = price - blocksize
                elif price <= lower_thresh: #create a down block

                    indicator_val -= blocksize #continuing a downtrend
                    lower_thresh -= blocksize
                    upper_thresh = lower_thresh + 3*blocksize

                elif price >= upper_thresh: #create an up block

                    indicator_val += blocksize #continuing an uptrend
                    upper_thresh += blocksize
                    lower_thresh = upper_thresh - 3*blocksize

                    renko[i] = indicator_val

                period = 5
                indicator = np.empty_like(prices)

            # #Loop to interpret the renko to be more useful
            # for i, item in enumerate(renko):
            #     if i == 0:
            #         indicator[i] = item
            #     elif i < period:
            #         indicator[i] = renko[0:i].mean()
            #     else:
            #         indicator[i] = renko[(i - period):i].mean() - renko[i-period]

            self.df['Renko'] = renko


        def add_sentiment(mydata):
            """This function pulls sentiment data from the crypto fear and greed index. That data is updated daily.
            This is the link to the website: https://alternative.me/crypto/fear-and-greed-index/#fng-history
            The 'limit' argument is the number of data points to fetch (one for each day).
            The given value is on a scale of 0 - 100, with 0 being extreme fear and 100 being extreme greed."""


            #Get the oldest date and figure out how long ago it was
            days = (datetime.now() - mydata.Date.min()).days + 1 #validated this is correct
            url = "https://api.alternative.me/fng/?limit="+ str(days) +"&date_format=us"

            data = requests.get(url).json()["data"] #returns a list of dictionaries

            sentiment_df = pd.DataFrame(data)

            #Drop unnecessaary columns
            sentiment_df.drop(columns = ["time_until_update", "value_classification"], inplace = True)
            #Rename the columns
            sentiment_df.rename(columns={'timestamp': 'Date', 'value': 'Value'}, inplace = True)
            #Format the dates
            sentiment_df['Date'] = pd.to_datetime(sentiment_df["Date"], format = "%m-%d-%Y")
            #Convert value to int, and center the sentiment value at 0
            sentiment_df["Value"] = sentiment_df['Value'].apply(int)
            sentiment_df['Value'] = sentiment_df['Value'] - 50

            #This should really be done with resample but I had a tough time getting it to work that way

            sentiment = np.empty_like(mydata.BTCClose)

            for i, row in mydata.iterrows():
                #Checked that both dfs have pandas timestamp date datatypes
                try:
                    sentiment[i] = sentiment_df[sentiment_df.Date == row.Date.floor(freq = 'D')].Value
                except ValueError:
                    print('Index: ' + str(i))
                    print(sentiment)

                    # print("SENTIMENT VECTOR: : ")
                    # print(sentiment)
                    mydata['Sentiment'] = sentiment

        # base = 50
        # add_sma_as_column(input_df, base)
        # add_sma_as_column(input_df, int(base*8/5))
        # add_sma_as_column(input_df, int(base*13/5))
        # add_sentiment(input_df)

        # add_renko(renko_block)

        for market in self.markets:
            token = market[4:7] #this is something like 'BTC' or 'ETH'

            self.df[token + 'OBV'] = ta.volume.on_balance_volume(self.candle_df[token + 'Close'], self.candle_df[token + 'Volume'], fillna  = True)
            self.df[token + 'OBV'] = self.df[token + 'OBV'] - self.df[token + 'OBV'].shift(1)
            self.df[token + 'MACD'] = ta.trend.macd_diff(self.candle_df[token + 'Close'], fillna  = True)
            self.df[token + 'RSI'] = ta.momentum.rsi(self.candle_df[token + 'Close'], fillna  = True) - 50


    def save_data(self):
        # This function writes the information in the original format to the csv file
        # including new datapoints that have been fetched

        print('Writing data to CSV.')

        path = paths['cum data csv']

        # datetimes to strings
        print('Fetching data from the download file...', end = ' ')
        # get the historic data. Columns are TimeStamp	Open	High	Low	Close	Volume_(BTC)	Volume_(Currency)	Weighted_Price

        def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
        cols_to_use = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(Currency)']
        orig_df = pd.read_csv(paths['downloaded csv'], usecols=cols_to_use, parse_dates = ['Timestamp'], date_parser=dateparse)

        name_map = {'Open': 'BTCOpen', 'High': 'BTCHigh', 'Low': 'BTCLow', 'Close': 'BTCClose', 'Volume_(Currency)': 'BTCVolume'}
        orig_df.rename(columns= name_map, inplace=True)
        orig_df.set_index('Timestamp', inplace = True, drop = True)
        print('Fetched.')
        df_to_save = self.df.append(orig_df, sort = True)

        #Drop features and indicators, all non savable data
        for col in df_to_save.columns:
            if not col[3::] in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df_to_save.drop(col, axis = 1, inplace = True)

        df_to_save = self._format_df(df_to_save)

        # df_to_save = filter_error_from_download_data(df_to_save)
        df_to_save.to_csv(path, index = True, index_label = 'TimeStamp', date_format = "%Y-%m-%d %I-%p-%M")

        # df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d %I-%p-%M")               # added this so it doesnt change if passed by object... might be wrong but appears to make a difference. Still dont have a great grasp on pass by obj ref.``
        print('Data written.')


    def test_data_stationarity(self):
        """This method performs an ADF test on the transformed df. Code is borrowed from
        https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
        Quote: If the test statistic is less than the critical value, we can reject the null
        hypothesis (aka the series is stationary). When the test statistic is greater
        than the critical value, we fail to reject the null hypothesis (which means
        the series is not stationary)."""

        # #These lines make the data stationary for stationarity testing
        # #log(0) = -inf. Some indicators have 0 values which causes problems w/ log
        transformed_df = self.df.copy()

        # Strip out open high low close
        for market in self.markets:
            token = market[4:7]
            for col in df_cols:
                if col in [token + 'Open', token + 'High', token + 'Low']:
                    transformed_df.drop(columns=[col], inplace = True)

        cols = transformed_df.columns
        for i in range(2*self.n_asset): #loop through prices and volumnes
            col = cols[i]
            transformed_df[col] = transformed_df[col] - transformed_df[col].shift(1)

        # transformed_df.drop(transformed_df.index[0], inplace = True)

        #this assumes that the stationary method used on the df is the same used in _get_state()
        # print("TRANSFORMED DATA: ")
        # print(transformed_df.head())
        # print(transformed_df.tail())

        #Perform Dickey-Fuller test:
        print ('Results of Dickey-Fuller Test:')
        index=['Test Statistic','p-value','#Lags Used','Number of Observations Used', 'Success']
        for col in transformed_df.columns:
            print('Results for ' + col)
            dftest = adfuller(transformed_df[col], autolag='AIC')
            success = dftest[0] < dftest[4]['1%']
            dfoutput = pd.Series([*dftest[0:4], success], index = index)
            for key,value in dftest[4].items():
               dfoutput['Critical Value (%s) conf.'%key] = value
            print (dfoutput)
            print(' ')


    def plot_agent_history(self):
        """This method plots performance of an agent over time.
        """
        print(self.log.head())

        assert not self.log.empty
        fig, (ax1, ax2) = plt.subplots(2, 1)  # Create the figure

        for market in self.markets:
            token = market[4:7]

            market_perf = ROI(self.df[token + 'Close'].iloc[0], self.df[token + 'Close'].iloc[-1])
            fig.suptitle('Market performance: ' + str(market_perf), fontsize=14, fontweight='bold')
            self.df.plot( y=token +'Close', ax=ax1)

        my_roi = ROI(self.log['Total Value'].iloc[0], self.log['Total Value'].iloc[-1])
        sharpe = my_roi/self.log['Total Value'].std()
        print(f'Sharpe Ratio: {sharpe}') #one or better is good

        self.log.plot(y='Total Value', ax=ax2)
        self.log.plot(y='BTC', ax = ax2)            # !!! not formatted to work with multiple coins
        fig.autofmt_xdate()


    def plot_market_data(self):

        # I had a ton of trouble getting the plots to look right with the dates.
        # This link was really helpful http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html
        assert not self.df.empty
        fig, (ax1, ax2) = plt.subplots(2, 1)  # Create the figure

        for market in self.markets:
            token = market[4:7]

            market_perf = ROI(self.df[token + 'Close'].iloc[0], self.df[token + 'Close'].iloc[-1])
            fig.suptitle('Market performance: ' + str(market_perf), fontsize=14, fontweight='bold')
            self.df.plot( y=token +'Close', ax=ax1)


        for col in self.df.columns:
            if not col[3:] in ['Open', 'High', 'Low', 'Close', 'Volume']:
                self.df.plot(y=col, ax=ax2)

        fig.autofmt_xdate()


    def plot_stationary_data(self):
        assert not self.transformed_df.empty
        fig, ax = plt.subplots(1, 1)  # Create the figure

        fig.suptitle('Transformed data', fontsize=14, fontweight='bold')

        for col in self.transformed_df.columns:
                self.transformed_df.plot(y=col, ax=ax)

        fig.autofmt_xdate()


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

    def __init__(self, start, end, initial_investment=100):
        ExchangeEnvironment.__init__(self)

        """The data, for asset_data can be thought of as nested arrays, where indexing the
        highest order array gives a snapshot of all data at a particular time, and information at the point
        in time can be captured by indexing that snapshot."""
        self.asset_data = None

        self.fetch_data(start, end)
        self.prepare_data()
        # n_step is number of samples, n_stock is number of assets. Assumes to datetimes are included
        self.n_step, n_features = self.asset_data.shape

        # instance attributes
        self.initial_investment = initial_investment
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
        btc_amt = self._get_btc()*self.asset_prices[0]

        if self.should_log:
            self.log = self.log.append(pd.DataFrame.from_records(
                [dict(zip(self.log.columns, [btc_amt, cur_val]))]), ignore_index=True)

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


    def _get_btc(self):
        return self.assets_owned[0]


    def _get_state(self):
        """This method returns the state, which is an observation that has been transformed to be stationary.
        Note that the state could later be expanded to be a stack of the current state and previous states
        The structure of the state WAS PREVIOUSLY [amount of each asset held, value of each asset, cash in hand, volumes, indicators]
        The structure of the state IS NOW [value of each asset, volumes, indicators]
        For reference if reconstructing the state, the data is ordered in self.asset_data as [asset prices, asset volumes, asset indicators]"""

        #assets_owned, USD, volume, indicators
        state = np.empty(self.state_dim)

        #These were previosly incorporated in the state.
        # state[0:self.n_asset] = self.last_action   #self.last_action is set in in the 'trade' method
        # state[self.n_asset] = self.USD     #self.USD is set in in the 'trade' method

        #Instituted a try catch here to help with debugging and potentially as a solution to handling invalid/inf values in log
        try:
            if self.cur_step == 0:
                stationary_slice = np.zeros(len(self.asset_data[0]))

            else:   #Make data stationary
                slice = self.asset_data[self.cur_step]
                last_slice = self.asset_data[self.cur_step - 1]

                def transform(x): return np.sign(x)*(np.absolute(x)**.5)

                # state =  transform(slice) - transform(last_slice)
                #below is full way, currently throwing errors
                # state = np.nan_to_num(np.log(slice) - np.log(last_slice))

                # print(slice[0:n])
                #BELOW IS THE OLD WAY OF DOING IT
                n = self.n_asset*2 #price and volume
                state[0:n] = slice[0:n] - last_slice[0:n] #simple differencing for price and volume
                # state[0:n] = np.log(slice[0:n]) - np.log(last_slice[0:2]) #this is price and volume

                state[n::] = slice[n::] #these are indicators. Valdiated they are preserved
                # print(state)

        except ValueError:  #Print shit out to help with debugging then throw error
            print("Error in simulated market class, _get_state method.")
            print('State: ', end = ' ')
            print(state)
            print('Slice: ', end = ' ')
            print(slice)
            # print(state)
            raise(ValueError)

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

    def __init__(self, money_to_use=10):
        ExchangeEnvironment.__init__(self)

        # instance attributes
        self.initial_investment = money_to_use

        self.asset_volumes = None


        self.print_account_health()

        end = datetime.now()
        start = end - timedelta(days = 1)
        self.fetch_data(start, end)
        self.prepare_data()

        # self.state, self.cur_val = self.reset()


    def reset(self):
        # Resets the environement to the initial state
        end = datetime.now()
        start = end - timedelta(days = 1)
        self.fetch_data(start, end)
        self.prepare_data()

        self.cancel_all_orders()
        self.get_current_prices()
        self.get_all_balances()

        # #Put all money into USD
        # if self.assets_owned[0] > 0:
        #     success = False
        #     while not success:
        #         success = self._trade(-self.assets_owned[0])

        return self._get_state(), self._get_val()

    def update(self):
        end = datetime.now()
        start = datetime.now() - timedelta(days = 1)

        env.fetch_data(start, end)
        env.prepare_data()


    def _get_state(self):
          # Returns the state (for now state, and observation are the same.
          # Note that the state could be a transformation of the observation, or
          # multiple past observations stacked.)
          state = np.empty(self.state_dim)  #assets_owned, USD

          slice_df = self.df.tail(2)                            #snapshot of df with last 2 rows

          cols = [c for c in slice_df.columns if c.lower()[3:] != 'open', 'high', 'low']
          slice_df = slice_df[cols]

          penult_row = slice_df.head(1)                         #making first row in slice_df an array
          ult_row = slice_df.tail(1)                            #last row in slice_df an array

          for i, a in enumerate(self.markets):
              ult_row[0+2*i] -= penult_row[0+2*i]                # correcting each market's close to be a delta rather than its value
              ult_row[1+2*i] -= penult_row[1+2*i]                # correcting each market's volume to be a delta

          state = ult_row.values

          return state


    def _act(self, action):
        """
        # index the action we want to perform
        # action_vec = [(desired amount of stock 1), (desired amount of stock 2), ... (desired amount of stock n)]

        action_vec = self.action_list[action] #a vectyor like [0.1, 0.5] own 0.1*val of BTC, 0.5*val of ETH etc.

        if action_vec != self.last_action:  # if attmepting to change state

        #THIS WILL NEED TO BE MORE COMPLEX IF MORE ASSETS ARE ADDED
        #Calculate the changes needed for each asset
        delta = [s_prime - s for s_prime, s in zip(action_vec, self.last_action)]
        """
        #currently set up for only bitcoin
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


                if -decimal_diff > threshhold:                          #sell if decimal_diff is sufficiently negative

                    print("Aw jeez, I've got " + str(round(decimal_diff*100,2)) + "% too much of my portfolio in " + str(currency_pair[4:]))

                    trade_amount = decimal_diff * cur_val               #amount to sell of coin in USD, formatted to be neg for _trade logic

                    self._trade(currency_pair, trade_amount)            #pass command to sell trade @ trade_amount

            for i, a in enumerate(action_vec): #for buying assets


                current_holding = (self.assets_owned[i]*self.asset_prices[i])/cur_val       #amount of coin currently held as fraction of total portfolio value, between 0 and 1
                currency_pair = self.markets[i]                         #which currency pair is being evaluated
                decimal_diff = a - current_holding                      #want minus have
                threshhold = 0.05                                       #trades only executed if difference between want and have is sufficiently high enough


                if decimal_diff > threshhold:                         #buy if decimal_diff is sufficiently positive

                    print("Oh boy, time to spend " + str(round(decimal_diff*100,2)) + "% more of my portfolio on " + str(currency_pair[4:]))

                    trade_amount = decimal_diff * cur_val               #amount to buy of coin in USD, formatted to be pos for _trade logic

                    self._trade(currency_pair, trade_amount)            #pass command to sell trade @ trade_amount


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


    def get_current_prices(self):
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
                        print('get_current_prices has failed. exiting the method...')
                        break
                    else:
                        attempts_left -= 1
                        time.sleep(1)
                        print('Retrying...')
                else: #success
                    # print(ticker['result'])
                    print('success.')
                    self.asset_prices[i] = ticker['result']['Last']
                    break


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
                    candle_df = self._process_candle_dict(candle_dict, market)
                    print("Success.")
                    self.df.append(candle_df, sort = False)
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
        """This method retrieves the account balanes for each currency including
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

            attempts_left -= 1


    def print_account_health(self):
        """This method prints out a variety of information about the account. In the future this should print a dataframe
        for cleaner formatting (if mutliple currency trading strategies are implemented)"""

        self.get_all_balances()
        self.get_current_prices()

        labels = ['Account Value', 'USD', *[x[4:7] for x in self.markets]]
        info = [self._get_val(), self.USD, *self.assets_owned]
        df = pd.Series(info, index = labels)

        print(' ')
        print('ACCOUNT INFO: ')
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
            print(result)


    def _trade(self, currency_pair, amount):
        """This method will execute a limit order trade for the 'amount' in USD passed. The limit is set at a price
        similar to the mean of the order book. THe method accepts positive or negative values in the 'amount' field.
        A positive value indicates buying, and a negative value indicates selling. VALIDATED"""

        # Note that bittrex exchange is based in GMT 8 hours ahead of CA

        self.get_current_prices()

        # Enter a trade into the market.
        #The bittrex.bittrex buy_limit method takes 4 arguments: market, amount, rate
        # Example result  {'success': True, 'message': '', 'result': {'uuid': '2641035d-4fe5-4099-9e7a-cd52067cde8a'}}

        if amount > 0:  # buy
            rate = round(self.asset_prices[0]*(1 + self.mean_spread/2), 3)
            amount_currency = round(amount/rate, 4)
            coin_index = self.markets.index(currency_pair)          #index of currency pair in market list to correlate to trade amounts
            order_entry_status = self.bittrex_obj_1_1.buy_limit(currency_pair, amount_currency, rate)
            side = 'buying'
        else:       # Sell
            rate = round(self.asset_prices[0]*(1 - self.mean_spread/2), 3)
            amount_currency = round(-amount/rate, 4)
            order_entry_status = self.bittrex_obj_1_1.sell_limit(currency_pair, amount_currency, rate)
            side = 'selling'

        # Check that an order was entered
        if not order_entry_status['success']:
            print('Trade attempt for ' + side + ' failed: ', end = ' ')
            print(order_entry_status['message'])
            if order_entry_status['message'] == 'INSUFFICIENT_FUNDS':
                print('Amount: ' + str(amount))
            return False
        else: #order has been successfully entered to the exchange
            print(f'Order for {side} {amount_currency:.8f} {currency_pair[4:]} at a price of ${rate:.2f} has been submitted to the market.')
            uuid = order_entry_status['result']['uuid']

            # Loop for a time to see if the order has been filled
            order_is_filled = self._monitor_order_status(uuid) #True if order is filled


            if order_is_filled == True:
                print(f'Order has been filled. uuid: {uuid}.')
                self.get_all_balances()        #updating with new amount of coin
                self.print_account_health()
            #this saves the information regardless of if the trade was successful or not
            self._get_and_save_order_data(uuid)

            return order_is_filled


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
        path = paths['order log']

        dict = self.bittrex_obj_1_1.get_order(uuid)

        # in order to construct a df, the values of the dict cannot be scalars, must be lists, so convert to lists
        results = {}
        for key in dict['result']:
            results[key] = [dict['result'][key]]
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

        # #Load the trade log from csv
        #Note that the dateformat for this is different than for the price history.
        #The format is the same in csv as the bittrex API returns for order data
        def dateparse(x): return pd.datetime.strptime(x, date_format)
        df = pd.read_csv(path, parse_dates = ['Opened', 'Closed'], date_parser=dateparse)
        df.set_index('Opened', inplace = True, drop = True)
        df = df.append(order_df, sort = False)

        # Format and save the df
        df.sort_index(inplace = True)
        df.to_csv(path, index = True, index_label = 'Opened', date_format = date_format)
        print('Order log csv has been updated.')


    def view_order_data(self):

        def dateparse(x): return pd.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")
        df = pd.read_csv(paths['order log'], parse_dates = ['Opened', 'Closed'], date_parser=dateparse)
        df.set_index('Opened', inplace = True, drop = True)
        print(' ')
        print('ALL ORDER INFORMATION:')
        print(df)


    def get_and_save_order_history(self):
        """FOR NOW I AM LEAVING THIS INCOMPLETE. THE GET_ORDER METHOD RETRIEVE MORE INFORMATION ON
        EACH ORDER, AND IS A MORE COMPLETE METHOD OF LOGGING ORDER INFO.
        This method retrieves trade history for all currency pairs from the exchange, creates a dataframe with the orders,
        and then appends them to the CSV trade log. Trade history is stored locally since the bittrex API only returns trades
        that happened within a recent timeframe. I am not sure what that time frame is."""

        print('Fetching all recent order data...', end = ' ')
        date_format = "%Y-%m-%dT%H:%M:%S"
        path = paths['order log']

        #This section fetches order data from the exchange for each relevant currency pair.
        #This will eventually need to be updated to accomodate for inter asset trading

        for i, currency_pair in enumerate(self.markets):
            order_history_dict = self.bittrex_obj_1_1.get_order_history(market = currency_pair)
            order_df = pd.DataFrame(order_history_dict['result'])
            order_df.drop(columns=['IsConditional', 'Condition', 'ConditionTarget',
                                   'ImmediateOrCancel', 'Closed'], inplace=True)
            print(order_df.columns)
            order_df.set_index('TimeStamp', drop = True, inplace = True)

            # dates into datetimes
            order_df.TimeStamp = pd.to_datetime(order_df.TimeStamp, format=date_format)
            # order_df.Opened = pd.to_datetime(order_df.Opened, format=date_format)

            #Create or append to the df
            if i == 0: df = order_df
            else: df = df.append(order_df, sort = False)

        print('fetched.')
        #This section reads in the order log from csv, and appends any new data
        try:
            def dateparse(x):
                try:
                    return pd.datetime.strptime(x, date_format)
                except ValueError:  #handles cases for incomplete trades where 'Closed' is NaT
                    return x

            # old_df = pd.read_csv(path, parse_dates = ['Opened', 'Closed'], date_parser=dateparse)
            # old_df.set_index('Timestamp', inplace = True, drop = True)
            # print(old_df)
            # old_df = old_df.append(df, sort = False)
            #
            # old_df.sort_values(by='Opened', inplace=True)
            #
            # old_df.to_csv(path, index = True, index_label = 'Opened', date_format = date_format)

        except KeyError:
            print('Order log is empty.')
        print('Data written to test trade log.')
        print(old_df)
