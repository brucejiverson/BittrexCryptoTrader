from tools.tools import f_paths
from bittrex.bittrex import *
from tools.tools import ROI, printProgressBar, percent_change_column
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import logging
import mplfinance as mpf


class ExchangeEnvironment(object):
    """All other environment classes inherit from this class. This is done to ensure similar architecture
    between different classes (eg simulated vs real exchange), and to make it easy to change that
    architecture by having change be in a single place."""

    def __init__(self, gran, feature_dict):

        #get my keys
        with open(f_paths['secret']) as secrets_file:
            keys = json.load(secrets_file) #loads the keys as a dictionary with 'key' and 'secret'
            secrets_file.close()

        #Need both versions of the interface as they each provide certain useful functions
        # try:
        self.bittrex_obj_1_1 = Bittrex(keys["key"], keys["secret"], api_version=API_V1_1)
        self.bittrex_obj_2 = Bittrex(keys["key"], keys["secret"], api_version=API_V2_0)
        # except 
        self.markets = ['USD-BTC']#, 'USD-ETH', 'USD-LTC']    #Alphabetical
        self.n_asset = len(self.markets)
        self.granularity = gran # minutes

        # Calculate the number of indicators for later state dimension construction
        self.n_indicators = 0
        stack = 0
        if not feature_dict is None:
            for feature, vals in feature_dict.items():      # iterate over the keys
                if feature == 'stack':
                    stack = vals[0]                         # Store this for use with state_dim
                elif feature == 'BollingerBands':
                    self.n_indicators += 3
                elif len(vals) == 0:
                    self.n_indicators += 1
                else:
                    self.n_indicators += len(vals)
                # printing for debugging
                # print(f'Feature: {feature}, vals: {vals}, n: {self.n_indicators}')

        portfolio_granularity = 1  # Smallest fraction of portfolio for investment in single asset (.01 to 1)
        # The possible portions of the portfolio that could be allocated to a single asset
        possible_vals = [x / 100 for x in list(range(0, 101, int(portfolio_granularity * 100)))]

        # calculate all possible allocations of wealth across the available assets
        self.action_list = []
        permutations_list = list(map(list, itertools.product(possible_vals, repeat=self.n_asset)))
        #Only include values that are possible (can't have more than 100% of the portfolio)
        for item in permutations_list:
            if sum(item) <= 1:
                self.action_list.append(item)

        #This list is for indexing each of the actions
        self.action_space = np.arange(len(self.action_list))

        #BELOW IS THE OLD WAY THAT INCLUDES THE CURRENT ASSETS HELD
        # calculate size of state (amount of each asset held, value of each asset, volumes, USD/cash, indicators for each asset)
        self.state_dim = self.n_asset*2 + self.n_asset*self.n_indicators #price and volume, and then the indicators
        if stack != 0:
            self.state_dim *= (stack + 1)
        self.last_action = [] #The action where nothing is owned

        self.assets_owned = [0]*self.n_asset
        self.asset_prices = [0]*self.n_asset

        self.USD = None
        self.candle_df_1min = None
        self.candle_df = None
        self.df = None
        self.transformed_df = None
        self.asset_data = None
        self.should_log = False
        self.log = AccountLog()


    def _parse_candle_dict(self, candle_dictionary, market):
        # Dataframe formatted the same as in other functions
        # V and BV refer to volume and base volume
        ticker = market[4:7]
        df = pd.DataFrame(candle_dictionary['result'])
        df['T'] = pd.to_datetime(df['T'], format="%Y-%m-%dT%H:%M:%S")
        df = df.rename(columns={'T': 'TimeStamp', 'O': ticker + 'Open', 'H': ticker +'High', 'L': ticker +'Low', 'C': ticker +'Close', 'V': ticker +'Volume'})
        df.set_index('TimeStamp', drop = True, inplace = True)
        df.drop(columns=["BV"], inplace=True)

        #Reorder the columns
        df = df[[ticker +'Open', ticker +'High', ticker +'Low', ticker +'Close', ticker +'Volume']]
        # print(df.head())
        return df


    def _fetch_candle_data(self, start_date, end_date):
        """This function pulls data from the exchange, the cumulative repository, and the data download in that order
        depending on the input date range. Typically called in the __init__ method"""

        print('Fetching historical data...')

        """algorithm pseudocode:
        scrape the latest data
        try:
            load to the 1 min binary file
        except
            load from csv
        save to 1 min file
        if gran  1
            filter date range
            return 1 min data
        try
            find the full daterange of binary gran file
            get the 1 min data outside of the binary gran file daterange
            change the granularity on the 1 min data
        except FileNotFound
            convert all of the 1 min data into gran 
        append to binary gran file
        save
        filter date range
        return
        """

        def fetch_csv():
            try:
                df = pd.DataFrame()
                # Read the original download file
                # get the historic data. Columns are TimeStamp	Open	High	Low	Close	Volume_(BTC)	Volume_(Currency)	Weighted_Price
                def dateparse(x): return pd.Timestamp.fromtimestamp(int(x))
                cols_to_use = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(Currency)']
                orig_df = pd.read_csv(f_paths['downloaded csv'], 
                                                usecols=cols_to_use, 
                                                parse_dates=['Timestamp'], 
                                                date_parser=dateparse)
                name_map = {'Open': 'BTCOpen', 'High': 'BTCHigh', 'Low': 'BTCLow',
                    'Close': 'BTCClose', 'Volume_(Currency)': 'BTCVolume'}
                orig_df.rename(columns=name_map, inplace=True)
                orig_df.set_index('Timestamp', inplace=True, drop=True)

                assert not orig_df.empty

                df = df.append(orig_df, sort=True)
                # apparently this is necessary otherwise the orig_df is in reverse order
                df.sort_index(inplace=True)
                return df
            except FileNotFoundError:
                print('no csv file found. Please download the csv historical data file from kaggle.')
                raise(FileNotFoundError)

        df_1_min = pd.DataFrame()
        scraped_df = pd.DataFrame()
        # Fetch candle data from bittrex for each market
        if end_date > datetime.now() - timedelta(days=9):
            for i, market in enumerate(self.markets):
                print('Fetching ' + market + ' historical data from the exchange.')
                attempts = 0
                while True:
                    print('Fetching candles from Bittrex... ', end = " ")
                    candle_dict = self.bittrex_obj_2.get_candles(market, 'oneMin')

                    if candle_dict['success']:
                        # print(candle_dict)
                        candle_df = self._parse_candle_dict(candle_dict, market)
                        print("done.")
                        # print(candle_df.head())
                        break
                    # If there is an error getting the proper data
                    else:
                        print("Failed to get candle data. Candle dict: ", end = ' ')
                        print(candle_dict)
                        time.sleep(2*attempts)
                        attempts += 1

                        if attempts == 5:
                            print('Exceeded maximum number of attempts.')
                            raise(TypeError)
                        print('Retrying...')
                # Handle joining data from multiple currencies
                if i == 0: scraped_df = scraped_df.append(candle_df, sort = True)
                else: scraped_df = pd.concat([scraped_df, candle_df], axis = 1)
                # print('CANDLE DF:')
                # print(candle_df.tail())
                # df_1_min = df_1_min.append(candle_df, sort = True) #ok this works
                # print(df_1_min.tail())

        df_1_min = df_1_min.append(scraped_df, sort=True)
        df_1_min = self._format_df(df_1_min)
        # Update the binary file
        print('Fetching from cum. data repository... ', end='')
        try:
            cum_df = pd.read_pickle(f_paths['cum data pickle']+'1.pkl')
            df_1_min = df_1_min.append(cum_df, sort=True)
            print('binary file 1 min. gran. loaded... ', end='')
        except FileNotFoundError:
            print('Could not find a binary 1 minute granularity file.')
            cum_df = fetch_csv()
            print(cum_df.head())
            # print('jhere')
            print('csv file 1 min. gran. loaded... ', end='')
        
        df_1_min = df_1_min.append(cum_df, sort=True)
        df_1_min = self._format_df(df_1_min)
        df_1_min.to_pickle(f_paths['cum data pickle'] + '1.pkl')
        print(' written to file.')
        self.candle_df_1min = df_1_min
        print('1 MINUTE DATA:')
        print(df_1_min.head())
        print(df_1_min.tail())
        
        assert(df_1_min.index.min() <= start_date)

        # Drop undesired currencies
        for col in df_1_min.columns:
            market = 'USD-' + col[0:3]  # should result in 'USD-ETH' or similar
            # Drop column if necessary
            if not market in self.markets:
                df_1_min.drop(columns=[col], inplace=True)
        
        df_gran = pd.DataFrame()                         # This is the dataframe that will
        gran = self.granularity
        if gran == 1:
            df_1_min = df_1_min.loc[df_1_min.index >
                                    start_date + timedelta(hours=7)]
            df_1_min = df_1_min.loc[df_1_min.index < end_date + timedelta(hours=7)]
            self.candle_df =  self._format_df(df_1_min)
            return
        # Change the granularity of the data
        gran_path = f_paths['cum data pickle']+str(gran)+'.pkl'
        try:
            read_df_gran = pd.read_pickle(gran_path)
            df_gran = df_gran.append(read_df_gran, sort=True)
            df_gran = self._format_df(df_gran)
            # find the full daterange of binary gran file
            data_end = df_gran.index[-1]

            # Get the 1 minute data that is outside of the daterange
            df_to_filter = df_1_min.loc[(df_1_min.index >= data_end - timedelta(minutes=gran+1))]

        # If file not found, convert the 1 minute df_gran into the desired granularity
        except FileNotFoundError:
            print('Changing full 1 minute dataset granularity to ' +
                    str(gran) + ' minutes.')
            
            df_to_filter = df_1_min.copy()

        filtered = self._change_granularity(df_to_filter)
        df_gran = df_gran.append(filtered, sort=True)
        print('')
        df_gran = self._format_df(df_gran)
        df_gran.to_pickle(gran_path)

        time_shift_from_bittrex = timedelta(hours=7)

        # Filter df_gran
        df_gran = df_gran.loc[df_gran.index > start_date + time_shift_from_bittrex]
        df_gran = df_gran.loc[df_gran.index < end_date + time_shift_from_bittrex]

        self.candle_df = df_gran # Completely resets the candle_df
        # return


    def _format_df(self, df):
        """This function formats the dataframe according to the assets that are in it.
        Needs to be updated to handle multiple assets. Note that this should only be used before high low open are stripped from the data."""
        # input_df = input_df[['Date', 'BTCClose']]
        formatted_df = df.copy()
        formatted_df = formatted_df.loc[~formatted_df.index.duplicated(keep = 'first')]     # this is intended to remove duplicates. ~ flips bits in the mask
        # formatted_df = formatted_df[~formatted_df.isin([np.nan, np.inf, -np.inf]).any(1)]

        # cols = formatted_df.columns # Started writing this cause it doesnt work for mulitple currencies
        # bool_series = df['BTCClose'].notnull()
        # formatted_df = formatted_df[bool_series.values]  # Remove non datapoints from the set

        formatted_df.sort_index(inplace = True)
        # formatted_df = input_df[['Date', 'BTCOpen', 'BTCHigh', 'BTCLow', 'BTCClose', 'BTCVolume']]  #Reorder

        return formatted_df.dropna()


    def _change_granularity(self, input_df):
        """This function looks at the Date columns of the df and modifies the df according to the granularity (in minutes).
        This function expects self.granularity to be an positive int <= 60*24"""
        df = input_df.copy()
        gran = self.granularity
        if gran == 1:
            print("Granularity is set to 1 minute.")
            return df

        new_df = pd.DataFrame(columns = df.columns)
        start = df.index[0]
        # Get the starting minute as a multiple
        m = start.minute
        start += timedelta(minutes=(gran - m + 1))
        
        oldest = max(df.index)

        #Loop over the entire dataframe. assumption is that candle df is in 1 min intervals
        length = df.shape[0]
        i = 0
        while True:
            if i > 100 and i%20 ==0:
                printProgressBar(i, length, prefix='Progress:', suffix='Complete')

            end = start + timedelta(minutes=gran-1)
            data = df.loc[(df.index >= start) & (df.index <= end)]
            
            try:   
                # Note that timestamps are the close time
                candle = pd.DataFrame({'BTCOpen': data.iloc[0]['BTCOpen'],
                                        'BTCHigh': max(data['BTCHigh']),
                                        'BTCLow': min(data['BTCLow']),
                                        'BTCClose': data.iloc[-1]['BTCClose'],
                                        'BTCVolume': sum(data['BTCVolume'])},
                                        index=[end])
                new_df = new_df.append(candle)
            # Handle empty slices (ignore)
            except IndexError:
                pass
            if end >= oldest: break
            start += timedelta(minutes=gran)
            # This is for printing the progress bar
            try:
                i = df.index.get_loc(start)
            except KeyError:
                pass
        
        # print('')
        # print('Dataframe with updated granularity:')
        # print(new_df.head())
        return new_df


    def test_data_stationarity(self):
        """This method performs an ADF test on the transformed df. Code is borrowed from
        https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
        Quote: If the test statistic is less than the critical value, we can reject the null
        hypothesis (aka the series is stationary). When the test statistic is greater
        than the critical value, we fail to reject the null hypothesis (which means
        the series is not stationary)."""

        # # These lines make the data stationary for stationarity testing
        # # log(0) = -inf. Some indicators have 0 values which causes problems w/ log
        transformed_df = self.df.copy()

        cols = transformed_df.columns
        for i in range(2*self.n_asset): #loop through prices and volumnes
            col = cols[i]
            transformed_df[col] = transformed_df[col] - transformed_df[col].shift(1, fillna=0)

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


        fig, ax = plt.subplots(1, 1)  # Create the figure
        fig.suptitle('Transformed data', fontsize=14, fontweight='bold')

        for col in transformed_df.columns:
                transformed_df.plot(y=col, ax=ax)

        fig.autofmt_xdate()


    def plot_market_data(self):

        # I had a ton of trouble getting the plots to look right with the dates.
        # This link was really helpful http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html (no longer using this)
        assert not self.df.empty

        # Find out if there are any features. This way is easier/more truthful than referencing feature list
        is_features = False
        for col in self.df.columns:
            if not col[3:] in ['Open', 'High', 'Low', 'Close', 'Volume']:
                is_features = True
                break

        #THIS IS OLD
        # if is_features:
        #     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # Create the figure
        #     candle_ax = ax1
        #     # plot the features
        #     for col in self.df.columns:
        #         if not col[3:] in ['Open', 'High', 'Low', 'Close', 'Volume']:
        #             self.df.plot(y=col, ax=ax2)

        # # make a single subplot is there are no features
        # else:
        #     fig, candle_ax = plt.subplots(1, 1, sharex=True)  # Create the figure

        for market in self.markets:
            token = market[4:7]

            # market_perf = ROI(self.df[token + 'Close'].iloc[0], self.df[token + 'Close'].iloc[-1])
            # fig.suptitle(f'Market performance: {market_perf}%', fontsize=14, fontweight='bold')
            # SEE THIS https://github.com/matplotlib/mplfinance#usage
            columns =  {token + 'Open': 'Open', 
                        token + 'High': 'High', 
                        token + 'Low': 'Low', 
                        token + 'Close': 'Close',
                        token + 'Volume': 'Volume'}            
            
            df = self.candle_df.copy()
            df.index.name = 'Date'
            df = df.rename(columns, axis=1)
            try:
                mpf.plot(df, type='candle', volume=True, show_nontrading=True, 
                style='yahoo', title='BTC Candle Data', ylabel='OHLC Candles')
            except KeyError:
                print('Error plotting (see parent exchange class) :(')
                print(df.head())


class AccountLog:
    
    def __init__(self):
        log_columns = ['$ of BTC', 'Total Value']
        # self.data = pd.DataFrame(columns=log_columns)
        self.data = []  # np.empty([0,0])
        self.df = None


    def update(self, info):
        self.data.append(list(info.values()))   # Append to numpy array
    

    def to_dataframe(self):
        log_columns = ['$ of BTC', 'Total Value', 'Actions', 'Timestamp']
        df = pd.DataFrame(columns=log_columns, data=self.data)
        df.set_index('Timestamp', inplace=True, drop=True)
        self.df = df


    def save(self):
        """This method append to the log to the binary file.
        Saves to the live log. Sim logging saves are not currently supported"""

        path = f_paths['live log']
        self.to_dataframe()
        df = self.df
        try:
            df = pd.read_pickle(path)
            df = df.append(self.data, sort = True)
        except pd.errors.EmptyDataError:
            print('There was no data in the log. Saving data generated during this run... ', end = ' ')
            maybe_make_dir(path[:-21])
            df = self.data
        df.to_pickle(path)
        print('done.')


    def plot(self, df, name=''):
        """This method plots performance of an agent over time.
        """
        self.to_dataframe()

        # Get the buys and the sells
        history = self.df.copy()
        history['delta'] = history['Actions'] - history['Actions'].shift(1)  # Converts the column to % change
        history['delta'].fillna(0)

        history = percent_change_column('Total Value', history)
        history['BTCClose'] = df['BTCClose']
        
        buys = history[history['delta'] > 0]
        sells = history[history['delta'] < 0]
        
        print(f'Number of buys: {buys.shape[0]}.')
        print(f'Number of sells: {sells.shape[0]}.')

        assert not df.empty
        # fig, ax2 = plt.subplots(1, 1)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # Create the figure
        # for market in self.markets:
        token = 'BTC'   # market[4:7]

        market_perf = ROI(df[token + 'Close'].iloc[0], df[token + 'Close'].iloc[-1])
        # fig.suptitle(f'Market performance: {market_perf}%', fontsize=14, fontweight='bold')
        df.plot(y=token +'Close', ax=ax1)
        
        buys.reset_index().plot(y = 'BTCClose', x='Timestamp', ax=ax1, kind='scatter', marker='^', c='g', zorder=4)
        sells.reset_index().plot(y = 'BTCClose', x='Timestamp', ax=ax1, kind='scatter', marker='v', c='r', zorder=4)
        fig.autofmt_xdate()
        fig.suptitle(f'Performance for agent name: {name}', fontsize=14, fontweight='bold')
        self.df.reset_index().plot(x='Timestamp', y='Total Value', ax=ax2)
        # self.df.reset_index().plot(x='Timestamp',y='$ of BTC', ax = ax3)            # !!! not formatted to work with multiple coins
        # df.plot(y='BBInd5', ax=ax4)
        fig.autofmt_xdate()