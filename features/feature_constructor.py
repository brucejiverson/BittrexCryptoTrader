import numpy as np
import pandas as pd
import ta
from features.classifiers import knn_classifier, mlp_classifier, svc_classifier
from features.probability import statisticsBoy
from datetime import datetime
import requests
from tools.tools import percent_change_column

def add_EMA_as_column(token, p, df):
    # p is a number
    price = df[token + 'Close'].values  # returns an np price, faster

    ema = np.empty_like(price)
    k = 2/(p - 1)
    for i, item in enumerate(np.nditer(price)):
        if i == 0:
            ema[i] = item
        elif i < p: #EMA
            ema[i] = price[0:i].mean()
        else:
            ema[i] = price[i]*k + ema[i-1]*(1- k) #price[(i - p):i].mean()

    col_name = 'EMA_' + str(p)
    df[col_name] = ema  # modifies the input df
    df[col_name] = df[col_name] - df[token + 'Close']


def add_renko(blocksize, token, input_df):
    #reference for how bricks are calculated https://www.tradingview.com/wiki/Renko_Charts
    # p is a number
    df = input_df.copy()
    prices = input_df[token + 'Close'].values  # returns an np price, faster

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
    
    df['Renko'] = renko
    return df


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


def sign_mapping(input_df):
    df = input_df.copy()
    # Loop to feature map
    for col in df.columns:
        if col[3:7] not in ['Open', 'High', 'Low']:
            new_name = col + '-sign'
            df[new_name] = 0
            df[new_name].loc[df[col] - df[col].shift(1) > 0] = 1
    
    return df


def time_of_day(input_df):
    df = input_df.copy()
    "Adds the time of day as a column"
    df['Time of Day'] = [date.hour for date in df.index]
    return df


def stack(n, df, mode='Close'):

    if mode == 'Close':
        # This function stacks the states so that both the current state and previous n states are passed.
        new_df = df.copy()
        for i in range(1, n+1):
            shifted_df = df.shift(i)
            # remove open high low from shift
            for col in shifted_df.columns:
                if not col == 'BTCClose':
                    shifted_df.drop(col, axis=1, inplace=True)
            new_cols = [c + "_shift_" + str(i) for c in shifted_df.columns]
            shifted_df.columns = new_cols
            new_df = pd.concat([new_df, shifted_df], axis=1, sort=True)
        return new_df
        
    else:
        # This function stacks the states so that both the current state and previous n states are passed.
        new_df = df.copy()
        for i in range(1, n+1):
            shifted_df = df.shift(i)
            # remove open high low from shift
            for col in shifted_df.columns:
                if col[3:] in ['Open', 'High', 'Low']:
                    shifted_df.drop(col, axis=1, inplace=True)
            new_cols = [c + "_shift_" + str(i) for c in shifted_df.columns]
            shifted_df.columns = new_cols
            new_df = pd.concat([new_df, shifted_df], axis=1, sort=True)
        return new_df


def build_features(testing_df, markets, feature_dict, training_df=None):
        """This method takes the raw candle data and constructs features based on the provided list of strings"""
        
        print('Constructing features... ', end = ' ')
        
        if feature_dict is None:
            print('no features built.')
            return candle_df
        test_df = testing_df.copy()       # At this higher level, it is safer to construct a new dataframe instead of modifying the one that gets passed.
        train_df = training_df.copy()
        #This section constructs engineered features and adds them as columns to the df
        """ If you change the number of indicators in this function, be sure to also change the expected number in the enviroment"""

        # Check for illegal feature names
        acceptable_feature_names = ['sign', 'high', 'low',
                                    'OBV', 'EMA', 'MACD', 'RSI', 'BollingerBands', 'BBInd', 'BBWidth',
                                    'sentiment', 'renko', 'time of day', 'discrete_derivative',
                                    'knn', 'mlp', 'svc', 'ridge', 'stack',
                                    'probability']
        for f in feature_dict:           # Iterate over the keys
            if not f in acceptable_feature_names:
                print('An unrecognized feature has been passed to the feature contructor (not in the acceptable feature list).')
                print(f'Illegal feature: {f}')
                raise(ValueError)
        print(feature_dict)
        #Add the features to the df. It would be great to have a dict of features and functions but im not sure how to do that
        for df in [test_df, train_df]:
            for market in markets:
                token = market[4:7] #this is something like 'BTC' or 'ETH'
                for feature in feature_dict:               # Iterate over the keys
                    if feature == 'high':
                        df[token+'HighInd'] = df[token+'High'] - df[token+'Open']
                    elif feature == 'low':
                        df[token+'LowInd'] = df[token+'Low'] - df[token+'Open']
                    elif feature == 'sign':
                        df = sign_mapping(df)
                    elif feature == 'OBV':
                        df[token + 'OBV'] = ta.volume.on_balance_volume(df[token + 'Close'], df[token + 'Volume'], fillna  = True)
                        df[token + 'OBV'] = df[token + 'OBV'] - df[token + 'OBV'].shift(1, fill_value=0)
                    elif feature == 'MACD':
                        df[token + 'MACD'] = ta.trend.macd_diff(df[token + 'Close'], fillna  = True)
                    elif feature == 'RSI':
                        df[token + 'RSI'] = ta.momentum.rsi(df[token + 'Close'], fillna  = True)
                    elif feature == 'EMA':
                        for base in feature_dict[feature]:
                            add_EMA_as_column(token, base, df)
                        # add_EMA_as_column(token, int(base*8/3))
                        # add_EMA_as_column(token, int(base*13/3))
                    elif feature == 'BollingerBands':
                        bb_drop = True
                        for base in feature_dict[feature]:
                            if feature == 'do not drop':
                                bb_drop = False
                                continue
                            temp_df = df.copy().iloc[::base, :]
                            
                            # Initialize Bollinger Bands Indicator
                            indicator_bb = ta.volatility.BollingerBands(close=temp_df[token+"Close"], n=20, ndev=2)

                            # Add Bollinger Bands features
                            b = str(base)
                            temp_df['bb_bbm' + b] = indicator_bb.bollinger_mavg()
                            temp_df['bb_bbh' + b] = indicator_bb.bollinger_hband()
                            temp_df['bb_bbl' + b] = indicator_bb.bollinger_lband()

                            # Interpolate the more sparse temp_df into the full df
                            try:
                                for col in ['bb_bbm' + b, 'bb_bbh' + b, 'bb_bbl' + b,]:
                                    transformed_col = []
                                    for i, val in enumerate(temp_df[col].values):
                                        try:
                                            next_val = temp_df[col].values[i+1]
                                        except IndexError:
                                            next_val = val
                                        for j in range(base):
                                            interpolated = j*(next_val - val)/base + val
                                            transformed_col.append(interpolated)
                                    # remove items until lengths match (usually 0 or 1 items)
                                    while len(transformed_col) > df.shape[0]:
                                        transformed_col.pop(0)
                                    df[col] = transformed_col
                            except ValueError:
                                print(len(transformed_col))
                                print(df.shape)
                                print(col)
                                print(temp_df.head())
                    elif feature == 'BBInd':
                        for base in feature_dict['BollingerBands']:
                            b = str(base)
                            # The fractional position of the price relative to the high and low bands
                            diff = df['BTCClose'] - df['bb_bbl' + b]
                            df['BBInd'+ b] = diff/(df['bb_bbh' + b] - df['bb_bbl' + b]) - 0.5
                    elif feature == 'BBWidth':
                        for base in feature_dict['BollingerBands']:
                            b = str(base)
                            df['BBWidth'+ b] = (df['bb_bbh' + b] - df['bb_bbl' + b])
                    # discrete_derivative('EMA_' + str(base))
                    # discrete_derivative('EMA_' + str(int(base*8/3)))
                    # discrete_derivative('EMA_' + str(int(base*13/3)))
                    elif feature == 'discrete_derivative':
                        for col_name in feature_dict['discrete_derivative']:
                            temp = percent_change_column(col_name, df)
                            df['ddt_' + str(col_name)] = temp[col_name]
                    elif feature == 'sentiment':
                        add_sentiment(df)
                    elif feature == 'renko':
                        block = 50
                        add_renko(block, token)
                    elif feature == 'time of day':
                        df = time_of_day(df)
                    elif feature == 'stack':
                        n = feature_dict[feature][0]
                        df = stack(n, df)
                    elif feature == 'knn':
                        hyperparams = {'polynomial_features__degree': (1, 2, 3, 4), 
                        'knn__n_neighbors': (2, 3, 4, 5, 6, 7, 8, 9)}
                        knn = knn_classifier(token, hyperparams, n_predictions=5)
                        knn.load()
                        # knn.optimize_parameters(df)
                        # knn.train(df)
                        df = knn.build_feature(df)
                    elif feature == 'mlp':
                        hyperparams = {'polynomial_features__degree': (1, 2, 3, 4, 5)}
                        mlp = mlp_classifier(token, hyperparams)
                        mlp.load()
                        # mlp.optimize_parameters(df)
                        # mlp.train(df)
                        df = mlp.build_feature(df)
                    elif feature == 'svc':
                        hyperparams = {'polynomial_features__degree': (1, 2, 3, 4, 5),
                                        'svc__penalty': ['l1', 'l2'],
                                        'svc__C': [.25, .5, .75, 1],
                                        'svc__max_iter': [50000]}    
                        svc = svc_classifier(token, hyperparams)
                        svc.load()
                        # svc.optimize_parameters(df)
                        # svc.train(df)
                        df = svc.build_feature(df)
                    elif feature == 'probability':
                        pass
                        
                    else:
                        print(f'Feature: {feature} not found.')

                if bb_drop == True:
                    for feature in feature_dict['BollingerBands']:
                        names = [n + str(feature) for n in ['bb_bbl', 'bb_bbm', 'bb_bbh']]
                        try:
                            df.drop(columns=names, inplace=True)
                        except KeyError:
                            print(f'Error dropping column {names}')
                            print(df.columns)
                            # print(df.loc[names].iloc[0])
            
        # separated out for easy looping over both dicts with the other functions. Useful for lazy learners
        if 'probability' in feature_dict.keys():
            assert(not train_df is None)        # Need training data to do this properly
            # Feature names are passed to the constructor
            feat1_name, feat2_name = feature_dict['probability']
            my_stats = statisticsBoy(target_change=0.1, allowable_reversal=-.5)
            temp_df = my_stats.make_criteria(train_df, target_time_range=13)
            my_stats.do_analysis(temp_df, [feat1_name, feat2_name])
            test_df = my_stats.build_feature(test_df, [feat1_name, feat2_name])

        print('done.')

        df_cols = test_df.columns

        # Strip out open high low close
        for market in markets:
            token = market[4:7]
            for col in df_cols:
                if col in [token + 'Open', token + 'High', token + 'Low']:
                    df.drop(columns=[col], inplace = True)

        return test_df