import numpy as np
import pandas as pd
import ta
from features.classifiers import knn_classifier, mlp_classifier, svc_classifier
from features.label_data import make_criteria
from features.probability import statisticsBoy
from datetime import datetime, timedelta
import requests
from tools.tools import percent_change_column


def divide(name1, name2, input_df):
    print('here in divide')
    df = input_df.copy()
    df[name1 + '/' + name2] = np.where(df[name2] < 10**(-5), np.ones(df[name1].shape), df[name1]/df[name2] - 1) 
    return df

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


def get_probability(train_df, names):
    # Feature names are passed to the constructor
    my_stats = statisticsBoy()
    train_df = make_criteria(train_df, target_time_range=13)
    train_df = make_criteria(train_df, target_time_range=13, buy_not_sell=False)
    my_stats.do_analysis(train_df, names)        # Statistical analysis on the training data
    return my_stats


def probability_predict(my_stats, test_df, names):
    return my_stats.build_feature(test_df, names)      # Making predictions on the testing data


def rolling_probability(testing_df, training_df, names):

    # Create copies as we are going to alter and return the frames
    train_df = training_df.copy()
    test_df = testing_df.copy()

    # Print out some information about the dates (convert this to logging later)
    # print('')
    # print(f'train df start date: {train_df.index.min()}')
    # print(f'train df end date:   {train_df.index.max()}')
    # print(f'test df start date:  {test_df.index.min()}')
    # print(f'test df end date:    {test_df.index.max()}')

    # Feature names are passed to the constructor
    window_size = timedelta(days=20)
    refresh_rate = timedelta(days=14)
    my_stats = statisticsBoy()
    my_stats = statisticsBoy()

    new_df = pd.DataFrame()
    # get the first training window from the training df
    prediction_start_date = test_df.index.min()
    prediction_end_date = prediction_start_date + refresh_rate
    training_start_date = test_df.index.min() - window_size
    training_window = train_df.loc[
                                (train_df.index > training_start_date) & 
                                (train_df.index <=prediction_start_date)]

    # Initialize the dates (for loop conditions check)
    first_done = False        # This ensures loop will run at least once
    while prediction_end_date <= test_df.index.max() or not first_done:

        # get the training window if its not the first time, else it is preinitialized
        if first_done:
            # Shift the dates
            prediction_start_date = training_start_date + window_size
            prediction_end_date = prediction_start_date + refresh_rate
            # Make the training window
            training_window = test_df.loc[(test_df.index > training_start_date)&
                                        (test_df.index <= prediction_start_date)]
        else: #not first_done:
            first_done = True
        
        # get the prediction window
        prediction_window = test_df.loc[
                                    (test_df.index > prediction_start_date) & 
                                    (test_df.index <= prediction_end_date)].copy()

        # print('loop')
        # # print(prediction_end_date < test_df.index.max())
        # print(f'Training start date:   {training_start_date}')
        # print(f'Prediction start date: {prediction_start_date}')
        # print(f'Prediction end date    {prediction_end_date}')
        # # print(f'Test max date          {test_df.index.max()}')
        # # print(f'First done:            {first_done}')

        print(prediction_window)

        # Train
        # print('Training window:')
        # print(training_window.head())
        my_stats.do_analysis(training_window, names)                                    # Statistical analysis on the training data

        # Build 
        built_df = my_stats.build_feature(prediction_window, names)
        # print('Build df:')
        # print(built_df.head())
        new_df = new_df.append(built_df)                                                # Making predictions on the testing data
        training_start_date += refresh_rate
        
    return new_df.copy(), train_df.copy()


def make_knn(df):
    pass


def make_mlp():
    pass


def make_logistic_regression():
    pass


class featuresHandler():
    """This method takes the raw candle data and constructs features based on
    the provided list of strings. It also can create, train, and hold on to 
    predictors. """
    
    def __init__(self, feature_dict):
        self.feature_dict=feature_dict
        print(f'f dict : {self.feature_dict}')
        self.predictors = {}
        self.predictor_make_functions = {'probability': get_probability, 
                                    'rolling probability': rolling_probability, 
                                    'knn': make_knn, 
                                    'mlp': make_mlp,
                                    'logistic regression': make_logistic_regression}


        self.predictor_predict_functions = {'probability': probability_predict, 
                                        'rolling probability': None, 
                                        'knn': make_knn, 
                                        'mlp': make_mlp,
                                        'logistic regression': make_logistic_regression}


    def printData(self, train, test):
        print('TRAINING')
        print(train.head())
        print('TESTING')
        print(test.head())


    def build_basic_features(self, input_df, markets):
        print('Constructing basic features... ', end = ' ')
        
        if self.feature_dict is None:
            print('no features built.')
            return input_df
        df = input_df.copy()       # At this higher level, it is safer to construct a new dataframe instead of modifying the one that gets passed.
        #This section constructs engineered features and adds them as columns to the df
        """ If you change the number of indicators in this function, be sure to also change the expected number in the enviroment"""

        # Check for illegal feature names
        acceptable_feature_names = ['sign', 'high', 'low',
                                    'OBV', 'EMA', 'MACD', 'RSI', 'BollingerBands', 'BBInd', 'BBWidth',
                                    'sentiment', 'renko', 'time of day', 'discrete_derivative',
                                    'knn', 'mlp', 'svc', 'ridge', 'stack',
                                    'probability', 'rolling probability',
                                    'divide']
        for f in self.feature_dict:           # Iterate over the keys
            if not f in acceptable_feature_names:
                print('An unrecognized feature has been passed to the feature contructor (not in the acceptable feature list).')
                print(f'Illegal feature: {f}')
                raise(ValueError)

        # Add the features to the df. It would be great to have a dict of features and functions but im not sure how to do that
        for market in markets:
            token = market[4:7] #this is something like 'BTC' or 'ETH'
            for feature in self.feature_dict:               # Iterate over the keys
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
                    for base in self.feature_dict[feature]:
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
                    for base in self.feature_dict['BollingerBands']:
                        b = str(base)
                        # The fractional position of the price relative to the high and low bands
                        diff = df['BTCClose'] - df['bb_bbl' + b]
                        df['BBInd'+ b] = diff/(df['bb_bbh' + b] - df['bb_bbl' + b]) - 0.5
                elif feature == 'BBWidth':
                    for base in self.feature_dict['BollingerBands']:
                        b = str(base)
                        df['BBWidth'+ b] = (df['bb_bbh' + b] - df['bb_bbl' + b])
                elif feature == 'discrete_derivative':
                    for col_name in self.feature_dict['discrete_derivative']:
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
                    pass

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
                elif feature == 'divide':
                    remember_divide = True
                elif feature in ['probability', 'rolling probability']:
                    # These get handled later
                    pass
                    
                else:
                    print(f'Feature: {feature} not found.')

            try:
                if bb_drop == True:
                    for feature in self.feature_dict['BollingerBands']:
                        names = [n + str(feature) for n in ['bb_bbl', 'bb_bbm', 'bb_bbh']]
                        try:
                            df.drop(columns=names, inplace=True)
                        except KeyError:
                            print(f'Error dropping column {names}')
                            print(df.columns)
                            # print(df.loc[names].iloc[0])
            except UnboundLocalError:
                pass
            
            # try:
            #     if remember_divide == True:
            #         df = divide(*self.feature_dict['divide'], df)
            # except UnboundLocalError:
            #     pass
        return df


    def train_predictors(self, input_df, markets, train_amount=0):
        """ This method creates and trains all of the predictors"""
        
        print('Train_predictors input:')
        print(input_df)

        # Check to see if there are any predictors to be trained
        if self.feature_dict is None:   # Or [self.feature_dict.keys in self.predictor_make_functions.keys()].any()
            return input_df.copy(), None    # Test, train
        
        # Create the training df (for ml, probability etc) Note that this gets done everytime regardless of whether its needed or not
        train_df = input_df.copy()

        # CATCH ANY INVALID VALUES GENERATED DURING FEATURE CONSTRUCTION
        # Replacing infinite with nan, drop all rows with nan values  
        train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        train_df.dropna(inplace=True)

        # alter the train and test datasets
        if train_amount > 0:
            # print('hey im splitting the dataset')
            split_date = train_df.index.min() + timedelta(days = train_amount)
            test_df = train_df.loc[train_df.index >= split_date].copy()
            train_df = train_df.loc[train_df.index < split_date]
        
        print('Split the data:')
        self.printData(train_df, test_df)
        
        train_df = make_criteria(train_df)
        train_df = make_criteria(train_df, buy_not_sell=False)
        
        test_df = make_criteria(test_df)
        test_df = make_criteria(test_df, buy_not_sell=False)

        # train all of the predictors
        for name in self.predictor_make_functions.keys():
            if name in self.feature_dict.keys():
                if name in ['probability']:
                    self.predictors[name] = self.predictor_make_functions[name](train_df, self.feature_dict[name])
                elif name in ['rolling probability']:
                    test_df, train_df = self.predictor_make_functions[name](test_df, train_df, self.feature_dict[name])
                else:
                    # self.predictors[name] = predictor_make_functions[name](test_df, train_df)
                    pass
        # separated out for easy looping over both dicts with the other functions. Useful for lazy learners
        # if 'probability' in feature_dict.keys():
        #     test_df = probability(test_df, train_df, self.feature_dict['probability'])

        # Loop the make the predictions
        # elif 'rolling probability' in feature_dict.keys():
        #     test_df = rolling_probability(test_df, train_df, self.feature_dict['rolling probability'])

        if 'knn' in self.feature_dict.keys():
            
            # Hyperparameters
            hyperparams = {'polynomial_features__degree': (1, 2), 
                            'knn__n_neighbors': (3, 4, 5)}
            buy_knn = knn_classifier(token, hyperparams)
            train_df = make_criteria(train_df, buy_not_sell=True)
            print('Fitting buy data.')
            # knn.train(df, 'Buy Labels')
            # buy_knn.optimize_parameters(train_df, 'Buy Labels')
            
            buy_knn.build_pipeline()    # Explicitly reset the model to an empty model
            buy_knn.train(train_df, 'Buy Labels')
            
            train_df.drop(columns='Buy Labels', inplace=True)
            df = buy_knn.build_feature(df, 'Buy')       # , ignore_names=['Buy Labels']
            
            print('Fitting sell data.')
            train_df = make_criteria(train_df, buy_not_sell=False)
            # knn.train(df, 'Sell Labels')
            sell_knn = knn_classifier(token, hyperparams)
            sell_knn.build_pipeline()
            sell_knn.train(train_df, 'Sell Labels', ignore_names=['BTCknn Buy'])
            df = sell_knn.build_feature(df, 'Sell', ignore_names=['Sell Labels', 'BTCknn Buy'])
            train_df.drop(columns='Sell Labels', inplace=True)

        print('done.')
        df_cols = test_df.columns

        # Strip out open high low close
        for market in markets:
            token = market[4:7]
            for col in df_cols:
                if col in [token + 'Open', token + 'High', token + 'Low']:
                    test_df.drop(columns=[col], inplace = True)

        label_cols = ['Buy Labels', 'Sell Labels']
        try:
            test_df.drop(columns=label_cols, inplace=True)
        except KeyError:
            pass
        try:
            train_df.drop(columns=label_cols, inplace=True)
        except KeyError:
            pass
        # test_df = divide(*self.feature_dict['divide'], test_df)
        return test_df, train_df


    def predict(self, test_df):
        """ Make predictions based on past trainings."""

        df = test_df.copy()

        # Loop over prediction functions
        for name in self.predictor_predict_functions.keys():
            if name in self.feature_dict.keys():
                if name in ['probability']:
                    df = self.predictor_predict_functions[name](self.predictors[name] , df, self.feature_dict[name])
                else:
                    df = self.predictor_predict_functions[name](df)
                    # df = predictor_make_functions[name](df, feature_dict[name])
        # df = probability(df, train_df, feature_dict['probability'])
        return df


if __name__ == "__main__":
    from environments.environments import SimulatedCryptoExchange

    start = datetime(2020, 1, 11)
    end = datetime(2020, 4, 30)

    features = {    # 'sign': ['Close', 'Volume'],
        # 'EMA': [50, 80, 130],
        # 'OBV': [],
        'RSI': [],
        'BollingerBands': [1, 3],
        'BBInd': [],
        'BBWidth': [],
        # 'discrete_derivative': ['BBWidth3'],
        'probability': ['BBInd3', 'BBWidth3'],
        # 'rolling probability': ['BBInd3', 'BBWidth3']
        # 'time of day': [],
        # 'stack': [5]
        }
    
    sim_env = SimulatedCryptoExchange(start, end, granularity=5, feature_dict=features, train_amount=10)
    df = sim_env.df.copy()

    builder = featuresHandler(features)
    df = builder.build_basic_features(df, markets=['USD-BTC'])
    df = builder.train_predictors(df, markets=['USD-BTC'], train_amount=10)
    # df = 


