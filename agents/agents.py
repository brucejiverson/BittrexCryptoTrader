import numpy as np
import warnings
from tools.tools import maybe_make_dir, f_paths, ROI, percent_change_column
import pickle
import logging
import stats

logging.basicConfig(level=logging.CRITICAL)

class scorer:
    """This function scores the agent performance by summing the returns at each time step and dividing by std dev.
    Summing roi of each timestep ensures equal weighting of all times (avoids prioritizing early trades)
    Standard deviation is taken from sharpe, penalizes volatity in returns (adjusts for risk)."""
    def __init__(self, history):
        self.roi = None
        self.sharpe = None
        self.annualized_sharpe = None
        self.custom_score = None
        self.alpha = None
        self.beta = None
        self.calculate_scores(history)
        self.trade_analysis(history)


    def print_scores(self):
        print(f'Return on investment:    {round(self.roi, 2)}')
        print(f'Sharpe ratio:            {round(self.sharpe, 2)}')
        print(f'Annulaized sharpe ratio: {round(self.annualized_sharpe, 2)}')
        # print(f'Alpha:                  {round(self.alpha, 2)}')
        # print(f'Beta:                   {round(self.beta, 2)}')
        print(f'Custom score:            {round(self.custom_score, 2)}')
        
        
    def calculate_scores(self, history):
        df = history.copy()
        self.roi = ROI(df['Total Value'].iloc[0], df['Total Value'].iloc[-1])
        self.sharpe = self.roi/df['Total Value'].std()
        gran = 5
        self.annualized_sharpe = self.sharpe*(60*24*365/gran)**.5
        # asset_change = (df['BTCClose'].iloc[-1] - df['BTCClose'].iloc[0])/df['BTCClose'].iloc[0]
        # self.alpha = self.roi - asset_change
        # self.beta = 
        df = percent_change_column('Total Value', df)
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
        col = df['Total Value'].values
        std_dev = df['Total Value'].std()
        if std_dev != 0:
            self.custom_score = np.sum(col)/std_dev
        else: 
            self.custom_score = 0


    def trade_analysis(self, history):
        df = history.copy()
        df['delta'] = df['Actions'] - df['Actions'].shift(1)  # Converts the column to % change
        df['delta'].fillna(0)

        df = percent_change_column('Total Value', df)
        
        buys = df[df['delta'] > 0]
        sells = df[df['delta'] < 0]
        
        print(f'Number of buys: {buys.shape[0]}.')
        print(f'Number of sells: {sells.shape[0]}.')
        positions = []
        lengths = []
        # for i in range(len(buys)):
        #     buy = buys.iloc[i]
        #     sell = sells.iloc[i]

        #     buy_val = buy.loc['$ of BTC']
        #     sell_val = sell.loc['$ of BTC']


        #     # print(buy.reset_index())
        #     buy_time = buy.reset_index().columns[1]
        #     sell_time = sell.reset_index().columns[1]
        #     positions.append(sell_val - buy_val)
        #     lengths.append(sell_time - buy_time)

        # mean = sum(positions) / len(positions) 
        # var = sum((i - mean) ** 2 for i in positions) / len(positions) 
        # print(f'The mean and variance return for a trade cycle is mean: {mean}, var: {var}')
        # mean = sum(lengths) / len(lengths) 
        # var = sum((i - mean) ** 2 for i in lengths) / len(lengths) 
        # print(f'The mean position time was {mean}, and the variance {var}')

        # print(f'Mean of return ')


    def get_scores(self):
        return [self.roi, self.sharpe, self.custom_score]


class Agent:
    """This is the class from which all other agents inherit from.
    The purpose is to enforce certain standards between all of the agents, 
    and to make additional functionality scale."""
    def __init__(self, name, feature_map, action_size):

        #Creating an object 
        self.logger=logging.getLogger() 
        # logging.basicConfig(level=logging.DEBUG)
        #Setting the threshold of logger to DEBUG 
        # self.logger.setLevel(logging.DEBUG)
        self.name = name
        self.feature_map = feature_map # This is an ordered tuple that informs the agent what each of the numbers in the state means
        self.state_size = len(feature_map)
        self.action_size = action_size

        self.stop_loss = None
        self.stop_order = None
        self.trailing_stop_loss = None
        self.trailing_stop_order = None

        self.reset_trade_conditions()

        self.states = (None)
        self.state_index = 0
        self.cur_state = None   # self.states[self.state_index]


    def transition_states(self):
        """This function will switch to the next state. currently, states are purely sequential, 
        but eventually they may become more complex, this function will help to keep that logic clear"""
        if self.state_index < len(self.states) - 1:
            self.state_index += 1
        else: 
            self.state_index = 0
        self.cur_state = self.states[self.state_index]
        # print(self.cur_state)


    def check_set_trade_conditions(self, cur_price):
        """Checks if stop loss or other future trades had been previously set, and returns
        -1 if should sell, 0 if neutral, 1 if should buy"""
        # self.logger.debug(self.trailing_stop_order)
        
        # self.logger.debug(f'{self.trailing_stop_loss}')
        #Check if value has been set
        if self.stop_loss:
            if cur_price <= self.stop_loss:
                # Condition is triggered, reset and return
                logging.debug('Stop loss triggered')

                self.stop_loss = None
                return -1

        #Check if value has been set
        if self.stop_order:
            if cur_price >= self.stop_order:
                # Condition is triggered, reset and return
                logging.debug('Stop order triggered')

                self.stop_order = None
                return 1

        #Check if value has been set
        if self.trailing_stop_loss['percent']:
            # Check if new extreme has been reached
            highest = self.trailing_stop_loss['highest']
            trig_price = self.trailing_stop_loss['trigger price']
            if cur_price > highest or highest is None:
                self.trailing_stop_loss['highest'] = cur_price
                highest = cur_price
                # Update the trigger price
                percent = self.trailing_stop_loss['percent']
                trig_price = highest*(1 - percent/100)
                self.trailing_stop_loss['trigger price'] = trig_price # for debugging
            if cur_price <= trig_price:
                # Condition is triggered
                logging.debug('Trailing stop loss triggered')
                return -1
        
        #Check if value has been set
        if self.trailing_stop_order['percent']:
            # Check if new extreme has been found
            lowest = self.trailing_stop_order['lowest']
            trig_price = self.trailing_stop_order['trigger price']
            if cur_price < lowest or lowest is None:
                self.trailing_stop_order['lowest'] = cur_price
                lowest = cur_price
                # Update the trigger price
                percent = self.trailing_stop_order['percent']
                lowest = self.trailing_stop_order['lowest']
                trig_price = lowest*(1 + percent/100)
                self.trailing_stop_order['trigger price'] = trig_price # for debugging

            if cur_price >= trig_price:
                # Condition is triggered
                logging.debug('Trailing stop order triggered')
                return 1
        else: return 0
    

    def reset_trade_conditions(self):
        self.stop_loss = None
        self.stop_order = None
        
        self.trailing_stop_loss = {'percent': None, 'highest': 0, 'trigger price': None}
        self.trailing_stop_order = {'percent': None, 'lowest': np.inf, 'trigger price': None}


class LinearModel():
    """ A linear regression model """

    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / \
            np.sqrt(input_dim)  # Random matrix
        self.b = np.zeros(n_action)  # Vector of zeros

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []

    def predict(self, X):
        # make sure X is N x D
        # throw error if X not 2D to abide by (skikitlearn dimensionality convention)
        assert(len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.0025, momentum=0.4):
        """One step of gradient descent.dddd
        learning rate was originally 0.01
        u = momentum term
        n = learning rate
        g(t) = gradient
        theta = generic parameter
        v(t) = u*v(t-1) - n*g(t)
        let theta = T
        T(t) = T(t-1) + v(t), T = {W,b)}
        """

        # make sure X is N x D
        assert(len(X.shape) == 2)

        # the loss values are 2-D
        # normally we would divide by N only
        # but now we divide by N x K
        num_values = np.prod(Y.shape)

        # do one step of gradient descent
        # we multiply by 2 to get the exact gradient
        # (not adjusting the learning rate)
        # i.e. d/dx (x^2) --> 2x
        with warnings.catch_warnings(record=True):
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            Yhat = self.predict(X)
            gW = 2 * X.T.dot(Yhat - Y) / num_values
            gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y)**2)  # Using the mean squared error (This was from the class code, started throwing runtime errors)
        # mse = ((Yhat - Y)**2).mean(axis = None) #still throws run time errors
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)


class DQNAgent(object):
    """ Responsible for taking actions, learning from them, and taking actions
    such that they will maximize future rewards
    """

    def __init__(self,  action_size):

        self.name = 'dqn'
        # These two correspond to number of inputs and outputs of the neural network respectively
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.005  # originally .01. The version here is set for training
        self.epsilon_decay = 0.95 # originall .995
        self.learning_rate = .004
        # Get an instance of our model
        self.model = LinearModel( action_size)

    def act(self, state):
        # This is the policy
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)  # Greedy case

        # Take argmax over model predictions to get action with max. Q value.
        # Output of model is batch sized by num of outputs to index by 0
        return np.argmax(act_values[0])  # returns action

    def train(self, state, action, reward, next_state, done):
        # This func. does the learning
        if done:
            target = reward
        else:
            target = reward + self.gamma * \
                np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target

        # Run one training step of gradient descent.
        self.model.sgd(state, target_full, self.learning_rate)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class knnAgent(Agent):

    def __init__(self, feature_map, action_size):
        super().__init__('knn algo', feature_map, action_size)

        self.last_action = 0
        
        self.price_at_prediction = None
        self.states = ('Waiting for buy',
                        'Waiting for sell')

    def act(self, state):

        i = self.feature_map['BTCClose']
        cur_price = state[i]
        i = self.feature_map['BTCknn Buy']
        buy_knn = state[i]      # Should be 0 or 1
        
        i = self.feature_map['BTCknn Sell']
        sell_knn = state[i]      # Should be 0 or 1
        
        if buy_knn == 1 and sell_knn != 1:    # and sell_knn == 0
            # Positive prediction of what is about to happen
            action = 1
            # self.trailing_stop_loss['percent'] = self.hyperparams['stop loss']
            self.transition_states()
        elif  sell_knn == 1:  # buy_knn == 0 and
            # Positive prediction of what is about to happen
            action = 0
            self.transition_states()
    
        else: action = self.last_action

        # handle stop loss and others for loss management
        # result = self.check_set_trade_conditions(cur_price)
        # if result == -1:    # Sell
        #     self.transition_states()    # moves to waiting for buy signal
        #     action = 0
        #     self.reset_trade_conditions()
        self.last_action = action
        return action


class MeanReversionAgent(Agent):
    """This agent is a variation on Simple agent. It is based on the principle that when the last price change was negative,
    there is a high probability that the next price change will be positive."""

    def __init__(self, feature_map, action_size):
        super().__init__('Mean Reversion', feature_map, action_size)

        self.last_state = np.zeros(self.state_size)
        self.hyperparams = [-2, 2, 0]  #Organized as price threshhold, last price threshold, hold threshold
        self.last_act = 0


    def act(self, state):

        price = state[0]
        last_price = self.last_state[0]

        # buy_signal = all([price < self.hyperparams[0]])
        buy_signal = all([price < self.hyperparams[0], last_price < self.hyperparams[1]])
        hold_signal = all([price > self.hyperparams[2], self.last_act == 1])
        if buy_signal: action = 1                               #buy
        elif hold_signal: action = 1       #hold
        else: action = 0

        self.last_state = state
        self.last_act = action
        return action


class doubleTrailingStop(Agent):

    def __init__(self, feature_map, action_size, hyperparams = {'order': 1, 'loss': 1}):
        super().__init__('doubleReverse', feature_map, action_size)

        self.last_action = 0
        self.reset_trade_conditions()
        
        self.hyperparams = hyperparams
        self.trailing_stop_order['percent'] = self.hyperparams['order']
        

    def act(self, state):

        i = self.feature_map['BTCClose']
        cur_price = state[i]      # Should be 0 or 1
        
        
        result = self.check_set_trade_conditions(cur_price)
        if result == -1:    # Sell
            action = 0
            self.reset_trade_conditions()
            self.trailing_stop_order['percent'] = self.hyperparams['order']
        elif result == 1:   # Buy
            action = 1
            self.reset_trade_conditions()
            self.trailing_stop_loss['percent'] = self.hyperparams['loss']
        else: action = self.last_action
        # print(f'stop loss {self.trailing_stop_loss}')
        # print(f'stop order {self.trailing_stop_order}')
        self.last_action = action
        return action


class biteBack(Agent):
    """This agent uses trailing stops and bollinger bands to capture mean reversion"""
    """Grid search results: gran 30, BBInd 5 ran with rules: trailing stop order enables below BBInd thresh, 
    trailing stop loss enabled immediately, stop loss with smaller percent enabled when above high thresh, 
    used 0s in grid search that would skip some rules. Results: Best roi: 11.83, 
    best params: {'high thresh': 0.3, 'low thresh': 0.6, 'order': 2, 'loss': 0, 'stop loss': 5} (many were at edges of parameters)
    Expanded to a year long, reran with widened search: Best roi: 125.75, best params: 
    {'ind base': 5, 'high thresh': 0.2, 'low thresh': 0.5, 'order': 3, 'loss': 0, 'stop loss': 7}
    Best roi: 81.81, best params: {'ind base': 10, 'high thresh': 0.2, 'low thresh': 0.5, 'order': 2, 'loss': 0, 'stop loss': 7}
    Shorter seach with lots of params: Best params: {'ind base': 5, 'high thresh': 0.5, 'low thresh': 0.5, 'order': 0.1, 'loss': 1, 'stop loss': None}
    Best params: {'ind base': 4, 'high thresh': 0.3, 'low thresh': 0.45, 'order': 0.25, 'loss': 1, 'stop loss': None}"""
    # Default hyperparams
    h = {'ind base': 10,'high thresh': 0.25, 'low thresh': 0.55, 'order': 2.5, 'loss': 0, 'stop loss': 7}
    def __init__(self, feature_map, action_size, hyperparams = h):
        super().__init__('biteBack', feature_map, action_size)

        self.last_action = 0
        self.reset_trade_conditions()
        
        self.hyperparams = hyperparams
        # self.trailing_stop_order['percent'] = self.hyperparams['order']
        self.states = ('waiting for buy signal', 
                        'trailing stop order', 
                        'waiting for sell signal', 
                        'trailing stop loss')
        self.state_index = 0
        self.cur_state = self.states[self.state_index]


    def act(self, state):

        # if self.cur_state == 'trailing stop loss':
        #     print(self.trailing_stop_loss)
        i = self.feature_map['BTCClose']
        cur_price = state[i]      # Should be 0 or 1
        ind_name = 'BBInd' + str(self.hyperparams['ind base'])
        i = self.feature_map[ind_name]
        bbind = state[i]      # Should be 0 or 1
        
        # Rules for transitioning states
        # Note that states also change when a trade is triggered
        if self.cur_state == 'waiting for buy signal':
            if bbind < -self.hyperparams['low thresh']:
                self.trailing_stop_order['percent'] = self.hyperparams['order']
                self.transition_states()
        # elif self.cur_state == 'trailing stop order':
        #     if bbind 
        elif self.cur_state == 'waiting for sell signal':
            if bbind > self.hyperparams['high thresh']:
                self.trailing_stop_loss['percent'] = self.hyperparams['loss']
                self.transition_states()

        result = self.check_set_trade_conditions(cur_price)
        if result == -1:    # Sell
            self.transition_states()    # moves to waiting for buy signal
            action = 0
            self.reset_trade_conditions()
        elif result == 1:   # Buy
            self.transition_states()    # moves to trailing stop loss
            self.reset_trade_conditions()
            self.trailing_stop_loss['percent'] = self.hyperparams['stop loss']
            action = 1
        else: action = self.last_action
        # print(f'stop loss {self.trailing_stop_loss}')
        # print(f'stop order {self.trailing_stop_order}')
        self.last_action = action
        return action


class probabilityAgent(Agent):
    """This agent uses a feature built on probability analysis"""
    # Default hyperparams
    h = {'buy thresh': 0.65, 'sell thresh': 0.65, 'stop loss': None}
    def __init__(self, feature_map, action_size, hyperparams = h):
        super().__init__('probability agent', feature_map, action_size)

        self.last_action = 0
        self.reset_trade_conditions()

        self.periods_since_buy = 0
        self.buy_certainty_thresh = hyperparams['buy thresh']
        self.sell_certainty_thresh = hyperparams['sell thresh']

        self.hyperparams = hyperparams
        # self.trailing_stop_order['percent'] = self.hyperparams['order']
        self.states = ('waiting for buy signal', 
                        'waiting for sell signal')
        self.state_index = 0
        self.cur_state = self.states[self.state_index]
        self.min_hold_period = 1
        self.hold_counter = 0


    def act(self, state):

        i = self.feature_map['BTCClose']
        cur_price = state[i]

        i = self.feature_map['Likelihood of buy given x']
        buy_given_x = state[i]      # Should be 0 or 1
        
        i = self.feature_map['Likelihood of sell given x']
        sell_given_x = state[i]      # Should be 0 or 1
        
        # Rules for transitioning states
        # Note that states also change when a trade is triggered
        if self.cur_state == 'waiting for buy signal':
            if buy_given_x >= self.buy_certainty_thresh and sell_given_x <= .4:
                self.reset_trade_conditions()
                action = 1
                if not self.hyperparams['stop loss'] is None:
                    self.stop_loss = cur_price*self.hyperparams['stop loss']
                # self.trailing_stop_loss['percent'] = .5
                self.transition_states()
            
            else: action = self.last_action

        elif self.cur_state == 'waiting for sell signal':
            if self.hold_counter >= self.min_hold_period:
                if sell_given_x >= self.sell_certainty_thresh and buy_given_x <= .4:
                    # sell immediately
                    self.reset_trade_conditions()
                    action = 0
                    self.transition_states()
                    self.hold_counter = 0
                    
                else: action = self.last_action
            else:
                self.hold_counter += 1
                action = self.last_action
        else:   # should never end up here
            raise ValueError

        result = self.check_set_trade_conditions(cur_price)
        if result == -1:    # Sell
            self.transition_states()    # moves to waiting for buy signal
            action = 0
            self.reset_trade_conditions()
        elif result == 1:   # Buy
            self.transition_states()    # moves to trailing stop loss
            self.reset_trade_conditions()
            # self.trailing_stop_loss['percent'] = self.hyperparams['stop loss']
            action = 1
        # else: action = self.last_action
        # print(f'stop loss {self.trailing_stop_loss}')
        # print(f'stop order {self.trailing_stop_order}')
        self.last_action = action
        return action



class ratioProbabilityAgent(Agent):
    """This agent uses a feature built on probability analysis"""
    # Default hyperparams
    h = {'buy thresh': .2, 'sell thresh': 0, 'stop loss': None}
    def __init__(self, feature_map, action_size, hyperparams = h):
        super().__init__('ratio probability agent', feature_map, action_size)

        self.last_action = 0
        self.reset_trade_conditions()

        self.periods_since_buy = 0
        self.buy_certainty_thresh = hyperparams['buy thresh']
        self.sell_certainty_thresh = hyperparams['sell thresh']

        self.hyperparams = hyperparams
        # self.trailing_stop_order['percent'] = self.hyperparams['order']
        self.states = ('waiting for buy signal', 
                        'waiting for sell signal')
        self.state_index = 0
        self.cur_state = self.states[self.state_index]
        self.min_hold_period = 1
        self.hold_counter = 0


    def act(self, state):

        i = self.feature_map['BTCClose']
        cur_price = state[i]

        i = self.feature_map['Likelihood of buy given x']
        buy_given_x = state[i]      # Should be 0 or 1
        
        i = self.feature_map['Likelihood of sell given x']
        sell_given_x = state[i]      # Should be 0 or 1
        
        ratio = buy_given_x/sell_given_x - 1 

        # Rules for transitioning states
        # Note that states also change when a trade is triggered
        if self.cur_state == 'waiting for buy signal':
            if ratio >= self.buy_certainty_thresh and buy_given_x > .3:
                self.reset_trade_conditions()
                action = 1
                if not self.hyperparams['stop loss'] is None:
                    self.stop_loss = cur_price*self.hyperparams['stop loss']
                # self.trailing_stop_loss['percent'] = .5
                self.transition_states()
            
            else: action = self.last_action

        elif self.cur_state == 'waiting for sell signal':
            if self.hold_counter >= self.min_hold_period:
                if ratio <= self.sell_certainty_thresh:
                    # sell immediately
                    self.reset_trade_conditions()
                    action = 0
                    self.transition_states()
                    self.hold_counter = 0
                    
                else: action = self.last_action
            else:
                self.hold_counter += 1
                action = self.last_action
        else:   # should never end up here
            raise ValueError

        result = self.check_set_trade_conditions(cur_price)
        if result == -1:    # Sell
            self.transition_states()    # moves to waiting for buy signal
            action = 0
            self.reset_trade_conditions()
        elif result == 1:   # Buy
            self.transition_states()    # moves to trailing stop loss
            self.reset_trade_conditions()
            # self.trailing_stop_loss['percent'] = self.hyperparams['stop loss']
            action = 1
        # else: action = self.last_action
        # print(f'stop loss {self.trailing_stop_loss}')
        # print(f'stop order {self.trailing_stop_order}')
        self.last_action = action
        return action


class simpleBiteBack(Agent):
    """This agent uses trailing stops and bollinger bands to capture mean reversion"""
    """ """
    # Default hyperparams
    h = {'ind base': 10,'high thresh': 0.25, 'low thresh': 0.45, 'stop loss': 3}
    def __init__(self, feature_map, action_size, hyperparams = h):
        super().__init__('simpleBiteBack', feature_map, action_size)

        self.last_action = 0
        self.reset_trade_conditions()
        
        self.hyperparams = hyperparams
        # self.trailing_stop_order['percent'] = self.hyperparams['order']
        self.states = ('waiting for buy signal', 
                        'waiting for sell signal')
        self.state_index = 0
        self.cur_state = self.states[self.state_index]


    def transition_states(self):
        """This function will switch to the next state. currently, states are purely sequential, 
        but eventually they may become more complex, this function will help to keep that logic clear"""
        if self.state_index < len(self.states) - 1:
            self.state_index += 1
        else: 
            self.state_index = 0
        self.cur_state = self.states[self.state_index]
        # print(self.cur_state)


    def act(self, state):

        # if self.cur_state == 'trailing stop loss':
        #     print(self.trailing_stop_loss)
        i = self.feature_map['BTCClose']
        cur_price = state[i]      # Should be 0 or 1
        ind_name = 'BBInd' + str(self.hyperparams['ind base'])
        i = self.feature_map[ind_name]
        bbind = state[i]      # Should be 0 or 1
        
        # Rules for transitioning states
        # Note that states also change when a trade is triggered
        action = self.last_action                           # default
        if self.cur_state == 'waiting for buy signal':
            if bbind < -self.hyperparams['low thresh']:
                action = 1
                self.transition_states()
                self.trailing_stop_loss['percent'] = self.hyperparams['stop loss']

        # elif self.cur_state == 'trailing stop order':
        #     if bbind 
        elif self.cur_state == 'waiting for sell signal':
            if bbind > self.hyperparams['high thresh']:
                action = 0
                self.transition_states()

        
        # logging.debug(f"{self.trailing_stop_loss}")

        result = self.check_set_trade_conditions(cur_price)
        if result == -1:    # Sell
            self.transition_states()    # moves to waiting for buy signal
            action = 0
            self.reset_trade_conditions()
        elif result == 1:   # Buy
            self.transition_states()    # moves to trailing stop loss
            self.reset_trade_conditions()
            action = 1
        
        # print(f'stop loss {self.trailing_stop_loss}')
        # print(f'stop order {self.trailing_stop_order}')
        self.last_action = action
        return action


class TestLogic(Agent):
    # Default hyperparams
    def __init__(self, feature_map, action_size):
        super().__init__('testTradeFunctionality', feature_map, action_size)

        self.last_action = 0
        self.reset_trade_conditions()

    def act(self, state):

        i = self.feature_map['BTCClose']
        cur_price = state[i]      # Should be 0 or 1
        # print(cur_price)
        result = self.check_set_trade_conditions(cur_price)
        # print(result)
        # print(self.trailing_stop_loss)
        if result == -1:    # Sell
            action = 0
            self.reset_trade_conditions()
        elif result == 1:   # Buy
            self.reset_trade_conditions()
            action = 1
        else: action = self.last_action
        self.last_action = action
        return action


class BuyAndHold(Agent):
    """ Responsible for taking actions, learning from them, and taking actions
    such that they will maximize future rewards
    """

    def __init__(self, feature_map, action_size):
        super().__init__('buy and hold', feature_map, action_size)


    def act(self, state):
        return 1