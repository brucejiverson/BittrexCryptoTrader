import numpy as np
import warnings
from tools.tools import maybe_make_dir, f_paths
import pickle

class Agent:
    """This is the class from which all other agents inherit from.
    The purpose is to enforce certain standards between all of the agents, 
    and to make additional functionality scale."""
    def __init__(self, name, feature_map, action_size):
        self.name = name
        self.feature_map = feature_map # This is an ordered tuple that informs the agent what each of the numbers in the state means
        self.state_size = len(feature_map)
        self.action_size = action_size

        self.stop_loss = None
        self.stop_order = None
        self.trailing_stop_loss = None
        self.trailing_stop_order = None
        # self.trade_conditions = {'stop loss': {'val': None, 'f': lambda price, stop: price <= stop},
        #                         'stop order': {'val': None, 'f': lambda price, stop: price >= stop},
        #                         'trailing stop loss': {'val': None, 'extreme': None, 'f':    lambda price, percent: price <= percent},
        #                         'trailing stop order': {'val': None, 'extreme': None, 'f': lambda price, percent: price >= percent}
        #                         }
        self.reset_trade_conditions()

    def check_set_trade_conditions(self, cur_price):
        """Checks if stop loss or other future trades had been previously set, and returns
        -1 if should sell, 0 if neutral, 1 if should buy"""

        #Check if value has been set
        if self.stop_loss:
            if cur_price <= self.stop_loss:
                # Condition is triggered, reset and return
                self.stop_loss = None
                return -1

        #Check if value has been set
        elif self.stop_order:
            if cur_price >= self.stop_order:
                # Condition is triggered, reset and return
                self.stop_order = None
                return 1

        #Check if value has been set
        elif self.trailing_stop_loss['percent']:
            # Check if new extreme has been reached
            highest = self.trailing_stop_loss['highest']
            trig_price = self.trailing_stop_loss['trigger price']
            if cur_price > highest:
                self.trailing_stop_loss['highest'] = cur_price
                highest = cur_price
                # Update the trigger price
                percent = self.trailing_stop_loss['percent']
                trig_price = highest*(1 - percent/100)
                self.trailing_stop_loss['trigger price'] = trig_price # for debugging
            if cur_price <= trig_price:
                # Condition is triggered
                return -1
        
        #Check if value has been set
        elif self.trailing_stop_order['percent']:
            # Check if new extreme has been found
            lowest = self.trailing_stop_order['lowest']
            trig_price = self.trailing_stop_order['trigger price']
            if cur_price < lowest:
                self.trailing_stop_order['lowest'] = cur_price
                lowest = cur_price
                # Update the trigger price
                percent = self.trailing_stop_order['percent']
                lowest = self.trailing_stop_order['lowest']
                trig_price = lowest*(1 + percent/100)
                self.trailing_stop_order['trigger price'] = trig_price # for debugging

            if cur_price >= trig_price:
                # Condition is triggered
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
        with warnings.catch_warnings(record=True) as w:
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


    def act(self, state):

        i = self.feature_map['BTCknn']
        knn = state[i]      # Should be 0 or 1
        action = int(knn)
        return action


class bollingerAgent(Agent):

    def __init__(self, feature_map, action_size):
        super().__init__('bollinger', feature_map, action_size)

        self.last_action = 0
        self.last_price = 0

    def act(self, state):

        i = self.feature_map['BTCClose']
        price = state[i]      # Should be 0 or 1
        i = self.feature_map['bb_bbli']
        bbli = state[i]      # Should be 0 or 1
        i = self.feature_map['bb_bbhi']
        bbhi = state[i]      # Should be 0 or 1

        # print(price)
        if bbli == 1:
            action = 0

        elif bbhi == 1:
            action = 1
            # self.stop_loss = 0.995*price
            # print(self.stop_loss)
        else:
            action = self.last_action

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


class EMAReversion(Agent):

    def __init__(self, feature_map, action_size):
        super().__init__('EMAReversion', feature_map, action_size)
        self.last_act = 0
        self.hyperparams = {'upper': 25, 'lower': 4}


    def act(self, state):
        features = state[-self.n_feat::].reshape((1, self.n_feat))
        # print(state)
        # print(features)

        ddt_EMA_big = state[-1]
        ddt_EMA_med = state[-2]
        ddt_EMA_small = state[-3]

        EMA_big = state[-4]
        EMA_med = state[-5]
        EMA_small = state[-6]
        obv = state[-7]

        if self.last_act == 0:
            if EMA_small > self.hyperparams['upper'] :#or EMA_small > EMA_big:    #Need to buy
                action = 1
            else: action = self.last_act
        elif EMA_small < self.hyperparams['lower'] and self.last_act == 1:  #Need to sell
            action = 0
        else: action = self.last_act

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
    {'ind base': 5, 'high thresh': 0.2, 'low thresh': 0.5, 'order': 3, 'loss': 0, 'stop loss': 7}"""
    # Default hyperparams
    h = {'ind base': 10,'high thresh': 0.25, 'low thresh': 0.55, 'order': 3, 'loss': 0, 'stop loss': 7}
    def __init__(self, feature_map, action_size, hyperparams = h):
        super().__init__('doubleReverse', feature_map, action_size)

        self.last_action = 0
        self.reset_trade_conditions()
        
        self.hyperparams = hyperparams
        # self.trailing_stop_order['percent'] = self.hyperparams['order']
        self.states = (
                        'waiting for buy signal', 
                        'trailing stop order', 
                        'waiting for sell signal', 
                        'trailing stop loss')
        self.state_index = 0
        self.cur_state = self.states[self.state_index]

    def transition_states(self):
        """This function will switch to the next state. currently, states are purely sequential, 
        but eventually they may become more complex, this function will help to keep that logic clear"""
        if self.state_index >= len(self.states) - 1:
            self.state_index += 1
        else: 
            self.state_index = 0


    def act(self, state):

        i = self.feature_map['BTCClose']
        cur_price = state[i]      # Should be 0 or 1
        ind_name = 'BBInd' + str(self.hyperparams['ind base'])
        i = self.feature_map[ind_name]
        bbind = state[i]      # Should be 0 or 1
        
        if self.cur_state == 'waiting for buy signal':
            if bbind < -self.hyperparams['low thresh']:
                self.trailing_stop_order['percent'] = self.hyperparams['order']
        elif self.cur_state == 'waiting for sell signal ':
            if bbind > self.hyperparams['high thresh']:
                self.trailing_stop_loss['percent'] = self.hyperparams['loss']

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


class BuyAndHold(Agent):
    """ Responsible for taking actions, learning from them, and taking actions
    such that they will maximize future rewards
    """

    def __init__(self, feature_map, action_size):
        super().__init__('buy and hold', feature_map, action_size)


    def act(self, state):
        return 1