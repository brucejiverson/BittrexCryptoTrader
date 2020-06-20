import numpy as np
import warnings
from environments.environments import f_paths
import pickle



class LinearModel:
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
        assert(len(X.shape) == 2)    sim_env = SimulatedCryptoExchange(start, end, initial_investment=initial_investment)

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

    def __init__(self, state_size, action_size):

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
        self.model = LinearModel(state_size, action_size)

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


class SimpleAgent():
    """ Simple agent that trades solely based on the RSI feature
    """

    def __init__(self, state_size, action_size):    #same

        self.name = 'simple'
        # These two correspond to number of inputs and outputs of the neural network respectively
        self.state_size = state_size
        self.action_size = action_size


    def act(self, state):
        # print(state)
        # This is the policy
        #Currently assumes only bitcoin is in use, and that RSI is the last feature in the state
        # print(state)
        rsi = state[-1]
        macd = state[-2]
        obv = state[-3]
        # renko = state[0][-3]
        # if rsi < 1 and renko > -7: #buy
        # if rsi < -2.5: #-2.5 gives no downdraw, 2.81% over 10 days. Positive vals no good
        # if macd > 1: # [0] 6%, [1] 8.68% over 10 days
        # if macd > 0 and rsi < -1: # 8.7% over 10 days [0,0]
        # if obv < 0: #[-.1] 10.22% [-.5] 9.35% better downdraw. [-1] 8.1% no downdraw 10 days.
        # if obv < -.5 and macd > 0 and rsi < -2:
        # if macd > 2.5 and rsi < -4: #this is one of the best ones I have found at giving good profits while limiting downdraw
        # if abs(obv) > 8:
        if obv < -4:
            return 1
        else: #sell
            return 0


class RegressionAgent():

    def __init__(self, state_size, action_size):

        self.name = 'linear_regression'
        # These two correspond to number of inputs and outputs of the neural network respectively
        self.state_size = state_size
        self.action_size = action_size

        self.n_feat = self.state_size
        self.last_action = 0

        #Load the weights
        path = f_paths['models'] + '/regression.pkl'
        with open(path, 'rb') as file:
            self.model = pickle.load(file)

    def act(self, state):
        features = state[-self.n_feat::].reshape((1, self.n_feat))
        # print(state)
        # print(features)
        y_pred = self.model.predict(features)[0] #predicted percentage change
        # print(y_pred)
        # ddt = state[-1]
        # EMA = state[-2]
        # obv = state[-3]

        #sell in times of high volatility
        # if abs(EMA) > 250:
        #     return 0

        # condition = all([y_pred[0] > .08, y_pred[1] > 0, y_pred[2] > -.25])
        # condition = all([y_pred[0] > .05, y_pred[1] > 0])
        condition = y_pred[0] > .05
        if condition:
            action = 1                                          #buy
        # elif y_pred[0] > 0 and self.last_action == 1:
        #     action = 1                                          #hold
        else:
            action = 0                                          #sell
        self.last_action = action
        return action

    def save_weights(self):
        #Save the weights
        path = f_paths['models'] + '/regression.pkl'
        with open(path, 'wb') as file:
            pickle.dump(model, file)


class ClassificationAgent():

    def __init__(self, state_size, action_size):

        self.name = 'classifier'
        # These two correspond to number of inputs and outputs of the neural network respectively
        self.state_size = state_size
        self.action_size = action_size

        self.last_action = 0

        #Load the weights
        path = f_paths['models'] + '/classification.pkl'
        # with open(path, 'rb') as file:
        #     self.model = pickle.load(file)

    def act(self, state):

        try:
            y_pred = self.model.predict([state]) #predicted percentage change
            action = int(y_pred[0])
            # print(action)
        except ValueError:
            print(f'State: {state}')
            raise(ValueError)
        # # condition = y_pred > .05
        # if y_pred == 1:
        #     action = 1                                     #hold
        # else:
        #     action = 0                                          #sell
        self.last_action = action
        return action
    #
    # def save_weights(self):
    #     #Save the weights
    #     path = f_paths['models'] + '/regression.pkl'
    #     with open(path, 'wb') as file:
    #         pickle.dump(model, file)


class MeanReversionAgent():
    """This agent is a variation on Simple agent. It is based on the principle that when the last price change was negative,
    there is a high probability that the next price change will be positive."""

    def __init__(self, state_size, action_size):

        self.name = 'simple_momentum'
        # These two correspond to number of inputs and outputs of the neural network respectively

        self.state_size = state_size
        self.action_size = action_size

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


class EMAReversion():

    def __init__(self, state_size, action_size):    #same

        self.name = 'mean_reversion'
        # These two correspond to number of inputs and outputs of the neural network respectively
        self.state_size = state_size
        self.action_size = action_size

        self.n_feat = self.state_size
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


class MarketTester():
    def __init__(self, state_size, action_size):    #same

        self.name = 'tester'
        # These two correspond to number of inputs and outputs of the neural network respectively
        self.state_size = state_size
        self.action_size = action_size

        self.n_feat = self.state_size
        self.last_act = 0


    def act(self, state):

        if self.last_act == 0: action = 1
        else: action = 0

        self.last_act = action
        return action


class BenchMarker():
    """For now, this just uses the Renko strategy. Eventually,
    this should take in a string parameter that dictates which
    benchmarking strategy is used.

    Be careful with where each feature is in the state, as this class reads in
    features from the state by their position.
    The action to return should be a list of actions."""

    def __init__(state_size, action_size, block_size):
        self.action_size = action_size
        self.state_size = state_size

        self.block_size = block_size

    def act(self, state):
        thresh = 5 #change this
        if state[2] > thresh: #change this to match the position in the state
            return 1
        else:
            return 0


class BuyAndHold(object):
    """ Responsible for taking actions, learning from them, and taking actions
    such that they will maximize future rewards
    """

    def __init__(self, state_size, action_size):    #same

        # These two correspond to number of inputs and outputs of the neural network respectively
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.005  # originally .01. The version here is set for training
        self.epsilon_decay = 0.95 # originall .995
        self.learning_rate = .004
        # Get an instance of our model
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        # This is the policy
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)  # Greedy case

        # Take argmax over model predictions to get action with max. Q value.
        # Output of model is batch sized by num of outputs to index by 0
        return np.argmax(act_values[0])  # returns action (same)

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
