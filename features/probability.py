from tools.tools import percent_change_column
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from scipy import stats
from sklearn.model_selection import train_test_split


# Calculate what the sum for the next n timesteps are in terms of percentage

# I should be plotting parameter ratio over performance?

# HERE, THE DATA GETS "PRUNED" BASED ON THE RELATIVE DENSITY OF "BUY" OR "NOT BUY USING KERNEL DENSITY ESTIMATE"

class statisticsBoy():
    """ This function should do the following:
        get kernels for any new predictions (this is like a feature)
        adding some columns to the dataframe for basic probability"""
    def __init__(self, target_change = .15, allowable_reversal=-.15):
        self.kernels = {}
        self.prob_of_buy_signal = None
        self.prob_of_sell_signal = None
        
        self.target_change = target_change   # In percentage units
        self.allowable_reversal = allowable_reversal
        

    def make_criteria(self, input_df, target_time_range=13):
        # This is currently symmetric (prediction mirrored conditions for buy/sell signals)
        df = input_df.copy()

        #Loop to create percentage columns
        generated_names = []
        for i in range(target_time_range):
            df, label_name = percent_change_column('BTCClose', df, -(i+1))
            generated_names.append(label_name)

        df.dropna(inplace=True)

        # THIS IS WHERE THE CONDITIONS FOR BUYING ARE DEFINED
        # Calculate some basic parameters around take profit and downdraw (I may be using that word wrong?)
        df['Max increase'] = df[generated_names].max(axis=1)
        df['Max decrease'] = df[generated_names].min(axis=1)
        df['Labels'] = np.zeros(df.shape[0])      # Initialize as all samples as "not suitable for buying"
        df.loc[(df['Max increase'] > self.target_change) & (df['Max decrease'] > self.allowable_reversal), 'Labels'] = 1        # Predicted its going up
        df.loc[(df['Max increase'] < -self.allowable_reversal) & (df['Max decrease'] < -self.target_change), 'Labels'] = -1     # Predicted its going down
        
        # Drop the temporary columns
        df.drop(columns=['Max increase', 'Max decrease', *generated_names], inplace=True)
        return df

    def plot_make_criteria(self, input_df):
        df = input_df.copy()
        fig, ax = plt.subplots(1, 1)

        input_df.plot(y='BTCClose', ax=ax)
        print(df.head())
        print(df.reset_index().head())
        buys = input_df[input_df['Labels'] == 1]
        sells = input_df[input_df['Labels'] == -1]
        buys.reset_index().plot(y = 'BTCClose', x='index', ax=ax, kind='scatter', marker='^', c='g', zorder=4)
        sells.reset_index().plot(y = 'BTCClose', x='index', ax=ax, kind='scatter', marker='v', c='r', zorder=4)
        fig.autofmt_xdate()
        fig.suptitle(f'Make criteria', fontsize=14, fontweight='bold')


    def do_analysis(self, df, names):

        self.prob_of_buy_signal = df.loc[df['Labels'] == 1].shape[0]/df.shape[0]                             # odds of getting a buy signal
        self.prob_of_sell_signal = df.loc[df['Labels'] == -1].shape[0]/df.shape[0]                            # odds of getting a sell signal

        positions = df[names].values.T          # Get the data. 
        x_kernel = stats.gaussian_kde(positions)                   # create and fit the kernel obj
        self.kernels['x'] = x_kernel

        # Get the likelihood kernel for the "buy" samples given "X"
        should_buy_vals = df[df['Labels'] == 1][names].values.T    # Get a numpy array and transpose it so data is in rows
        x_given_buy_kernel = stats.gaussian_kde(should_buy_vals)
        self.kernels['x given buy'] = x_given_buy_kernel
        
        # Get the likelihood kernel for the "sell" samples given "X"
        should_sell_vals = df[df['Labels'] == -1][names].values.T    # Get a numpy array and transpose it so data is in rows
        x_given_sell_kernel = stats.gaussian_kde(should_sell_vals)
        self.kernels['x given sell'] = x_given_sell_kernel


    def build_feature(self, input_df, names):
        # For now, not doing train/test split, but this function should do that, use cross validation, and then rebuild all. I think
        df = input_df.copy()

        # Calculate the probabilities for each sample (evaluation of training data)
        
        # Get the kernels that should have been made earlier
        x_kernel = self.kernels['x']
        x_given_buy_kernel = self.kernels['x given buy']
        x_given_sell_kernel = self.kernels['x given sell']
        
        # The positions in the feature space at which to evaluate the kernels
        positions = df[names].values.T

        # Evaluate the kernels at positions
        x_pde = x_kernel(positions).T
        x_given_buy = x_given_buy_kernel(positions).T
        x_given_sell = x_given_sell_kernel(positions).T

        # Use bayes theorem to calculate the liklihood
        buy_given_x = np.multiply(x_given_buy, self.prob_of_buy_signal)                                      # multiply by odds of getting buy sig
        buy_given_x = np.divide(buy_given_x, x_pde, out=np.zeros_like(buy_given_x), where=x_pde!=0)     # divide by probability of getting a sample at X
        
        sell_given_x = np.multiply(x_given_sell, self.prob_of_sell_signal)                                   # multiply by odds of getting sell sig
        sell_given_x = np.divide(sell_given_x, x_pde, out=np.zeros_like(sell_given_x), where=x_pde!=0)  # divide by probability of getting a sample at X

        # Build columns in the dataframe (mostly used for plotting? And also filtering I think)
        df['Likelihood of buy given x'] = buy_given_x
        df['Likelihood of sell given x'] = sell_given_x

        # This worked with >70% untuned on training data when relative prob was difference. Thresh = .005
        thresh = 0.65
        # df['Prediction'] = np.zeros(df.shape[0])
        # df.loc[(df['Likelihood of buy given x'] >= thresh), 'Prediction'] = 1
        # df.loc[(df['Likelihood of sell given x'] >= thresh), 'Prediction'] = -1
        # df.loc[(df['Likelihood of sell given x'] >= thresh), (df['Likelihood of buy given x'] >= thresh), 'Prediction'] = 0         # This should never happen, its here just in case

        n_buy_sigs = df.loc[(df['Likelihood of buy given x'] >= thresh)].shape[0]
        mean_buy_freq = (24*60/5)*n_buy_sigs/df.shape[0]        # signals per sample *(1/ 5 minutes/sample)*60*24 minutes/day = signals/day
        print(f'Mean of {mean_buy_freq} buy signals per day')
        n_sell_sigs = df.loc[(df['Likelihood of sell given x'] >= thresh)].shape[0]
        mean_buy_freq = (24*60/5)*n_sell_sigs/df.shape[0]        # signals per sample *(1/ 5 minutes/sample)*60*24 minutes/day = signals/day
        print(f'Mean of {mean_buy_freq} sell signals per day')

        # Drop unwanted columns
        # cols = ['Likelihood of buy given x', 'Likelihood of sell given x']
        # df.drop(columns=cols, inplace=True)
        return df


    def build_filtered_label(self, input_df, names):
        # For now, not doing train/test split, but this function should do that, use cross validation, and then rebuild all. I think
        df = input_df.copy()

        # Calculate the probabilities for each sample (evaluation of training data)
        
        # Get the kernels that should have been made earlier
        x_kernel = self.kernels['x']
        x_given_buy_kernel = self.kernels['x given buy']
        x_given_sell_kernel = self.kernels['x given sell']
        
        # The positions in the feature space at which to evaluate the kernels
        positions = df[names].values.T

        # Evaluate the kernels at positions
        x_pde = x_kernel(positions).T
        x_given_buy = x_given_buy_kernel(positions).T
        x_given_sell = x_given_sell_kernel(positions).T

        # Use bayes theorem to calculate the liklihood
        buy_given_x = np.multiply(x_given_buy, self.prob_of_buy_signal)                                      # multiply by odds of getting buy sig
        buy_given_x = np.divide(buy_given_x, x_pde, out=np.zeros_like(buy_given_x), where=x_pde!=0)     # divide by probability of getting a sample at X
        
        sell_given_x = np.multiply(x_given_sell, self.prob_of_sell_signal)                                   # multiply by odds of getting sell sig
        sell_given_x = np.divide(sell_given_x, x_pde, out=np.zeros_like(sell_given_x), where=x_pde!=0)  # divide by probability of getting a sample at X

        # Build columns in the dataframe (mostly used for plotting? And also filtering I think)
        df['Likelihood of buy given x'] = buy_given_x
        df['Likelihood of sell given x'] = sell_given_x

        # This worked with >70% untuned on training data when relative prob was difference. Thresh = .005
        thresh = 0.65
        # df['Prediction'] = np.zeros(df.shape[0])
        # df.loc[(df['Likelihood of buy given x'] >= thresh), 'Prediction'] = 1
        # df.loc[(df['Likelihood of sell given x'] >= thresh), 'Prediction'] = -1
        # df.loc[(df['Likelihood of sell given x'] >= thresh), (df['Likelihood of buy given x'] >= thresh), 'Prediction'] = 0         # This should never happen, its here just in case

        n_buy_sigs = df.loc[(df['Likelihood of buy given x'] >= thresh)].shape[0]
        mean_buy_freq = (24*60/5)*n_buy_sigs/df.shape[0]        # signals per sample *(1/ 5 minutes/sample)*60*24 minutes/day = signals/day
        print(f'Mean of {mean_buy_freq} buy signals per day')
        n_sell_sigs = df.loc[(df['Likelihood of sell given x'] >= thresh)].shape[0]
        mean_buy_freq = (24*60/5)*n_sell_sigs/df.shape[0]        # signals per sample *(1/ 5 minutes/sample)*60*24 minutes/day = signals/day
        print(f'Mean of {mean_buy_freq} sell signals per day')

        # Drop unwanted columns
        # cols = ['Likelihood of buy given x', 'Likelihood of sell given x']
        # df.drop(columns=cols, inplace=True)
        return df


    def analyze_probability_threshold(self, df):
        # Should this be k fold crossvalidation? 
        # I want to know how many signals are predictable based on past data. It appears to be pretty random, but slightly repeatable. 
        # I need a list of open questions
            # which features lead to the most predictable values?
                # Keys to exectuting this: crossvalidation, loop, scoring
            # how should these predictions best be applied in an algorithm? This seems like a model/theory type question
                # can I look this up?
            # what do you do with multiple overlapping predictions?
        # find close to the optimal probability threshold to use
        threshholds = [.5, .55, .65, .70, .75, .80, .85, .9]
        # print(df.columns)
        for t in threshholds:
            n_good_sigs = df.loc[(df['Likelihood of buy given x'] >= t) & (df['Labels'] == 1)].shape[0]
            rough_profit = self.target_change*n_good_sigs
            rough_losses = df.loc[(df['Likelihood of buy given x'] >= t) & (df['Labels'] == 0), '% Change 5 steps in future'].sum()
            print(f'Estimated performance for threshold {t} is {rough_profit + rough_losses}')


    def analyze_make_criteria(self, df):
        # splits data at 80% train 80% test by default
        all_cols = list(df)
        earliest = df.index.min()
        most_recent = df.index.max()

        # calculate 80 percent
        timerange = most_recent - earliest 
        split_line = earliest + timerange/0.8
        print(split_line)
        X = df[all_cols.remove('Labels')].values
        Y = df['Labels'].values


        data_positions = df[[name1, name2]].values.T                    # Get the data. 

        # fit the model to the training data
        should_buy_vals = df[df['Labels'] == 1][[name1, name2]].values.T    # Get a numpy array and transpose it so data is in rows
        x_given_buy_kernel = stats.gaussian_kde(should_buy_vals)
        x_kernel = stats.gaussian_kde(data_positions)                       # create and fit the kernel obj


        # evalute the model on the test data
        # Use bayes theorem to calculate the liklihood
        buy_given_x = np.multiply(x_given_buy, self.prob_of_buy_signal)                                      # multiply by odds of getting buy sig
        buy_given_x = np.divide(buy_given_x, x_pde, out=np.zeros_like(buy_given_x), where=x_pde!=0)     # divide by probability of getting a sample at X
        # score the performance on the test data
        

    def plot_likelihood(self, df, names_to_use):
        """This function plots a surface using columns of a dataframe and labels them nicely. Note that it does not call plt.show()
        names 1,2,3 are x,y,z respectively"""
        
        name1, name2, name3 = names_to_use

        data_positions = df[[name1, name2]].values.T          # Get the data. 
        x_kernel = stats.gaussian_kde(data_positions)                   # create and fit the kernel obj

        # Get the likelihood kernel for the "buy" samples given "X"
        should_buy_vals = df[df['Labels'] == 1][[name1, name2]].values.T    # Get a numpy array and transpose it so data is in rows
        x_given_buy_kernel = stats.gaussian_kde(should_buy_vals)

        # Get the likelihood kernel for the "sell" samples given "X"
        should_sell_vals = df[df['Labels'] == -1][[name1, name2]].values.T    # Get a numpy array and transpose it so data is in rows
        x_given_sell_kernel = stats.gaussian_kde(should_sell_vals)

        # Calculate the mins and maxs for bounding the plot and generating positions to plot at
        mins = []
        maxs = []
        for name in [name1, name2]:
            mins.append(df[name].min(axis=0))
            maxs.append(df[name].max(axis=0))

        # Generate grid points
        x, y = np.mgrid[mins[0]:maxs[0]:100j, mins[1]:maxs[1]:100j]       # Generate a grid
        positions = np.vstack([x.ravel(), y.ravel()])         # Convert the positions of the grid into a 2d array with 2 rows, X and Y

        # Get the kernels that should have been made earlier using the do_analysis method
        # x_kernel = self.kernels['x']
        # x_given_buy_kernel = self.kernels['x given buy']
        # x_given_sell_kernel = self.kernels['x given sell']

        # Evaluate the kernels at positions
        x_pde = x_kernel(positions).T
        x_given_buy = x_given_buy_kernel(positions).T
        x_given_sell = x_given_sell_kernel(positions).T
        # x_pde = x_kernel(data_positions).T
        # x_given_buy = x_given_buy_kernel(data_positions).T
        # x_given_sell = x_given_sell_kernel(data_positions).T

        # Use bayes theorem to calculate the liklihood
        buy_given_x = np.multiply(x_given_buy, self.prob_of_buy_signal)                                      # multiply by odds of getting buy sig
        buy_given_x = np.divide(buy_given_x, x_pde, out=np.zeros_like(buy_given_x), where=x_pde!=0)     # divide by probability of getting a sample at X

        # buy_given_x = np.reshape(buy_given_x, x.shape)

        buy_given_x = np.where(x_pde > 5/10**5, buy_given_x, 0)     # Calibrated threshhold very roughly. The kde tends to overfit very low probability data points

        sell_given_x = np.multiply(x_given_sell, self.prob_of_sell_signal)                                   # multiply by odds of getting sell sig
        sell_given_x = np.divide(sell_given_x, x_pde, out=np.zeros_like(sell_given_x), where=x_pde!=0)  # divide by probability of getting a sample at X
        # sell_given_x = np.reshape(sell_given_x, x.shape)

        # Create the image
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #Label the image
        # fig.suptitle('Feature and price comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel(name1)
        ax.set_ylabel(name2)
        ax.set_zlabel(name3)
        
        # print(f'data position shape: {data_positions.shape}, buy given x: {buy_given_x.shape}, filtered: {filter_array.shape}')
        # Plot the surface
        # Grid currently has severe issue where a corner is way too high
        surf = ax.plot_trisurf(positions[0,:], positions[1,:], buy_given_x,  cmap='viridis', antialiased=False)

        # surf = ax.scatter(positions[0,:], positions[1,:], buy_given_x, c=buy_given_x, cmap='hot')
        # surf = ax.scatter(data_positions[0,:], data_positions[1,:], buy_given_x, c=buy_given_x, cmap='hot')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)    


    def plot_2d(self, df, names, label_name):
        # 2 D PLOTTING
        fig, ax = plt.subplots()
        fig.suptitle(' Feature and Future Price Correlation', fontsize=14, fontweight='bold')
        up = df[df[label_name] == 1][names]
        down = df[df[label_name] == -1][names]

        up.plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'g')
        down.plot(x= feat1_name, y=feat2_name, ax=ax, kind = 'scatter', color = 'r')

        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))


if __name__ == "__main__":
    from environments.environments import SimulatedCryptoExchange

    start = datetime(2019, 6, 15)
    end = datetime.now()

    features = {    # 'sign': ['Close', 'Volume'],
        # 'EMA': [50, 80, 130],
        'OBV': [],
        'RSI': [],
        'BollingerBands': [1, 3],
        'BBInd': [],
        'BBWidth': [],
        'discrete_derivative': ['BBWidth3'],
        # 'time of day': [],
        # 'stack': [5]
        }
    
    sim_env = SimulatedCryptoExchange(start, end, granularity=5, feature_dict=features)
    df = sim_env.df.copy()
    
    feat1_name = 'BBInd3'
    # feat1_name = 'ddt_BBWidth3'
    feat2_name = 'BBWidth3'       # 'ddt_BBWidth3'
    feat3_name = 'BBInd3'     # To be used later

    names = [feat1_name, feat2_name]
    
    my_stats = statisticsBoy()
    df = my_stats.make_criteria(df)
    # my_stats.plot_make_criteria(df)
    # my_stats.analyze_make_criteria(df)
    my_stats.do_analysis(df, names)
    df = my_stats.build_feature(df, names)
    # df = my_stats.analyze_probability_threshold(df)

    # my_stats.plot_2d(df,names, 'Filtered label')
    my_stats.plot_likelihood(df, [*names, 'Likelihood of buy given x'])

    plt.show()