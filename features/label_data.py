from tools.tools import percent_change_column
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def make_criteria(input_df,  target_change=.15, allowable_reversal=-.15, target_time_range=13, buy_not_sell=True):
        # This is currently symmetric (prediction mirrored conditions for buy/sell signals)
        df = input_df.copy()

        #Loop to create percentage columns to look into the future
        generated_names = []
        for i in range(target_time_range):
            df, label_name = percent_change_column('BTCClose', df, -(i+1))
            generated_names.append(label_name)

        df.dropna(inplace=True)

        # THIS IS WHERE THE CONDITIONS FOR BUYING ARE DEFINED
        # Calculate some basic parameters around take profit and downdraw (I may be using that word wrong?)
        df['Max increase'] = df[generated_names].max(axis=1)
        df['Max decrease'] = df[generated_names].min(axis=1)
        if buy_not_sell:
            df['Buy Labels'] = np.zeros(df.shape[0])      # Initialize as all samples as "not suitable for buying"
            df.loc[(df['Max increase'] > target_change) & (df['Max decrease'] > allowable_reversal), 'Buy Labels'] = 1        # Predicted its going up
        else:
            df['Sell Labels'] = np.zeros(df.shape[0])      # Initialize as all samples as "not suitable for selling"
            df.loc[(df['Max increase'] < -allowable_reversal) & (df['Max decrease'] < -target_change), 'Sell Labels'] = 1     # Predicted its going down

        # Drop the temporary columns
        df.drop(columns=['Max increase', 'Max decrease', *generated_names], inplace=True)
        return df


def plot_make_criteria(input_df):
    df = input_df.copy()
    fig, ax = plt.subplots(1, 1)

    input_df.plot(y='BTCClose', ax=ax)  
    # print(df.head())
    # print(df.reset_index().head())
    for name in ['Buy Labels', 'Sell Labels']:
        data = input_df[input_df[name] == 1]
        
        # Enforce different colors for buys and sells
        if 'buy' in name.lower():
            data.reset_index().plot(y = 'BTCClose', x='index', ax=ax, kind='scatter', marker='^', c='g', zorder=4)
        elif 'sell' in name.lower():
            data.reset_index().plot(y = 'BTCClose', x='index', ax=ax, kind='scatter', marker='v', c='r', zorder=4)
        fig.autofmt_xdate()
        fig.suptitle(f'Make criteria', fontsize=14, fontweight='bold')


# def build_filtered_label(self, input_df, names):
#     # For now, not doing train/test split, but this function should do that, use cross validation, and then rebuild all. I think
#     df = input_df.copy()

#     # Calculate the probabilities for each sample (evaluation of training data)
    
#     # Get the kernels that should have been made earlier
#     x_kernel = self.kernels['x']
#     x_given_buy_kernel = self.kernels['x given buy']
#     x_given_sell_kernel = self.kernels['x given sell']
    
#     # The positions in the feature space at which to evaluate the kernels
#     positions = df[names].values.T

#     # Evaluate the kernels at positions
#     x_pde = x_kernel(positions).T
#     x_given_buy = x_given_buy_kernel(positions).T
#     x_given_sell = x_given_sell_kernel(positions).T

#     # Use bayes theorem to calculate the liklihood
#     buy_given_x = np.multiply(x_given_buy, self.prob_of_buy_signal)                                      # multiply by odds of getting buy sig
#     buy_given_x = np.divide(buy_given_x, x_pde, out=np.zeros_like(buy_given_x), where=x_pde!=0)     # divide by probability of getting a sample at X
    
#     sell_given_x = np.multiply(x_given_sell, self.prob_of_sell_signal)                                   # multiply by odds of getting sell sig
#     sell_given_x = np.divide(sell_given_x, x_pde, out=np.zeros_like(sell_given_x), where=x_pde!=0)  # divide by probability of getting a sample at X

#     # Build columns in the dataframe (mostly used for plotting? And also filtering I think)
#     df['Likelihood of buy given x'] = buy_given_x
#     df['Likelihood of sell given x'] = sell_given_x

#     # This worked with >70% untuned on training data when relative prob was difference. Thresh = .005
#     thresh = 0.65
#     # df['Prediction'] = np.zeros(df.shape[0])
#     # df.loc[(df['Likelihood of buy given x'] >= thresh), 'Prediction'] = 1
#     # df.loc[(df['Likelihood of sell given x'] >= thresh), 'Prediction'] = -1
#     # df.loc[(df['Likelihood of sell given x'] >= thresh), (df['Likelihood of buy given x'] >= thresh), 'Prediction'] = 0         # This should never happen, its here just in case

#     n_buy_sigs = df.loc[(df['Likelihood of buy given x'] >= thresh)].shape[0]
#     mean_buy_freq = (24*60/5)*n_buy_sigs/df.shape[0]        # signals per sample *(1/ 5 minutes/sample)*60*24 minutes/day = signals/day
#     print(f'Mean of {mean_buy_freq} buy signals per day')
#     n_sell_sigs = df.loc[(df['Likelihood of sell given x'] >= thresh)].shape[0]
#     mean_buy_freq = (24*60/5)*n_sell_sigs/df.shape[0]        # signals per sample *(1/ 5 minutes/sample)*60*24 minutes/day = signals/day
#     print(f'Mean of {mean_buy_freq} sell signals per day')

#     # Drop unwanted columns
#     # cols = ['Likelihood of buy given x', 'Likelihood of sell given x']
#     # df.drop(columns=cols, inplace=True)
#     return df

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True,)
    # args = parser.parse_args()
    # name = args.name
    from environments.environments import SimulatedCryptoExchange
    from datetime import datetime
    #date range to train on
    start = datetime(2020, 8, 1)
    end = datetime(2020, 9, 1) #- timedelta(days = 1)
    features = {  # 'sign': ['Close', 'Volume'],
        
        }
    sim_env = SimulatedCryptoExchange(granularity=5, feature_dict=features)

    df = sim_env.df.copy()              # This is the dataframe with all of the features built.

    df = make_criteria(df)
    df = make_criteria(df, buy_not_sell=False)
    plot_make_criteria(df)
    plt.show()

# if label_type == 'standard':
#             """Builds labels (0, 1) or % depending on predictor type as a column on the dataframe, return np arrays for X and y"""
#             df = input_df.copy()
#             token = 'BTC'

#             df = df[np.isfinite(df).all(1)]     # keep only finite numbers
#             # print(df.isnull().values.any())

#             # These do time differencing for data stationarity
#             price_name = token + 'Close'
#             vol_name = token + 'Volume'
#             df = percent_change_column(price_name, df)
#             df = percent_change_column(vol_name, df)
            
#             if self.is_classifier:
#                 thresh = 0.1
#                 #Calculate what the next price will be
#                 all_labels = np.empty([df.shape[0],self.n_predictions])
#                 y_name = (f'Period {str(self.n_predictions)} Total % Change')
#                 for i in range(self.n_predictions):
#                     step = (i+1)
#                     series = df['BTCClose'].shift(-step)
#                     all_labels = np.hstack((all_labels, np.atleast_2d(series).T)) 

#                 # Take the mean accross the prices
#                 summed_labels = np.sum(all_labels, axis = 1)

#                 labels = np.empty_like(summed_labels)
#                 for j, item in enumerate(summed_labels):
#                     if item > thresh:
#                         labels[j] = 1
#                     else:
#                         labels[j] = 0
#                 df[y_name] = labels
#             else:
#                 #Calculate what the next price will be
#                 for i in range(self.n_predictions):
#                     step = (i+1)  # *-1
#                     df, y_name = percent_change_column('BTCClose', df, -step)

#             # print('Training data:')
#             # print(df.head(30))
#             features_to_use = list(df.columns)
#             # Remove the y labels from the 'features to use'
#             features_to_use.remove(y_name)
                
#             # Make sure you aren't building a classifier based on the outputs of other classifiers
#             classifier_names = ['knn', 'mlp', 'svc', 'ridge']
#             for item in features_to_use:
#                 for name in classifier_names:
#                     if name in item:
#                         features_to_use.remove(item)

#             df = df[np.isfinite(df).all(1)]     # keep only finite numbers
#             print('Training data sample:')
#             print(df.head(30))
#             X = df[features_to_use].values
#             y = df[y_name].values
#             return X, y

#         elif label_type == 'mean in time period':
#             """Builds labels (0, 1) or % depending on predictor type as a column on the dataframe, return np arrays for X and y"""
#             df = input_df.copy()
#             token = 'BTC'

#             df = df[np.isfinite(df).all(1)]     # keep only finite numbers
#             # print(df.isnull().values.any())

#             # These do time differencing for data stationarity
#             price_name = token + 'Close'
#             vol_name = token + 'Volume'
#             df = percent_change_column(price_name, df)
#             df = percent_change_column(vol_name, df)
            
#             if self.is_classifier:
#                 thresh = 0.15
#                 # Calculate what the next price will be. 
#                 # Create a matrix where each columns is the price a timestep in the future
#                 all_labels = np.empty([df.shape[0],self.n_predictions])
#                 y_name = (f'Period {str(self.n_predictions)} Mean % Change')
#                 for i in range(self.n_predictions):
#                     step = (i+1)
#                     series = df['BTCClose'].shift(-step)
                    
#                     all_labels = np.hstack((all_labels, np.atleast_2d(series).T)) 

#                 # Take the mean accross the prices
#                 summed_labels = np.sum(all_labels, axis = 1)

#                 labels = np.empty_like(summed_labels)
#                 for j, item in enumerate(summed_labels):
#                     if item > thresh:
#                         labels[j] = 1
#                     else:
#                         labels[j] = 0
#                 df[y_name] = labels
#             else:
#                 #Calculate what the next price will be
#                 for i in range(self.n_predictions):
#                     step = (i+1)  # *-1
#                     df, y_name = percent_change_column('BTCClose', df, -step)

#             # print('Training data:')
#             # print(df.head(30))
#             features_to_use = list(df.columns)
#             # Remove the y labels from the 'features to use'
#             features_to_use.remove(y_name)
                
#             # Make sure you aren't building a classifier based on the outputs of other classifiers
#             classifier_names = ['knn', 'mlp', 'svc', 'ridge']
#             for item in features_to_use:
#                 for name in classifier_names:
#                     if name in item:
#                         features_to_use.remove(item)

#             df = df[np.isfinite(df).all(1)]     # keep only finite numbers
#             X = df[features_to_use].values
#             y = df[y_name].values
#             return X, y

#     def analyze_make_criteria(self, df):
#         # splits data at 80% train 80% test by default
#         all_cols = list(df)
#         earliest = df.index.min()
#         most_recent = df.index.max()

#         # calculate 80 percent
#         timerange = most_recent - earliest 
#         split_line = earliest + timerange/0.8
#         print(split_line)
#         X = df[all_cols.remove('Labels')].values
#         Y = df['Labels'].values


#         data_positions = df[[name1, name2]].values.T                    # Get the data. 

#         # fit the model to the training data
#         should_buy_vals = df[df['Labels'] == 1][[name1, name2]].values.T    # Get a numpy array and transpose it so data is in rows
#         x_given_buy_kernel = stats.gaussian_kde(should_buy_vals)
#         x_kernel = stats.gaussian_kde(data_positions)                       # create and fit the kernel obj


#         # evalute the model on the test data
#         # Use bayes theorem to calculate the liklihood
#         buy_given_x = np.multiply(x_given_buy, self.prob_of_buy_signal)                                      # multiply by odds of getting buy sig
#         buy_given_x = np.divide(buy_given_x, x_pde, out=np.zeros_like(buy_given_x), where=x_pde!=0)     # divide by probability of getting a sample at X
#         # score the performance on the test data
        