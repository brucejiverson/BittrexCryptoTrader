from tools.tools import percent_change_column
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def make_criteria(input_df,  target_change=.15, allowable_reversal=-.15, target_time_range=13):
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
        df.loc[(df['Max increase'] > target_change) & (df['Max decrease'] > allowable_reversal), 'Labels'] = 1        # Predicted its going up
        df.loc[(df['Max increase'] < -allowable_reversal) & (df['Max decrease'] < -target_change), 'Labels'] = -1     # Predicted its going down
        
        # Drop the temporary columns
        df.drop(columns=['Max increase', 'Max decrease', *generated_names], inplace=True)
        return df

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

#     def plot_make_criteria(input_df, label_name):
#         df = input_df.copy()
#         fig, ax = plt.subplots(1, 1)

#         input_df.plot(y='BTCClose', ax=ax)
#         print(df.head())
#         print(df.reset_index().head())
#         buys = input_df[input_df[label_name] == 1]
#         sells = input_df[input_df[label_name] == -1]
#         buys.reset_index().plot(y = 'BTCClose', x='index', ax=ax, kind='scatter', marker='^', c='g', zorder=4)
#         sells.reset_index().plot(y = 'BTCClose', x='index', ax=ax, kind='scatter', marker='v', c='r', zorder=4)
#         fig.autofmt_xdate()
#         fig.suptitle(f'Make criteria', fontsize=14, fontweight='bold')


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
        