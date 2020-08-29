import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def g(z):   #confirmed g works
    """This is the sigmoid function"""
    return 1/(1 + np.exp(-z))


def J(thetas, xs, ys):
    """This is the cost function"""
    J = 0
    dJ = np.zeros_like(xs[0])           #this is a vector
    ddJ = np.zeros_like(xs[0:3])        #this is H, the hessian
    #loop over all samples to sum the cost. Probably a shorter way, but this is easier to write and debug
    for i, x in enumerate(xs):
        y = ys[i]
        z = np.dot(thetas,y*x)          #confirmed y*x works, confirmed  whole thing works

        #confirmed that += works element wise
        J += np.log(g(z))
        dJ += (1 - g(z))*y*x            #confirmed this works

        #easier to loop to create H
        for j in range(3):              #loop over the x index
            for k in range(3):
                ddJ[j, k] += g(z)*(g(z) - 1)*x[j]*x[k]

    J = -J/99 #confirmed this works, and works element wise
    dJ = -dJ/99
    ddJ = -ddJ/99
    return J, dJ, ddJ


# xs and ys should be separate dataframes
#date range to train on
start = datetime(2020, 1, 1)
end = datetime(2020, 1, 30) #- timedelta(days = 1)
features = {  # 'sign': ['Close', 'Volume'],
    # 'EMA': [],
    'OBV': [],
    'RSI': [],
    # 'high': [],
    # 'low': [],
    'BollingerBands': [3, 5],
    'BBInd': [],
    'BBWidth': [],
    'discrete_derivative': ['BBWidth3'],
    # 'time of day': [],
    # 'stack': [4]
    }
sim_env = SimulatedCryptoExchange(granularity=5, feature_dict=features)

df = sim_env.df.copy()              # This is the dataframe with all of the features built.


#Get some information about our data
n_steps, n_features = xs.shape
print(f'There are {n_steps} samples in the dataset.')

#Format the df a little
xs.rename({0: 'X1', 1: 'X2'}, axis = 1, inplace = True)
xs['X0'] = np.ones(n_steps)
xs = xs[['X0', 'X1', 'X2']]

ys.rename({0:'Y'}, axis = 1, inplace = True)
df = pd.concat([xs, ys], axis = 1, sort = True)                     #did this just for nice printing of info, and for plotting
print('\nALL DATA:')
print(df.head(3))
print(df.tail(3))
print('')

#convert xs, ys into numpy arrays
xs = xs.values
ys = ys.values

thetas = np.zeros(n_features + 1)                                   #n_features + the intercept term

cost_log = []
min_change = .0001
change = 1                                                          #initialize at greater than min change
n_iterations = 0
while change > min_change:                                          #Loop to converge the model
    cost, d_cost, H = J(thetas, xs, ys)                             #get the costs
    cost_log.append(cost)                                           #Log the cost
    delta_theta = np.matmul(np.linalg.inv(H), d_cost)               #alter the weights according to Newtons method
    changes = abs(delta_theta)
    change = changes.max()
    thetas -= delta_theta

    print('Cost: ', end = ' ')
    print(round(cost,2), end = ' ')
    print('Thetas: ', end = ' ')
    print(thetas, end = ' ')
    print('dCost: ', end = ' ')
    print(d_cost)
    # print('H: ', end = ' ')
    # print(H)
    n_iterations += 1
    if n_iterations > 80:
        print('Reached maximum numbers of iterations')
        break

print('')
print(f'The coefficients theta after training are {thetas}')
fig, ax = plt.subplots(1,1)
fig.suptitle('Training Data and Decision Boundary', fontsize=14, fontweight='bold')
df[df['Y'] == 1].plot(x = 'X1', y = 'X2', ax = ax, kind = 'scatter', color = 'b', marker = '+')
df[df['Y'] == -1].plot(x = 'X1', y = 'X2', ax = ax, kind = 'scatter', color = 'r', marker = 'x')
ax.legend(['Y = 1', 'Y = -1'])

# print(df['X1'].min())
x1s = np.linspace(round(df['X1'].min()) - 1, round(df['X1'].max()), 8 + 1)
x2s = -(thetas[1]*x1s + thetas[0])/thetas[2]    #y = mx + b

plt.plot(x1s, x2s, 'g')

#Plot the cost over time
fig = plt.figure()
fig.suptitle('Cost During Training', fontsize=14, fontweight='bold')
plt.plot(cost_log)
plt.show()
