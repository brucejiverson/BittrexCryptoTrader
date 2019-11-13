import matplotlib.pyplot as plt
import numpy as np


def return_on_investment(final, initial):
    # Returns the percentage increase/decrease
    return round(final / initial - 1, 4) * 100

mode = 'test'

a = np.load(f'agent_rewards/{mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}, average roi: {return_on_investment(a.mean(), 10**7)}")

plt.hist(a, bins=30)
plt.title(mode)
plt.show()
