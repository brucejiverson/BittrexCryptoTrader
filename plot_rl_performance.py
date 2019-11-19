import matplotlib.pyplot as plt
import numpy as np


mode = 'train'

a = np.load(f'agent_rewards/{mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

plt.hist(a, bins=20)
plt.title(mode)
plt.show()
