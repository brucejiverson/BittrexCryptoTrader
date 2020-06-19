import matplotlib.pyplot as plt
import numpy as np
import argparse


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--mode', type=str, required=True,
    #                     help='either "train" or "test"')
    # args = parser.parse_args()
    mode = 'train'
    rewards = np.load(f'agent_rewards/{mode}.npy')

    print(f"average reward: {rewards.mean():.2f}, min: {rewards.min():.2f}, max: {rewards.max():.2f}")

    a_sma = []

    p = 50

    for i, item in enumerate(rewards):
        if i < p:
            a_sma.append(rewards[0:i].mean())
        else:
            a_sma.append(rewards[i - p:i].mean())

    # plt.hist(rewards, bins=20)
    plt.plot(rewards)
    plt.plot(a_sma)
    plt.title(mode)
    plt.show()
