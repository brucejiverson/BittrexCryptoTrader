import matplotlib.pyplot as plt
import numpy as np
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    args = parser.parse_args()
    mode = args.mode
    a = np.load(f'agent_rewards/{mode}.npy')

    print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

    a_filtered = []
    print(type(a))

    p = 60

    for i, item in enumerate(a):
        if i < p:
            a_filtered.append(a[0:i].mean())
        else:
            a_filtered.append(a[i - p:i].mean())

    # plt.hist(a, bins=20)
    plt.plot(a)
    plt.plot(a_filtered)
    plt.title(mode)
    plt.show()
