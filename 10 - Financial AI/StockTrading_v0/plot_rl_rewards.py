import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True, help='eihter "train" or "test"')

args = parser.parse_args()

data = np.load(f'linear_rl_trader_rewards/{args.mode}.npy')
print("Average reward: {:.2f}, min {:.2f}, max: {:.2f}".format(data.mean(), data.min(), data.max()))

plt.hist(data, bins=20)
plt.title(args.mode)
plt.show()
