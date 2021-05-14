import random
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, stock_data: pd.DataFrame):

        super(StockTradingEnv, self).__init__()

        self.stock_data = stock_data
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([3, 1]),
            dtype=np.float16
        )

        # Data contains the normalized (Open, High, Low, Close, Volume) values for the last five prices + agent info row
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(6, 6),
            dtype=np.float16
        )

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array(
            [
                self.stock_data.loc[self.current_step: self.current_step + 5, 'Open'].values / MAX_SHARE_PRICE,
                self.stock_data.loc[self.current_step: self.current_step + 5, 'High'].values / MAX_SHARE_PRICE,
                self.stock_data.loc[self.current_step: self.current_step + 5, 'Low'].values / MAX_SHARE_PRICE,
                self.stock_data.loc[self.current_step: self.current_step + 5, 'Close'].values / MAX_SHARE_PRICE,
                self.stock_data.loc[self.current_step: self.current_step + 5, 'Volume'].values / MAX_NUM_SHARES,
            ]
        )

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame,
                        [
                            [
                            self.balance / MAX_ACCOUNT_BALANCE,
                            self.max_net_worth / MAX_ACCOUNT_BALANCE,
                            self.shares_held / MAX_NUM_SHARES,
                            self.cost_basis / MAX_SHARE_PRICE,
                            self.total_shares_sold / MAX_NUM_SHARES,
                            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
                            ]
                        ],
                        axis=0
        )

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.stock_data.loc[self.current_step, "Open"],
            self.stock_data.loc[self.current_step, "Close"]
        )

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            additional_cost = shares_bought * current_price
            self.balance -= additional_cost

            prev_cost = self.cost_basis * self.shares_held
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):

        self._take_action(action)
        self.current_step += 1

        if self.current_step > len(self.stock_data.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, len(self.stock_data.loc[:, 'Open'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print('Step: {}'.format(self.current_step))
        print('Balance: {}'.format(self.balance))
        print('Shares held: {}'.format(self.shares_held))
        print('Total sold: {}'.format(self.total_shares_sold))
        print('Avg cost for held shares: {}'.format(self.cost_basis))
        print('Total sales value: {}'.format(self.total_sales_value))
        print('Net worth: {}'.format(self.net_worth))
        print('Max net worth: {}'.format(self.max_net_worth))
        print('Profit: {}'.format(profit))
