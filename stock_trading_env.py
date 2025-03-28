# stock_trading_env.py
import gym
from gym import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    """
    Custom Environment for stock trading.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.current_step = 0
        self.balance = 10000  # Initial balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        return self._next_observation()

    def _next_observation(self):
        frame = np.array([
            self.df.iloc[self.current_step]['Close'],
            self.df.iloc[self.current_step]['SMA'],
            self.df.iloc[self.current_step]['EMA'],
            self.df.iloc[self.current_step]['RSI'],
            self.balance,
            self.shares_held
        ])
        return frame

    def step(self, action):
        self.current_step += 1

        if self.current_step > len(self.df) - 1:
            self.current_step = 0

        reward = 0
        if action == 0:  # Buy
            self.shares_held += 1
            self.balance -= self.df.iloc[self.current_step]['Close']
        elif action == 1:  # Sell
            self.shares_held -= 1
            self.balance += self.df.iloc[self.current_step]['Close']
            self.total_shares_sold += 1
            self.total_sales_value += self.df.iloc[self.current_step]['Close']
            reward = self.df.iloc[self.current_step]['Close']
        elif action == 2:  # Hold
            pass

        done = self.balance <= 0 or self.current_step == len(self.df) - 1

        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        profit = self.balance + self.shares_held * self.df.iloc[self.current_step]['Close'] - 10000
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Total shares sold: {self.total_shares_sold}')
        print(f'Total sales value: {self.total_sales_value}')
        print(f'Profit: {profit}')