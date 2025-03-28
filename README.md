# Stock Trading Agent using Deep Q Networks (DQN)

## Overview

This project implements a stock trading agent using Deep Q Networks (DQN). The agent is trained to make buy, sell, or hold decisions based on historical stock data. The project is modular and written in a pythonic way for ease of understanding and maintenance.

## Conceptual Understanding

### Reinforcement Learning

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. The agent interacts with the environment in discrete time steps. At each time step, the agent:
1. Observes the current state.
2. Chooses an action based on a policy.
3. Receives a reward from the environment.
4. Transitions to a new state.

The goal of the agent is to learn a policy that maximizes the expected cumulative reward over time.

### Deep Q Networks (DQN)

Deep Q Networks (DQN) is a type of RL algorithm that combines Q-Learning with deep neural networks. Q-Learning is a value-based method where the agent learns the value of taking a particular action in a particular state. The Q-value is updated using the Bellman equation:

\[ Q(s, a) = r + \gamma \max_{a'} Q(s', a') \]

Where:
- \( Q(s, a) \) is the Q-value for state \( s \) and action \( a \).
- \( r \) is the reward received after taking action \( a \) in state \( s \).
- \( \gamma \) is the discount factor for future rewards.
- \( s' \) is the next state.
- \( a' \) is the next action.

In DQN, a neural network is used to approximate the Q-value function. The network is trained using experience replay, where past experiences are stored in a memory buffer and sampled randomly to break the correlation between consecutive experiences.

## Project Structure

The project consists of the following modules:

1. `data_fetcher.py`: Fetches stock data from Yahoo Finance API and calculates technical indicators.
2. `stock_trading_env.py`: Defines the custom Gym environment for stock trading.
3. `dqn_agent.py`: Implements the DQN algorithm.
4. `train.py`: Trains the DQN agent.
5. `test.py`: Tests the trained DQN agent.

### Module Relationships

- `data_fetcher.py`: Fetches and processes stock data.
- `stock_trading_env.py`: Uses the processed data to create a trading environment.
- `dqn_agent.py`: Interacts with the environment to learn the trading policy.
- `train.py`: Coordinates the training process.
- `test.py`: Evaluates the performance of the trained agent.

## Detailed Explanation

### `data_fetcher.py`

This module fetches historical stock data from Yahoo Finance API and calculates technical indicators such as Simple Moving Average (SMA), Exponential Moving Average (EMA), and Relative Strength Index (RSI).

```python
import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, start_date, end_date):
    # Fetch stock data for multiple tickers
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data[ticker] = stock.history(start=start_date, end=end_date)
    return data

def fetch_technical_data(data):
    # Calculate technical indicators
    data['SMA'] = data['Close'].rolling(window=20).mean()
    data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(window=14).mean()))
    return data

def fetch_fundamental_data(ticker):
    # Fetch fundamental data
    stock = yf.Ticker(ticker)
    fundamentals = stock.financials.T
    return fundamentals
```

### `stock_trading_env.py`

This module defines a custom Gym environment for stock trading. The environment simulates trading by allowing the agent to buy, sell, or hold stocks based on the current state.

```python
import gym
from gym import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.action_space = spaces.Discrete(3)
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
        if action == 0:
            self.shares_held += 1
            self.balance -= self.df.iloc[self.current_step]['Close']
        elif action == 1:
            self.shares_held -= 1
            self.balance += self.df.iloc[self.current_step]['Close']
            self.total_shares_sold += 1
            self.total_sales_value += self.df.iloc[self.current_step]['Close']
            reward = self.df.iloc[self.current_step]['Close']
        elif action == 2:
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
```

### `dqn_agent.py`

This module implements the DQN algorithm. The agent uses a neural network to approximate the Q-value function and learns from past experiences stored in a memory buffer.

```python
import numpy as np
import random
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
```

### `train.py`

This module trains the DQN agent using the custom stock trading environment and the fetched stock data.

```python
import numpy as np
from stock_trading_env import StockTradingEnv
from dqn_agent import DQNAgent
from data_fetcher import fetch_stock_data, fetch_technical_data

if __name__ == "__main__":
    tickers = ["AAPL"]
    start_date = "2024-01-01"
    end_date = "2024-11-01"
    
    data = fetch_stock_data(tickers, start_date, end_date)
    technical_data = {ticker: fetch_technical_data(data[ticker]) for ticker in tickers}
    
    envs = {ticker: StockTradingEnv(technical_data[ticker]) for ticker in tickers}
    state_size = envs[tickers[0]].observation_space.shape[0]
    action_size = envs[tickers[0]].action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000

    for e in range(episodes):
        for ticker in tickers:
            env = envs[ticker]
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            for time in range(500):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"episode: {e}/{episodes}, ticker: {ticker}, score: {time}, e: {agent.epsilon:.2}")
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
        if e % 10 == 0:
            agent.save(f"model_{e}.weights.h5")
```

### `test.py`

This module tests the trained DQN agent using new stock data.

```python
import numpy as np
from stock_trading_env import StockTradingEnv
from dqn_agent import DQNAgent
from data_fetcher import fetch_stock_data, fetch_technical_data

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2021-01-01"
    end_date = "2021-12-31"

    data = fetch_stock_data(ticker, start_date, end_date)
    data = fetch_technical_data(data)

    env = StockTradingEnv(data)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("model_990.h5")

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = np.reshape(next_state, [1, state_size])
        env.render()
        if done:
            break
```

## Conclusion

This project demonstrates how to build a stock trading agent using Deep Q Networks. The agent is trained to make buy, sell, or hold decisions based on historical stock data. The modular structure of the project makes it easy to understand and extend.

### References
https://www.youtube.com/watch?v=x83WmvbRa2I
https://thecleverprogrammer.com/2025/02/18/building-an-ai-agent-using-agentic-ai/