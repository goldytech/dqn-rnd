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



### `stock_trading_env.py`

This module defines a custom Gym environment for stock trading. The environment simulates trading by allowing the agent to buy, sell, or hold stocks based on the current state.



### `dqn_agent.py`

This module implements the DQN algorithm. The agent uses a neural network to approximate the Q-value function and learns from past experiences stored in a memory buffer.


### `train.py`

This module trains the DQN agent using the custom stock trading environment and the fetched stock data.



### `test.py`

This module tests the trained DQN agent using new stock data.



## Conclusion

This project demonstrates how to build a stock trading agent using Deep Q Networks. The agent is trained to make buy, sell, or hold decisions based on historical stock data. The modular structure of the project makes it easy to understand and extend.

### References
* https://www.youtube.com/watch?v=x83WmvbRa2I
* https://thecleverprogrammer.com/2025/02/18/building-an-ai-agent-using-agentic-ai/