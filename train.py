# train.py
import numpy as np
from stock_trading_env import StockTradingEnv
from dqn_agent import DQNAgent
from data_fetcher import fetch_stock_data, fetch_technical_data, fetch_fundamental_data

if __name__ == "__main__":
    tickers = ["AAPL"]  # List of tickers
    start_date = "2024-01-01"
    end_date = "2024-11-01"
    
    data = fetch_stock_data(tickers, start_date, end_date)
    technical_data = {ticker: fetch_technical_data(data[ticker]) for ticker in tickers}
    fundamental_data = fetch_fundamental_data(tickers)
    
    envs = {ticker: StockTradingEnv(technical_data[ticker], fundamental_data[ticker]) for ticker in tickers}
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


