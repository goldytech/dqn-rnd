# dqn_agent.py
import numpy as np

import random

from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
# https://www.youtube.com/watch?v=x83WmvbRa2I 
# In the context of Deep Q Networks (DQN), `epsilon` and `gamma` are important hyperparameters that influence the learning process of the agent.
# 
# 1. **Epsilon (ε)**: This parameter is used in the epsilon-greedy policy to balance exploration and exploitation.
# - **Exploration**: The agent tries out new actions to discover their effects, which helps in learning better policies.
# - **Exploitation**: The agent uses the knowledge it has already gained to maximize the reward.
# - **Epsilon-greedy policy**: With probability `epsilon`, the agent chooses a random action (exploration), and with probability `1 - epsilon`, it chooses the best-known action (exploitation).
# - **Decay**: `epsilon` typically starts high (e.g., 1.0) and decays over time (e.g., `epsilon_decay = 0.995`) to reduce exploration as the agent learns more about the environment.
# 
# 2. **Gamma (γ)**: This is the discount factor used in the Q-learning algorithm.
# - **Purpose**: It determines the importance of future rewards. A value of `gamma` close to 1 (e.g., 0.95) means that future rewards are considered almost as important as immediate rewards.
# - **Effect**: A higher `gamma` encourages the agent to consider long-term rewards, while a lower `gamma` makes the agent focus more on immediate rewards.

# These parameters are crucial for the balance and efficiency of the learning process in reinforcement learning.

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
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