import gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
alpha = 0.8      # learning rate
gamma = 0.95     # discount factor
epsilon = 0.1    # exploration rate
episodes = 1000

for episode in range(episodes):
    state = env.reset()[0]
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(q_table[state])  # exploit

        next_state, reward, done, truncated, _ = env.step(action)

        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state

print("Training completed.\n")
print("Q-table:")
print(q_table)
