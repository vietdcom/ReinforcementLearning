import numpy as np
import matplotlib.pyplot as plt

# Thiết lập môi trường lưới 5x5
size = 5
goal_state = (4, 4)
obstacles = [(1,1), (2,2), (3,1)]
actions = ['up', 'down', 'left', 'right']
q_table = np.zeros((size, size, len(actions)))

def get_next_state(state, action):
    i, j = state
    if action == 'up': i = max(i-1, 0)
    elif action == 'down': i = min(i+1, size-1)
    elif action == 'left': j = max(j-1, 0)
    elif action == 'right': j = min(j+1, size-1)
    return (i, j)

def get_reward(state):
    if state == goal_state:
        return 100
    elif state in obstacles:
        return -100
    else:
        return -1

# Q-learning hyperparameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount
epsilon = 0.2    # exploration
episodes = 1000

for ep in range(episodes):
    state = (0, 0)
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action_index = np.random.randint(4)
        else:
            action_index = np.argmax(q_table[state[0], state[1]])

        action = actions[action_index]
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)

        best_next = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action_index] += alpha * (
            reward + gamma * best_next - q_table[state[0], state[1], action_index]
        )

        if next_state == goal_state or next_state in obstacles:
            done = True

        state = next_state

print("🏁 Q-learning hoàn tất!")

# 👣 In đường đi tốt nhất từ (0,0) → (4,4)
state = (0, 0)
path = [state]
while state != goal_state:
    action_index = np.argmax(q_table[state[0], state[1]])
    state = get_next_state(state, actions[action_index])
    path.append(state)
    if len(path) > 50: break

print("Path learned:", path)

# 🧭 Vẽ lưới + đường đi
grid = np.zeros((size, size))
for obs in obstacles:
    grid[obs] = -10
grid[goal_state] = 10

for pos in path:
    grid[pos] = 5

plt.imshow(grid, cmap='coolwarm')
plt.title("Robot Path to Goal")
plt.colorbar()
plt.show()
