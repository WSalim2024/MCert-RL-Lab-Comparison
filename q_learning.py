import numpy as np


def run_q_learning(env, episodes=500):
    # Hyperparameters
    alpha = 0.1  # Learning Rate
    gamma = 0.9  # Discount Factor
    epsilon = 0.1  # Exploration Rate

    # Initialize Q-Table
    Q = np.zeros((env.n_states, env.n_actions))

    rewards_history = []

    print("🚀 Training Q-Learning Agent...")
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:
            # Epsilon-Greedy Strategy
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, env.n_actions)  # Explore
            else:
                action = np.argmax(Q[state, :])  # Exploit

            # Take Action
            next_state, reward, done = env.step(action)

            # Bellman Update Equation
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state
            total_reward += reward
            steps += 1

        rewards_history.append(total_reward)

    return rewards_history