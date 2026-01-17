import numpy as np


def run_q_learning(env, episodes=500):
    # Hyperparameters
    alpha = 0.1  # Learning Rate
    gamma = 0.9  # Discount Factor
    epsilon = 0.1  # Exploration Rate

    # Initialize Q-Table
    Q = np.zeros((env.n_states, env.n_actions))

    # NEW: Dictionary to store all metrics
    metrics = {
        'rewards': [],
        'lengths': [],
        'success_rate': [],
        'exploration_ratio': []
    }

    print("🚀 Training Q-Learning Agent with Advanced Metrics...")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        # Track decisions for this episode
        explore_count = 0
        exploit_count = 0

        while not done and steps < 100:
            # Epsilon-Greedy Strategy with Counting
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, env.n_actions)  # Explore
                explore_count += 1
            else:
                action = np.argmax(Q[state, :])  # Exploit
                exploit_count += 1

            # Take Action
            next_state, reward, done = env.step(action)

            # Bellman Update Equation
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state
            total_reward += reward
            steps += 1

        # --- LOG METRICS ---
        metrics['rewards'].append(total_reward)
        metrics['lengths'].append(steps)

        # Success = 1 if it hit the goal (+10 reward), else 0
        # Note: In environment.py, goal reward is 10.
        is_success = 1 if reward == 10 else 0
        metrics['success_rate'].append(is_success)

        # Calculate Exploration Ratio
        total_actions = explore_count + exploit_count
        ratio = explore_count / total_actions if total_actions > 0 else 0
        metrics['exploration_ratio'].append(ratio)

    return metrics