import numpy as np
import tensorflow as tf


def run_policy_gradient(env, episodes=500, alpha=0.01, gamma=0.9, track_state=0):
    # One-hot encoding size (Input Layer)
    input_dim = env.n_states
    output_dim = env.n_actions

    # Build the Policy Network (Neural Network)
    # Input: State Vector -> Hidden Layer -> Output: Action Probabilities
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
    rewards_history = []

    # NEW: Log probabilities for the tracked state (Thinking History)
    policy_history = []

    # Helper to format state for Neural Net
    def one_hot(state):
        vec = np.zeros((1, input_dim))
        vec[0, state] = 1
        return vec

    # print("🧠 Training Policy Gradient Agent...") # Commented out for cleaner UI logs

    for episode in range(episodes):
        state = env.reset()
        done = False

        # Memory for this episode
        states_mem = []
        actions_mem = []
        rewards_mem = []
        steps = 0

        while not done and steps < 100:
            # 1. Get Action Probabilities from Neural Net
            state_in = one_hot(state)
            probs = model(state_in).numpy()[0]

            # 2. Sample an action based on probabilities
            action = np.random.choice(env.n_actions, p=probs)

            # 3. Take Action
            next_state, reward, done = env.step(action)

            # 4. Store memory
            states_mem.append(state_in[0])
            actions_mem.append(action)
            rewards_mem.append(reward)

            state = next_state
            steps += 1

        # --- END OF EPISODE: TRAIN THE NETWORK (REINFORCE Algorithm) ---

        # Calculate Cumulative Discounted Rewards (Return Gt)
        cumulative_rewards = np.zeros_like(rewards_mem, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards_mem))):
            running_add = running_add * gamma + rewards_mem[t]
            cumulative_rewards[t] = running_add

        # Normalize returns (stabilizes training)
        if len(cumulative_rewards) > 1:
            cumulative_rewards = (cumulative_rewards - np.mean(cumulative_rewards)) / (
                        np.std(cumulative_rewards) + 1e-8)

        # Gradient Descent Step
        with tf.GradientTape() as tape:
            all_states = np.array(states_mem)
            all_probs = model(all_states)

            indices = tf.range(len(actions_mem))
            action_indices = tf.stack([indices, actions_mem], axis=1)
            picked_action_probs = tf.gather_nd(all_probs, action_indices)

            loss = -tf.reduce_mean(tf.math.log(picked_action_probs) * cumulative_rewards)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        rewards_history.append(sum(rewards_mem))

        # NEW: Spy on the brain! Record probabilities for the tracked state
        tracked_state_in = one_hot(track_state)
        tracked_probs = model(tracked_state_in).numpy()[0]
        policy_history.append(tracked_probs)

    return rewards_history, np.array(policy_history)