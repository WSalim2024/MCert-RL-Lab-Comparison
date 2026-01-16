import matplotlib.pyplot as plt
import pandas as pd
from environment import GridEnvironment
from q_learning import run_q_learning
from policy_gradient import run_policy_gradient


def main():
    # 1. Setup
    print("--- 🤖 RL Lab: Command Line Experiment ---")
    env = GridEnvironment()
    episodes = 500

    # 2. Run Q-Learning
    print(f"\n[1/2] Running Q-Learning ({episodes} episodes)...")
    q_rewards = run_q_learning(env, episodes=episodes)

    # 3. Run Policy Gradient
    print(f"\n[2/2] Running Policy Gradient ({episodes} episodes)...")
    # Note: Policy Gradient might need a restart if it gets stuck in the first few tries
    pg_rewards = run_policy_gradient(env, episodes=episodes, alpha=0.01, gamma=0.9)

    # 4. Process Data (Moving Average for smooth plots)
    window = 25
    q_smooth = pd.Series(q_rewards).rolling(window=window).mean()
    pg_smooth = pd.Series(pg_rewards).rolling(window=window).mean()

    # 5. Plot and Save
    print("\n[3/3] Generating Results Graph...")
    plt.figure(figsize=(10, 5))
    plt.plot(q_smooth, label="Q-Learning", color="teal")
    plt.plot(pg_smooth, label="Policy Gradient", color="purple")

    plt.title("RL Algorithm Comparison (Cumulative Rewards)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the file instead of showing it
    filename = "rl_comparison_results.png"
    plt.savefig(filename)
    print(f"✅ Success! Graph saved as '{filename}'")


if __name__ == "__main__":
    main()