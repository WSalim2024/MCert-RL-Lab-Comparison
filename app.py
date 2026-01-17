import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from environment import GridEnvironment
from q_learning import run_q_learning
from policy_gradient import run_policy_gradient

# --- PAGE CONFIG ---
st.set_page_config(page_title="RL Lab: Algorithm Arena", page_icon="🤖", layout="wide")

st.title("🤖 RL Lab: Q-Learning vs. Policy Gradients")
st.markdown("""
Compare two fundamental Reinforcement Learning approaches on a **5x5 Grid World**.
""")

# --- SIDEBAR: HYPERPARAMETERS ---
st.sidebar.header("⚙️ Configuration")
episodes = st.sidebar.slider("Training Episodes", 100, 1000, 500, step=100)
gamma = st.sidebar.slider("Discount Factor (Gamma)", 0.5, 0.99, 0.9)
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")

# Brain Scanner Selection
st.sidebar.markdown("---")
st.sidebar.header("🧠 Brain Scanner")
spy_state = st.sidebar.selectbox("Select State to Monitor",
                                 options=[0, 12, 24],
                                 format_func=lambda x: f"State {x} (Start)" if x == 0 else (
                                     f"State {x} (Pit)" if x == 12 else f"State {x} (Goal)"))

if st.sidebar.button("🚀 Start Training Race"):

    env = GridEnvironment()

    # Create Columns for progress
    col1, col2 = st.columns(2)

    # --- 1. RUN Q-LEARNING (Value-Based) ---
    with col1:
        st.subheader("1. Q-Learning")
        with st.spinner("Training Q-Agent..."):
            # Returns a DICTIONARY now
            q_metrics = run_q_learning(env, episodes=episodes)
            q_rewards = q_metrics['rewards']

        st.success("Q-Learning Complete!")
        st.metric("Avg Final Reward", f"{sum(q_rewards[-50:]) / 50:.2f}")

        # NEW: Advanced Metrics Expander
        with st.expander("📊 View Advanced Q-Metrics"):
            st.markdown("### 📉 Efficiency (Steps per Episode)")
            st.line_chart(q_metrics['lengths'])

            st.markdown("### 🎯 Success Rate (Moving Avg)")
            # 1 = Goal, 0 = Pit/Timeout. Moving Average for smoothness.
            success_smooth = pd.Series(q_metrics['success_rate']).rolling(window=20).mean()
            st.line_chart(success_smooth)

            st.markdown("### 🎲 Exploration Ratio")
            st.caption("% of actions that were random")
            st.line_chart(q_metrics['exploration_ratio'])

    # --- 2. RUN POLICY GRADIENT (Policy-Based) ---
    with col2:
        st.subheader("2. Policy Gradient")
        with st.spinner("Training Neural Network..."):
            # Returns a TUPLE (Rewards, Brain_History)
            pg_rewards, policy_history = run_policy_gradient(env, episodes=episodes, alpha=lr, gamma=gamma,
                                                             track_state=spy_state)
        st.success("Policy Gradient Complete!")
        st.metric("Avg Final Reward", f"{sum(pg_rewards[-50:]) / 50:.2f}")

    # --- 3. MAIN COMPARISON PLOT ---
    st.divider()
    st.subheader("📈 Performance Comparison (Cumulative Rewards)")

    window = max(1, int(episodes / 20))
    q_smooth = pd.Series(q_rewards).rolling(window=window).mean()
    pg_smooth = pd.Series(pg_rewards).rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(q_smooth, label="Q-Learning", color="teal", linewidth=2)
    ax.plot(pg_smooth, label="Policy Gradient", color="purple", linewidth=2)
    ax.set_title(f"Cumulative Rewards (Moving Avg Window: {window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # --- 4. BRAIN SCANNER PLOT ---
    st.divider()
    st.subheader(f"🧠 Inside the Policy Network (Monitoring State {spy_state})")
    st.markdown("This chart shows the **Probability** the agent assigns to each action over time.")

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    labels = ["Up", "Down", "Left", "Right"]
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F333FF"]

    for action_idx in range(4):
        probs = policy_history[:, action_idx]
        probs_smooth = pd.Series(probs).rolling(window=10).mean()
        ax2.plot(probs_smooth, label=labels[action_idx], color=colors[action_idx], linewidth=1.5, alpha=0.8)

    ax2.set_title(f"Action Probabilities for State {spy_state} over Time")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Confidence (0.0 - 1.0)")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)