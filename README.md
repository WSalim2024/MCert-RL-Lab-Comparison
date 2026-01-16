<div align="center">

# 🤖 RL Lab: Q-Learning vs. Policy Gradients

### **An Interactive Battleground for Reinforcement Learning**

*Visualize the Fundamental Trade-offs Between Value-Based and Policy-Based Methods*

---

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![RL](https://img.shields.io/badge/Reinforcement_Learning-Lab-purple?style=for-the-badge)


[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=flat-square&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/waqar-salim/)

<br>

[**Features**](#-key-features) · [**The Science**](#-the-science) · [**Installation**](#-installation) · [**Usage**](#-usage)

<br>

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "The best way to understand RL is to watch two agents learn —              ║
║    one by memorizing values, one by tuning a brain."                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [The Science](#-the-science)
- [Key Features](#-key-features)
- [The Grid World](#-the-grid-world)
- [Screenshots](#-screenshots)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Directory Structure](#-directory-structure)
- [Author](#-author)

---

## 🎯 Project Overview

**RL Lab: Algorithm Arena** is an **educational laboratory** designed to visualize the fundamental trade-offs in Reinforcement Learning. It pits two philosophically distinct algorithms against each other in a controlled **5×5 Grid World** environment.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           THE ALGORITHM ARENA                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                            🏟️ GRID WORLD ARENA                                  │
│                                                                                 │
│                        ┌───┬───┬───┬───┬───┐                                    │
│                        │   │   │   │   │ 🏆│                                    │
│                        ├───┼───┼───┼───┼───┤                                    │
│                        │   │ ☠️│   │   │   │                                    │
│                        ├───┼───┼───┼───┼───┤                                    │
│                        │   │   │   │ ☠️│   │                                    │
│                        ├───┼───┼───┼───┼───┤                                    │
│                        │   │   │   │   │   │                                    │
│                        ├───┼───┼───┼───┼───┤                                    │
│                        │🤖 │   │   │   │   │                                    │
│                        └───┴───┴───┴───┴───┘                                    │
│                                                                                 │
│        ┌─────────────────────┐       ┌─────────────────────┐                    │
│        │  🧠 Q-LEARNING      │  VS   │  🧬 POLICY GRADIENT │                    │
│        │                     │       │                     │                    │
│        │  "I memorize the    │       │  "I learn the       │                    │
│        │   value of every    │       │   probability of    │                    │
│        │   state-action"     │       │   every action"     │                    │
│        │                     │       │                     │
│        │  📊 Q-Table         │       │  🔮 Neural Network  │                    │
│        └─────────────────────┘       └─────────────────────┘                    │
│                                                                                 │
│                         WHO LEARNS FASTER? WHO WINS?                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Why This Project?

| Challenge | How RL Lab Solves It |
|-----------|---------------------|
| RL algorithms are abstract | **Visual dashboard** shows learning in real-time |
| Hard to compare methods | **Side-by-side race** with live reward graphs |
| Neural networks are "black boxes" | **Brain Scanner** reveals internal activations |
| Theory-practice gap | **Interactive sliders** let you experiment with hyperparameters |

---

## 🔬 The Science

### The Two Paradigms of Reinforcement Learning

RL algorithms can be broadly categorized into two families. This lab explores one representative from each:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    VALUE-BASED vs POLICY-BASED LEARNING                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   VALUE-BASED (Q-Learning)              POLICY-BASED (REINFORCE)                │
│   ────────────────────────              ────────────────────────                │
│                                                                                 │
│   "How good is this                     "What should I                          │
│    state-action pair?"                   probably do here?"                     │
│                                                                                 │
│        State + Action                        State                              │
│             │                                  │                                │
│             ▼                                  ▼                                │
│      ┌───────────┐                      ┌───────────┐                           │
│      │  Q-Table  │                      │  Neural   │                           │
│      │  (Lookup) │                      │  Network  │                           │
│      └─────┬─────┘                      └─────┬─────┘                           │
│            │                                  │                                 │
│            ▼                                  ▼                                 │
│      ┌───────────┐                      ┌───────────┐                           │
│      │  Q-Value  │                      │  Action   │                           │
│      │  (Number) │                      │  Probs    │                           │
│      └───────────┘                      └───────────┘                           │
│                                                                                 │
│      Q(s,a) = 0.73                      π(Up)=0.6, π(Down)=0.1                  │
│                                         π(Left)=0.1, π(Right)=0.2              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

### 🧠 Q-Learning (Value-Based)

**Philosophy:** Learn the *value* of every state-action pair, then act greedily.

<table>
<tr>
<td width="50%">

#### How It Works

1. Maintain a **Q-Table**: `Q[state][action]`
2. Take action, observe reward and next state
3. Update Q-value using **Bellman Equation**:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

4. Choose action: ε-greedy (explore vs exploit)

</td>
<td width="50%">

#### Characteristics

| Property | Value |
|----------|-------|
| **Representation** | Tabular (Q-Table) |
| **Stability** | ✅ Very stable |
| **Sample Efficiency** | ✅ High |
| **Scalability** | ❌ Limited to discrete states |
| **Convergence** | ✅ Guaranteed (under conditions) |

</td>
</tr>
</table>

#### Q-Table Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Q-TABLE EXAMPLE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   State    │    ↑ Up    │   ↓ Down   │   ← Left   │   → Right  │               │
│   ─────────┼────────────┼────────────┼────────────┼────────────┤               │
│   (0,0)    │    0.34    │    0.12    │    0.00    │   [0.78]   │ ← Best action │
│   (0,1)    │    0.45    │    0.23    │    0.11    │   [0.89]   │               │
│   (1,2)    │   [0.92]   │    0.15    │    0.33    │    0.67    │               │
│   ...      │    ...     │    ...     │    ...     │    ...     │               │
│   (4,4)    │    0.00    │    0.00    │    0.00    │    0.00    │ ← Goal state  │
│                                                                                 │
│   📊 25 states × 4 actions = 100 Q-values to learn                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 🧬 Policy Gradients (REINFORCE)

**Philosophy:** Directly learn a *policy* (probability distribution over actions) using a neural network.

<table>
<tr>
<td width="50%">

#### How It Works

1. Neural network outputs **action probabilities**
2. Sample action from distribution: $a \sim \pi_\theta(s)$
3. Collect entire episode trajectory
4. Update network using **Policy Gradient Theorem**:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

Where $G_t$ = cumulative future reward

</td>
<td width="50%">

#### Characteristics

| Property | Value |
|----------|-------|
| **Representation** | Neural Network |
| **Stability** | ⚠️ Can be unstable |
| **Sample Efficiency** | ❌ Lower (needs more episodes) |
| **Scalability** | ✅ Handles continuous actions |
| **Convergence** | ⚠️ May suffer catastrophic forgetting |

</td>
</tr>
</table>

#### Policy Network Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          POLICY NETWORK ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   INPUT                 HIDDEN LAYERS              OUTPUT                       │
│   ─────                 ─────────────              ──────                       │
│                                                                                 │
│   ┌─────┐              ┌─────────────┐            ┌─────────────┐               │
│   │  x  │───┐          │             │            │   ↑ Up      │──► 0.60      │
│   │coord│   │          │   Dense     │            ├─────────────┤               │
│   └─────┘   │──────►   │   (64)      │──────►     │   ↓ Down    │──► 0.10      │
│   ┌─────┐   │   ReLU   │             │   ReLU     ├─────────────┤               │
│   │  y  │───┘          │   Dense     │            │   ← Left    │──► 0.10      │
│   │coord│              │   (32)      │   Softmax  ├─────────────┤               │
│   └─────┘              │             │            │   → Right   │──► 0.20      │
│                        └─────────────┘            └─────────────┘               │
│                                                                                 │
│   State: (2,3)         64 + 32 neurons            Action Probabilities          │
│   → [2, 3]             with ReLU                  (sum to 1.0)                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### ⚔️ Head-to-Head Comparison

<div align="center">

| Aspect | Q-Learning | Policy Gradients |
|:-------|:----------:|:----------------:|
| **Learning Target** | State-Action Values | Action Probabilities |
| **Data Structure** | Q-Table (lookup) | Neural Network (function) |
| **Update Frequency** | Every step | End of episode |
| **Exploration** | ε-greedy | Stochastic sampling |
| **Stability** | ✅ Stable | ⚠️ High variance |
| **Sample Efficiency** | ✅ Efficient | ❌ Needs more data |
| **Catastrophic Forgetting** | ❌ No | ✅ Possible |
| **Continuous Actions** | ❌ No | ✅ Yes |

</div>

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🎛️ Interactive Dashboard

Built with **Streamlit** to adjust hyperparameters in real-time:

```
┌─────────────────────────────┐
│  ⚙️ Hyperparameters         │
│                             │
│  Learning Rate (α)          │
│  [0.01]────●────[0.5]       │
│           α = 0.1           │
│                             │
│  Discount Factor (γ)        │
│  [0.5]─────●────[0.99]      │
│           γ = 0.95          │
│                             │
│  Episodes                   │
│  [100]─────●────[2000]      │
│         n = 500             │
└─────────────────────────────┘
```

</td>
<td width="50%">

### 📊 Live Race Visualization

Watch cumulative reward graphs update as both agents train **side-by-side**:

```
Cumulative Reward
    │
 500├         ┌────── Q-Learning
    │        /
 400├       /    ┌─── Policy Gradient
    │      /    /
 300├     /    / (catching up)
    │    /    /
 200├   /    /
    │  /    /
 100├ /    /
    │/    /
   0├────/─────────────────────
    0   100   200   300   400
              Episodes
```

</td>
</tr>
<tr>
<td colspan="2">

### 🧠 The Brain Scanner — *Spy on the Neural Network*

A unique visualization that reveals the **Policy Network's internal confidence** in each action direction. Watch how the agent's "beliefs" evolve over training:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           🧠 THE BRAIN SCANNER                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Neural Network Action Confidence at State (0,0)                               │
│   ───────────────────────────────────────────────                               │
│                                                                                 │
│   EPISODE 10 (Random)          EPISODE 100              EPISODE 500 (Learned)   │
│   ───────────────────          ───────────              ─────────────────────   │
│                                                                                 │
│        ↑ 23%                       ↑ 18%                      ↑ 5%              │
│         │                           │                          │                │
│    ←────┼────→                 ←────┼────→                ←────┼────→           │
│   28%   │  26%                12%   │  45%               3%    │  [87%]         │
│         │                           │                          │                │
│        ↓ 23%                       ↓ 25%                      ↓ 5%              │
│                                                                                 │
│   "I have no idea"            "Right seems good"         "Go RIGHT! (87%)"      │
│   (uniform distribution)      (learning...)              (confident policy)     │
│                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│   💡 INSIGHT: Watch the network's confidence shift from uniform to peaked       │
│              as it discovers the optimal path to the goal.                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</td>
</tr>
</table>

---

## 🗺️ The Grid World

The custom 5×5 Grid World environment serves as the controlled arena:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           5×5 GRID WORLD ENVIRONMENT                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                        ┌───┬───┬───┬───┬───┐                                    │
│                        │   │   │   │   │🏆 │  (4,4) = GOAL                      │
│                        │   │   │   │   │+10│  Reward: +10                       │
│                        ├───┼───┼───┼───┼───┤                                    │
│                        │   │☠️ │   │   │   │  (1,3) = TRAP                      │
│                        │   │-5 │   │   │   │  Reward: -5                        │
│                        ├───┼───┼───┼───┼───┤                                    │
│                        │   │   │   │☠️ │   │  (3,2) = TRAP                      │
│                        │   │   │   │-5 │   │  Reward: -5                        │
│                        ├───┼───┼───┼───┼───┤                                    │
│                        │   │   │   │   │   │                                    │
│                        │   │   │   │   │   │                                    │
│                        ├───┼───┼───┼───┼───┤                                    │
│                        │🤖 │   │   │   │   │  (0,0) = START                     │
│                        │ S │-1 │-1 │-1 │-1 │  Step cost: -1                     │
│                        └───┴───┴───┴───┴───┘                                    │
│                                                                                 │
│   ACTIONS: ↑ Up | ↓ Down | ← Left | → Right                                     │
│                                                                                 │
│   REWARDS:                                                                      │
│   • Reach goal (🏆): +10                                                        │
│   • Hit trap (☠️): -5 (episode ends)                                            │
│   • Each step: -1 (encourages efficiency)                                       │
│   • Hit wall: Stay in place, -1                                                 │
│                                                                                 │
│   OPTIMAL PATH: (0,0) → → → → ↑ ↑ ↑ ↑ → (4,4) = 9 steps, +1 total reward       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📸 Screenshots

<div align="center">

### Dashboard Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    🎛️ Interactive Dashboard with Live Training                  │
│                                                                                 │
│                         Add image: assets/dashboard.png                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Live Race Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    📊 Q-Learning vs Policy Gradient Reward Curves               │
│                                                                                 │
│                         Add image: assets/live_race.png                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Brain Scanner Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    🧠 Neural Network Action Confidence Evolution                │
│                                                                                 │
│                         Add image: assets/brain_scanner.png                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

*Screenshots will be added after deployment.*

</div>

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Version | Purpose |
|:-----:|:----------:|:-------:|:--------|
| 🐍 | **Python** | 3.10 | Core runtime |
| 🧠 | **TensorFlow** | 2.x | Deep learning (Policy Network) |
| | | `Keras` | High-level neural network API |
| 🖥️ | **Streamlit** | 1.28+ | Interactive dashboard |
| 🔢 | **NumPy** | 1.24+ | Q-Table operations |
| 📊 | **Matplotlib** | 3.7+ | Reward curve plotting |
| 📋 | **Pandas** | 2.0+ | Data logging & export |

</div>

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         STREAMLIT DASHBOARD                             │   │
│   │                           (app.py)                                      │   │
│   └───────────────────────────────┬─────────────────────────────────────────┘   │
│                                   │                                             │
│                    ┌──────────────┴──────────────┐                              │
│                    │                             │                              │
│                    ▼                             ▼                              │
│   ┌───────────────────────────┐   ┌───────────────────────────┐                 │
│   │      Q-LEARNING           │   │    POLICY GRADIENT        │                 │
│   │    (q_learning.py)        │   │  (policy_gradient.py)     │                 │
│   │                           │   │                           │                 │
│   │  ┌─────────────────────┐  │   │  ┌─────────────────────┐  │                 │
│   │  │     Q-Table         │  │   │  │   Neural Network    │  │                 │
│   │  │   (NumPy array)     │  │   │  │   (TensorFlow)      │  │                 │
│   │  └─────────────────────┘  │   │  └─────────────────────┘  │                 │
│   └─────────────┬─────────────┘   └─────────────┬─────────────┘                 │
│                 │                               │                               │
│                 └───────────────┬───────────────┘                               │
│                                 │                                               │
│                                 ▼                                               │
│                    ┌───────────────────────────┐                                │
│                    │       ENVIRONMENT         │                                │
│                    │    (environment.py)       │                                │
│                    │                           │                                │
│                    │    5×5 Grid World         │                                │
│                    │    • state, action, reward│                                │
│                    │    • done flag            │                                │
│                    └───────────────────────────┘                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📥 Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.10+ | [Download](https://python.org) |
| **pip** | Latest | Included with Python |
| **Git** | Any | [Download](https://git-scm.com) |

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/WSalim2024/RL-Lab-Comparison.git

# Navigate to project directory
cd RL-Lab-Comparison

# Install dependencies
pip install -r requirements.txt

# Launch the Lab
streamlit run app.py
```

### requirements.txt

```
streamlit>=1.28.0
tensorflow>=2.12.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
```

---

## ▶️ Usage

### Launch the Dashboard

```bash
streamlit run app.py
```

### Access in Browser

```
Local URL: http://localhost:8501
```

### Recommended Experiments

| Experiment | Settings | What to Observe |
|------------|----------|-----------------|
| **Baseline** | α=0.1, γ=0.95, 500 eps | Q-Learning converges faster |
| **High Learning Rate** | α=0.5 | Policy Gradient becomes unstable |
| **Low Discount** | γ=0.5 | Both agents become short-sighted |
| **Long Training** | 2000 episodes | Policy Gradient eventually catches up |

---

## 📁 Directory Structure

```
RL-Lab-Comparison/
│
├── 📄 app.py                    # Streamlit dashboard & comparison logic
├── 📄 environment.py            # Custom 5×5 Grid World engine
├── 📄 q_learning.py             # Tabular Q-Learning implementation
├── 📄 policy_gradient.py        # Deep Policy Gradient (REINFORCE)
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
└── 📄 .gitignore                # Git ignore rules
```

### Module Responsibilities

| File | Description |
|------|-------------|
| `app.py` | Main entry point; renders dashboard, orchestrates training |
| `environment.py` | Defines Grid World: states, actions, rewards, transitions |
| `q_learning.py` | Q-Table initialization, ε-greedy action selection, Bellman updates |
| `policy_gradient.py` | Keras model definition, episode collection, gradient computation |

---

## 🔮 Future Roadmap

| Feature | Description | Status |
|:--------|:------------|:------:|
| **DQN (Deep Q-Network)** | Neural network version of Q-Learning | 🔜 Planned |
| **Actor-Critic** | Hybrid value + policy method | 🔜 Planned |
| **Custom Grid Editor** | User-defined obstacles and goals | 🔜 Planned |
| **Training Replay** | Step-by-step episode playback | 🔜 Planned |

---

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning

---

## 👨‍💻 Author

<div align="center">

### **Waqar Salim**

*Master's Student & IT Professional*

---

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=for-the-badge&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/waqar-salim/)
---

**Built with 🤖 algorithms, 🧠 neural networks, and 🎮 curiosity**

*RL Lab: Algorithm Arena — Where Value Meets Policy*

---

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "An agent is only as good as its representation of the world —             ║
║    whether that's a table of values or a network of neurons."                 ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>
