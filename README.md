<div align="center">

# 🧪 RL Lab: Algorithm Arena

### **A High-Fidelity Reinforcement Learning Workbench**

*Visualize the Fundamental Trade-offs Between Value-Based and Policy-Based Methods*

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=flat-square&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/waqar-salim/)

<br>

[**Features**](#-key-features) · [**Architecture**](#-technical-architecture) · [**Installation**](#-installation-and-setup) · [**User Guide**](#-user-guide)

<br>

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "The best way to understand RL is to watch two philosophies compete —      ║
║    one learns by memorizing values, the other by tuning a neural brain."     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [What This Project Is About](#-what-this-project-is-about)
4. [What It Does](#-what-it-does)
5. [What Is The Logic](#-what-is-the-logic)
6. [How Does It Work](#-how-does-it-work)
7. [What Are The Requirements](#-what-are-the-requirements)
8. [Technical Architecture](#-technical-architecture)
9. [Model Specifications](#-model-specifications)
10. [Tech Stack](#-tech-stack)
11. [Install Dependencies](#-install-dependencies)
12. [Installation and Setup](#-installation-and-setup)
13. [Launching the Cockpit](#-launching-the-cockpit)
14. [User Guide](#-user-guide)
15. [Restrictions and Limitations](#-restrictions-and-limitations)
16. [Disclaimer](#-disclaimer)
17. [Author](#-author)

---

## 🚀 Overview

**RL Lab: Algorithm Arena** is a high-fidelity Reinforcement Learning workbench designed to visualize the fundamental trade-offs between **Value-Based** (Q-Learning) and **Policy-Based** (Policy Gradients) methods.

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
│                        │   │   │   │   │   │                                    │
│                        ├───┼───┼───┼───┼───┤                                    │
│                        │   │   │ ☠️│   │   │                                    │
│                        ├───┼───┼───┼───┼───┤                                    │
│                        │   │   │   │   │   │                                    │
│                        ├───┼───┼───┼───┼───┤                                    │
│                        │🤖 │   │   │   │   │                                    │
│                        └───┴───┴───┴───┴───┘                                    │
│                                                                                 │
│        ┌─────────────────────┐       ┌─────────────────────┐                    │
│        │  🧠 Q-LEARNING      │  VS   │  🧬 POLICY GRADIENT │                    │
│        │                     │       │                     │                    │
│        │  Tabular Method     │       │  Deep Learning      │                    │
│        │  25×4 Q-Table       │       │  Neural Network     │                    │
│        │                     │       │                     │                    │
│        │  📊 Lookup Table    │       │  🔮 Function Approx │                    │
│        └─────────────────────┘       └─────────────────────┘                    │
│                                                                                 │
│                         WHO LEARNS FASTER? WHO WINS?                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### The Two Paradigms

| Paradigm | Representative | Learning Target | Representation |
|:---------|:---------------|:----------------|:---------------|
| **Value-Based** | Q-Learning | State-Action Values | Tabular (Q-Table) |
| **Policy-Based** | REINFORCE | Action Probabilities | Neural Network |

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🧠 The Brain Scanner

A real-time visualization of the **Policy Network's internal activations**, showing how "confidence" in specific actions evolves over time.

```
Episode 10:          Episode 500:
   ↑ 25%                ↑ 3%
←──┼──→ 25%         ←──┼──→ [91%]
   ↓ 25%                ↓ 3%

"Random guessing"    "Confident policy"
```

*Watch the neural network's decision-making sharpen from uniform randomness to peaked certainty.*

</td>
<td width="50%">

### 📊 Advanced Analytics

Tracks comprehensive metrics beyond simple rewards:

| Metric | Description |
|--------|-------------|
| **Efficiency** | Steps per Episode (lower = better) |
| **Success Rate** | Goal reached vs Pit fallen (%) |
| **Exploration Ratio** | Random vs Greedy actions |
| **Cumulative Reward** | Total reward over time |

</td>
</tr>
<tr>
<td width="50%">

### 🏎️ Live Algorithm Race

Side-by-side training visualization comparing:

- **Tabular Agent** (Q-Learning)
- **Deep Learning Agent** (Policy Gradient)

Watch convergence speed, stability, and performance unfold in real-time.

</td>
<td width="50%">

### ⚙️ Dynamic Tuning

Adjust hyperparameters on the fly via sidebar sliders:

- **Learning Rate** ($\alpha$): 0.001 - 0.5
- **Discount Factor** ($\gamma$): 0.5 - 0.99
- **Episodes**: 100 - 5000
- **Epsilon** (Q-Learning): 0.01 - 1.0

</td>
</tr>
</table>

---

## 🎓 What This Project Is About

This project **bridges the gap between theory and practice** by providing a visual "sandbox" to observe how different RL algorithms solve the same navigation problem differently.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        BRIDGING THEORY AND PRACTICE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   TEXTBOOK KNOWLEDGE                          VISUAL UNDERSTANDING              │
│   ──────────────────                          ────────────────────              │
│                                                                                 │
│   "Q-Learning uses the                        Watch the Q-Table                 │
│    Bellman Equation to                        values update in                  │
│    iteratively update                  ───►   real-time as the                  │
│    state-action values"                       agent explores                    │
│                                                                                 │
│   "Policy Gradients can                       See the Brain Scanner             │
│    suffer from high                           show confidence                   │
│    variance and                        ───►   oscillating during                │
│    instability"                               unstable training                 │
│                                                                                 │
│   ABSTRACT EQUATIONS                          CONCRETE VISUALIZATIONS           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Learning Objectives

| Concept | How RL Lab Demonstrates It |
|---------|---------------------------|
| **Exploration vs Exploitation** | ε-greedy slider shows the trade-off |
| **Temporal Difference Learning** | Q-value updates visible step-by-step |
| **Policy Gradient Theorem** | Neural network confidence evolution |
| **Sample Efficiency** | Compare episodes needed to converge |
| **Stability vs Flexibility** | Q-Learning stability vs PG instability |

---

## ⚡ What It Does

The RL Lab performs three core functions:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CORE FUNCTIONALITY                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│   │  1️⃣ SIMULATE    │    │  2️⃣ TRAIN       │    │  3️⃣ VISUALIZE   │            │
│   │                 │    │                 │    │                 │            │
│   │  5×5 Grid World │───►│  Two Agents     │───►│  Live Graphs    │            │
│   │  Environment    │    │  Simultaneously │    │  & Analytics    │            │
│   │                 │    │                 │    │                 │            │
│   │  • 25 States    │    │  • Q-Learning   │    │  • Rewards      │            │
│   │  • 4 Actions    │    │  • Policy Grad  │    │  • Efficiency   │            │
│   │  • Rewards      │    │  • Same Env     │    │  • Brain Scan   │            │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Specific Capabilities

1. **Simulates** a 5×5 Grid World environment with configurable rewards
2. **Trains** two distinct agents simultaneously on identical conditions
3. **Renders** live performance graphs comparing:
   - Stability (reward variance)
   - Convergence speed (episodes to optimal)
   - Decision-making confidence (action probabilities)

---

## 🧮 What Is The Logic

### The World

A **5×5 Grid** containing 25 discrete states:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           5×5 GRID WORLD LOGIC                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   STATE NUMBERING:                      SPATIAL LAYOUT:                         │
│   ────────────────                      ───────────────                         │
│                                                                                 │
│   ┌────┬────┬────┬────┬────┐           ┌───┬───┬───┬───┬───┐                   │
│   │ 20 │ 21 │ 22 │ 23 │ 24 │           │   │   │   │   │🏆 │  State 24 = GOAL  │
│   ├────┼────┼────┼────┼────┤           ├───┼───┼───┼───┼───┤                   │
│   │ 15 │ 16 │ 17 │ 18 │ 19 │           │   │   │   │   │   │                   │
│   ├────┼────┼────┼────┼────┤           ├───┼───┼───┼───┼───┤                   │
│   │ 10 │ 11 │ 12 │ 13 │ 14 │           │   │   │☠️ │   │   │  State 12 = PIT   │
│   ├────┼────┼────┼────┼────┤           ├───┼───┼───┼───┼───┤                   │
│   │  5 │  6 │  7 │  8 │  9 │           │   │   │   │   │   │                   │
│   ├────┼────┼────┼────┼────┤           ├───┼───┼───┼───┼───┤                   │
│   │  0 │  1 │  2 │  3 │  4 │           │🤖 │   │   │   │   │  State 0 = START  │
│   └────┴────┴────┴────┴────┘           └───┴───┴───┴───┴───┘                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Reward Structure

| Event | Reward | Effect |
|:------|:------:|:-------|
| 🏆 **Reach Goal** (State 24) | **+10** | Episode ends (success) |
| ☠️ **Fall in Pit** (State 12) | **-10** | Episode ends (failure) |
| 🚶 **Each Step** | **-1** | Encourages efficiency |
| 🧱 **Hit Wall** | **-1** | Stay in place |

### The Objective

**Maximize cumulative reward** by finding the shortest path to the goal while avoiding the center pit.

$$\text{Objective: } \max \sum_{t=0}^{T} \gamma^t r_t$$

---

## ⚙️ How Does It Work

### Q-Learning (Value-Based)

Uses a **lookup table** (25×4) and the **Bellman Equation** to memorize the value of every state-action pair.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Q-LEARNING MECHANISM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Q-TABLE (25 states × 4 actions):                                              │
│   ────────────────────────────────                                              │
│                                                                                 │
│   State │   ↑ Up   │  ↓ Down  │  ← Left  │ → Right  │                          │
│   ──────┼──────────┼──────────┼──────────┼──────────┤                          │
│     0   │   0.34   │   0.12   │   0.00   │  [0.78]  │ ← Best action            │
│     1   │   0.45   │   0.23   │   0.11   │  [0.89]  │                          │
│    ...  │   ...    │   ...    │   ...    │   ...    │                          │
│    24   │   0.00   │   0.00   │   0.00   │   0.00   │ ← Terminal (Goal)        │
│                                                                                 │
│   UPDATE RULE (Bellman Equation):                                               │
│   ─────────────────────────────────                                             │
│                                                                                 │
│   Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') - Q(s,a) ]                          │
│                         ─────────────────────────────                           │
│                              TD Target                                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Policy Gradient (REINFORCE)

Uses a **Neural Network** to output a probability distribution over actions, optimized via the **REINFORCE algorithm** (Monte Carlo Policy Gradient).

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        POLICY GRADIENT MECHANISM                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   NEURAL NETWORK FORWARD PASS:                                                  │
│   ────────────────────────────                                                  │
│                                                                                 │
│   State 7        One-Hot           Hidden         Output                        │
│   ───────        ───────           ──────         ──────                        │
│                                                                                 │
│     7      ►   [0,0,0,0,0,    ►   Dense(24)  ►   π(↑) = 0.15                   │
│                 0,0,1,0,0,        ReLU           π(↓) = 0.10                   │
│                 0,0,0,0,0,                       π(←) = 0.05                   │
│                 0,0,0,0,0,        Softmax        π(→) = 0.70                   │
│                 0,0,0,0,0]                                                      │
│                                                                                 │
│   Size: 25      Input: 25         24 neurons     Output: 4                      │
│                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│   UPDATE RULE (Policy Gradient Theorem):                                        │
│   ──────────────────────────────────────                                        │
│                                                                                 │
│   ∇θ J(θ) = E[ Σt ∇θ log π(at|st) · Gt ]                                       │
│                                                                                 │
│   Where Gt = Σk γ^k r(t+k) (Return from time t)                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Head-to-Head Comparison

| Aspect | Q-Learning | Policy Gradient |
|:-------|:----------:|:---------------:|
| **Representation** | 25×4 Table (100 values) | Neural Network (~700 params) |
| **Update Timing** | Every step (TD) | End of episode (MC) |
| **Exploration** | ε-greedy | Stochastic sampling |
| **Stability** | ✅ Very stable | ⚠️ High variance |
| **Sample Efficiency** | ✅ High | ❌ Lower |
| **Scalability** | ❌ Limited | ✅ Handles large spaces |

---

## 📦 What Are The Requirements

### System Requirements

| Requirement | Specification |
|:------------|:--------------|
| **Python** | 3.10 or higher |
| **OS** | Windows, macOS, or Linux |
| **RAM** | 4GB minimum (8GB recommended) |
| **Internet** | Required for initial package installation |

### Software Dependencies

All dependencies are installable via pip (see [Install Dependencies](#-install-dependencies)).

---

## 🏗️ Technical Architecture

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         STREAMLIT FRONTEND                              │   │
│   │                           (app.py)                                      │   │
│   │                                                                         │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │   │
│   │   │  Sidebar    │  │  Reward     │  │   Brain     │  │  Advanced   │   │   │
│   │   │  Controls   │  │  Graphs     │  │  Scanner    │  │  Metrics    │   │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │   │
│   └───────────────────────────────┬─────────────────────────────────────────┘   │
│                                   │                                             │
│                                   │ Orchestrates                                │
│                                   ▼                                             │
│                    ┌──────────────┴──────────────┐                              │
│                    │                             │                              │
│                    ▼                             ▼                              │
│   ┌───────────────────────────┐   ┌───────────────────────────┐                 │
│   │      Q-LEARNING           │   │    POLICY GRADIENT        │                 │
│   │    (q_learning.py)        │   │  (policy_gradient.py)     │                 │
│   │                           │   │                           │                 │
│   │  • Q-Table (NumPy)        │   │  • Keras Sequential       │                 │
│   │  • ε-greedy selection     │   │  • REINFORCE algorithm    │                 │
│   │  • Bellman updates        │   │  • Gradient ascent        │                 │
│   │                           │   │                           │                 │
│   │  Returns: {               │   │  Returns: {               │                 │
│   │    'rewards': [...],      │   │    'rewards': [...],      │                 │
│   │    'lengths': [...],      │   │    'lengths': [...],      │                 │
│   │    'success_rate': [...], │   │    'success_rate': [...], │                 │
│   │    'expl_ratio': [...]    │   │    'expl_ratio': [...]    │                 │
│   │  }                        │   │  }                        │                 │
│   └─────────────┬─────────────┘   └─────────────┬─────────────┘                 │
│                 │                               │                               │
│                 └───────────────┬───────────────┘                               │
│                                 │                                               │
│                                 ▼                                               │
│                    ┌───────────────────────────┐                                │
│                    │     GRID ENVIRONMENT      │                                │
│                    │    (environment.py)       │                                │
│                    │                           │                                │
│                    │  • 5×5 Grid (25 states)   │                                │
│                    │  • 4 Actions (↑↓←→)       │                                │
│                    │  • Reward logic           │                                │
│                    │  • Episode management     │                                │
│                    └───────────────────────────┘                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Component Responsibilities

| Component | File | Responsibility |
|:----------|:-----|:---------------|
| **Frontend** | `app.py` | UI rendering, training orchestration, visualization |
| **Engine** | `environment.py` | Custom GridEnvironment with state transitions |
| **Q-Agent** | `q_learning.py` | Tabular learning, returns metrics dictionary |
| **PG-Agent** | `policy_gradient.py` | Neural network training, returns metrics dictionary |

---

## 🤖 Model Specifications

### Q-Learning Agent

| Property | Specification |
|:---------|:--------------|
| **Type** | Tabular (Non-parametric) |
| **Structure** | 25 states × 4 actions = **100 Q-values** |
| **Update Rule** | Temporal Difference (TD-0) |
| **Action Selection** | ε-greedy |
| **Convergence** | Guaranteed (under conditions) |

### Policy Gradient Agent

| Property | Specification |
|:---------|:--------------|
| **Type** | Deep Neural Network (Parametric) |
| **Framework** | TensorFlow 2.x / Keras |
| **Architecture** | Sequential model |

**Network Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       POLICY NETWORK ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   LAYER              SPECIFICATION              OUTPUT SHAPE                    │
│   ─────              ─────────────              ────────────                    │
│                                                                                 │
│   Input              One-hot encoded state      (None, 25)                      │
│                      Size: 25                                                   │
│                                                                                 │
│   Hidden             Dense(24, activation='relu')                               │
│                      24 neurons with ReLU       (None, 24)                      │
│                                                                                 │
│   Output             Dense(4, activation='softmax')                             │
│                      4 action probabilities     (None, 4)                       │
│                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│   Total Parameters: (25 × 24) + 24 + (24 × 4) + 4 = 724 trainable params       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Version | Purpose |
|:-----:|:----------:|:-------:|:--------|
| 🐍 | **Python** | 3.10+ | Core runtime |
| 🧠 | **TensorFlow** | 2.x | Deep learning (Policy Network) |
| 🔢 | **NumPy** | Latest | Q-Table operations, array math |
| 📊 | **Matplotlib** | Latest | Reward curves, visualizations |
| 📋 | **Pandas** | Latest | Data logging, metrics tracking |
| 🖥️ | **Streamlit** | Latest | Interactive dashboard UI |

</div>

---

## 📥 Install Dependencies

Create a `requirements.txt` file with the following contents:

```
numpy
matplotlib
tensorflow
pandas
streamlit
```

Or install directly:

```bash
pip install numpy matplotlib tensorflow pandas streamlit
```

---

## 🔧 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/WSalim2024/MCert-RL-Lab-Comparison.git
```

### Step 2: Navigate to Project Directory

```bash
cd MCert-RL-Lab-Comparison
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "
import numpy
import tensorflow
import streamlit
import matplotlib
import pandas

print('✅ All dependencies installed successfully!')
print(f'   TensorFlow: {tensorflow.__version__}')
print(f'   NumPy: {numpy.__version__}')
"
```

---

## ▶️ Launching the Cockpit

### Start the Dashboard

```bash
streamlit run app.py
```

### Access in Browser

```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

---

## 📖 User Guide

### Step-by-Step Instructions

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           USER WORKFLOW                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   STEP 1                    STEP 2                    STEP 3                    │
│   ──────                    ──────                    ──────                    │
│                                                                                 │
│   ⚙️ Configure              🧠 Set Brain Scanner      🏁 Start Race             │
│                                                                                 │
│   Use Sidebar to set:       Select a state to        Click "Start              │
│   • Episodes (e.g., 500)    "spy on"                 Training Race"            │
│   • Learning Rate (α)                                                          │
│   • Discount Factor (γ)     Recommended:             Watch both agents         │
│                             Start State 0            train side-by-side        │
│                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│   STEP 4                                                                        │
│   ──────                                                                        │
│                                                                                 │
│   📊 Analyze Results                                                            │
│                                                                                 │
│   Open "Advanced Metrics" dropdown to view:                                     │
│   • Efficiency (steps per episode)                                              │
│   • Success Rate (goal reached %)                                               │
│   • Exploration Ratio                                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Recommended Experiments

| Experiment | Settings | Observation |
|:-----------|:---------|:------------|
| **Baseline** | α=0.1, γ=0.95, 500 eps | Q-Learning converges faster |
| **High LR** | α=0.5 | Policy Gradient may diverge |
| **Long Training** | 2000 episodes | PG eventually catches up |
| **Low Discount** | γ=0.5 | Both become short-sighted |

---

## 📸 Screenshots

<div align="center">

### Dashboard Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    🖥️ Main Dashboard with Sidebar Controls                      │
│                       Live Training Visualization                               │
│                                                                                 │
│                         Add image: assets/dashboard.png                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Brain Scanner Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    🧠 Neural Network Action Confidence Evolution                │
│                       Watch Policy Sharpen Over Training                        │
│                                                                                 │
│                         Add image: assets/brain_scanner.png                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Advanced Metrics Panel

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    📊 Efficiency, Success Rate & Exploration Analysis           │
│                       Detailed Performance Breakdown                            │
│                                                                                 │
│                         Add image: assets/advanced_metrics.png                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

*To add screenshots: Create an `assets/` folder and save your Streamlit app screenshots there.*

</div>

---

## ⚠️ Restrictions and Limitations

| Limitation | Description | Reason |
|:-----------|:------------|:-------|
| **Grid Size** | Fixed to 5×5 | Optimized for visualization clarity |
| **Compute** | CPU-optimized | High episode counts (>5000) may slow browser rendering |
| **PG Stability** | May occasionally diverge | Demonstrates real RL instability (feature, not bug!) |
| **No GPU** | TensorFlow runs on CPU | Small network doesn't benefit from GPU |

### Catastrophic Forgetting Warning

> ⚠️ **The Policy Gradient agent may occasionally diverge** (crash in performance) if the Learning Rate is set too high. This is **intentional** — it demonstrates a fundamental challenge in deep RL: instability and catastrophic forgetting.

---

## 📜 Disclaimer

<div align="center">

---

**🎓 EDUCATIONAL USE ONLY**

---

</div>

This tool is designed for **educational purposes**. Reinforcement Learning is inherently **stochastic** — results may vary slightly between runs due to random seed initialization.

- **Not for Production**: This is a learning tool, not a production RL system
- **Variability Expected**: Different runs may produce different learning curves
- **Simplified Environment**: The 5×5 Grid World is intentionally simple for pedagogical clarity

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
║   "Reinforcement Learning is the science of making decisions under           ║
║    uncertainty — and this lab lets you watch that uncertainty unfold."        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>
