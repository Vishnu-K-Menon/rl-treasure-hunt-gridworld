# Treasure Hunt Grid World – Reinforcement Learning with SARSA & N-step Double Q-Learning

This project defines and solves a custom **Treasure Hunt Grid World** environment using the **Gymnasium** API.  
We model the problem as a **Markov Decision Process (MDP)** and implement:

- **SARSA** – on-policy TD control
- **N-step Double Q-Learning** – off-policy method with reduced overestimation bias

The project was completed as part of **CSE 574-D: Introduction to Machine Learning (Fall 2024)** at the University at Buffalo.

---

## 🔍 Environment: Treasure Hunt Grid World

- **Grid size:** 5 × 5 (25 states)
- **State space:** \( S = \{(0, 0), (0, 1), ..., (4, 4)\} \)
- **Action space:** `{Up, Down, Left, Right}`
- **Rewards:**
  - `+100` for reaching the final goal
  - `+20` for stepping on a treasure (removed after collection)
  - `-20` for stepping on a trap
  - `-0.5` for all other moves (step penalty / time cost)
- **Objective:** Collect as many treasures as possible and reach the goal with maximum total return, within a step limit.
- **Visualization:**
  - Treasures: teal cells
  - Traps: dark purple cells
  - Goal: green cell
  - Agent: yellow cell

Safety constraints:

- The agent cannot move outside the grid.
- State transitions are bounded by the grid layout and action space.
- Episodes terminate when the goal is reached or when the max number of steps is exceeded.
- Reward design discourages unsafe or useless behavior (heavy penalty for traps, step cost, strong reward for reaching goal).

---

## 🧠 Methods

### 1. SARSA (On-policy TD Control)

SARSA learns an action-value function \(Q(s, a)\) by updating it from actual state–action pairs experienced under the current policy:

\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
\]

Key points:

- On-policy: learns the value of the behavior policy (ε-greedy in this project).
- Naturally safer / more conservative in stochastic or risky environments.
- Simple and computationally efficient.

We tune:

- Discount factor \( \gamma \in \{0.7, 0.89, 0.99\} \)
- Epsilon decay rate \( \in \{0.93, 0.975, 0.999\} \)

Best SARSA setup (for this environment):

- `alpha = 0.1`
- `gamma = 0.99`
- `epsilon_start = 1.0`
- `epsilon_decay = 0.975`
- `epsilon_min = 0.1`
- `episodes = 500`
- `max_timesteps = 100`

The learned policy reaches the goal reliably with an average greedy return of about **96.5**.

---

### 2. N-step Double Q-Learning

Double Q-Learning maintains **two** Q-tables, \(Q_A\) and \(Q_B\), to reduce overestimation bias:

\[
Q_A(s_t, a_t) \leftarrow Q_A(s_t, a_t) + \alpha \Big[ r_{t+1} + \gamma Q_B(s_{t+1}, \arg\max_a Q_A(s_{t+1}, a)) - Q_A(s_t, a_t) \Big]
\]

(and symmetrically for \(Q_B\)).

We extend this to **N-step returns**:

\[
G_{t:t+n} = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{n-1} r_{t+n} + \gamma^n Q_B(s_{t+n}, \arg\max_a Q_A(s_{t+n}, a))
\]

Then:

\[
Q_A(s_t, a_t) \leftarrow Q_A(s_t, a_t) + \alpha \big[ G_{t:t+n} - Q_A(s_t, a_t) \big]
\]

(and similarly for \(Q_B\)).

We experiment with:

- \( n \in \{1, 2, 3, 4, 5\} \)
- Discount factor \( \gamma \in \{0.7, 0.89, 0.99\} \)
- Several epsilon decay rates

Best Double Q-learning setup (for this environment):

- `alpha = 0.1`
- `gamma = 0.89`
- `epsilon_start = 1.0`
- `epsilon_decay = 0.93`
- `epsilon_min = 0.1`
- `episodes = 500`
- `max_timesteps = 100`

---

## 📊 Results (High-level)

- **SARSA**:
  - Rewards increase steadily over episodes and plateau at a high value.
  - Epsilon decays smoothly from 1.0 to 0.1, shifting from exploration to exploitation.
  - Greedy evaluation shows consistently high total reward per episode.

- **N-step Double Q-Learning**:
  - For \(n = 1\), behavior is similar to standard Double Q-Learning with fast convergence.
  - For \(n = 2, 3\), learning is more stable and returns improve due to better use of multi-step returns.
  - Very large n can make learning more variable because of higher variance in returns.

Overall, **n-step Double Q-Learning** with tuned hyperparameters slightly outperforms SARSA in final reward, while SARSA provides a strong, simple baseline.

Plots (see `/assets`):

- `assets/sarsa_total_reward.png`
- `assets/sarsa_epsilon_decay.png`
- `assets/doubleq_nstep_rewards.png`
- `assets/doubleq_nstep_epsilon.png`
- `assets/sarsa_vs_doubleq_rewards.png`

---
