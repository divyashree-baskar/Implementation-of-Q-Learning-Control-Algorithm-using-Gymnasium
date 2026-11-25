# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the Q-Learning algorithm using the Gymnasium CliffWalking environment, where an agent learns to navigate from the starting state (S) to the goal state (G) while avoiding cliff regions (C). The agent learns an optimal policy through trial-and-error interaction with the environment by updating its Q-values based on rewards and penalties received during training.


## Software Requirements
	Python 3.x
	Gymnasium
	NumPy
	Matplotlib
	Jupyter Notebook / Google Colab / Python IDE
  


## Environment Description
The CliffWalking environment in Gymnasium is a grid-world simulation consisting of 4 rows and 12 columns, with a total of 48 states. The agent starts at the bottom-left corner (S) and must reach the goal (G) at the bottom-right corner while avoiding the cliff region (C).
- States: 48 grid positions.
- Actions: Up, Down, Left and Right.
- Start state (S): Bottom-left corner.
- Goal state (G): Bottom-right corner.
- Cliff (C): Hazardous cells along the bottom row between the start and goal. Entering the cliff gives a penalty of −100 and returns the agent to the start.
- Reward: −1 for each step and −100 for falling into the cliff.
- Termination: The episode ends when the agent reaches the goal.
The agent uses Q-Learning with an epsilon-greedy strategy to learn a path to the goal while avoiding the cliff and minimizing penalties.


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
1. Create the Gymnasium CliffWalking-v1 environment and initialize the Q-table with random non-zero values.
2. Set the learning parameters: learning rate (\(\alpha\)), discount factor (\(\gamma\)), initial epsilon, minimum epsilon, and epsilon decay.
3. Reset the environment and observe the current state.
4. Select an action using the epsilon-greedy strategy.
5. Perform the action and observe the next state, reward, terminated and truncated conditions.
6. Calculate the target value using the maximum Q-value of the next state.
7. Update the Q-value using the Q-Learning update equation:

   Q(S,A)\leftarrow Q(S,A)+\alpha\left[R+\gamma\max_a Q(S',a)-Q(S,A)\right]
   
9. Repeat the process until the episode ends.
10. Gradually decrease epsilon and repeat training for multiple episodes.
11. Generate the learned policy, state-value function, average reward and learning curve.


## Python Program

```python

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

for episode in range(num_episodes):

    state, info = env.reset()

    terminated = False
    truncated = False
    total_reward = 0

    while not terminated and not truncated:

        # Choose action
        action = choose_action(state, epsilon)

        # Perform action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Calculate target
        if terminated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        # Update Q-value
        Q[state, action] += alpha * (
            target - Q[state, action]
        )

        state = next_state
        total_reward += reward

    # Store episode reward
    episode_rewards.append(total_reward)

    # Reduce epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Training completed!")






```
---

## Output


### Final Q-table:


<img width="701" height="717" alt="image" src="https://github.com/user-attachments/assets/679a689e-29aa-491c-88cc-b8f5b945f37a" />




### Estimated State-Value Function:


<img width="843" height="272" alt="image" src="https://github.com/user-attachments/assets/18a1b920-471b-4e53-a229-69db79e41cc8" />





### Learned Policy:


<img width="657" height="192" alt="image" src="https://github.com/user-attachments/assets/b51421b7-56f1-4d45-9777-e52c33f01843" />



### Average reward over last 1000 episodes:


<img width="685" height="77" alt="image" src="https://github.com/user-attachments/assets/509cff31-a6fe-46fe-8f63-c590ebba2fb6" />



## Result

```text
The Q-Learning algorithm was successfully implemented using the
FrozenLake-v1 environment. The agent learned the Q-values for
different state-action pairs and generated a learned policy for
reaching the goal while avoiding the holes. The learning performance
was evaluated using the average reward and learning curve.


```

---

## Inference

```text
The experiment shows that Q-Learning enables an agent to learn the
best actions through repeated interaction with the environment.
The Q-table improves with training, allowing the agent to select
suitable actions and achieve better rewards. The epsilon-greedy
strategy helps balance exploration and exploitation during learning.



```

---

