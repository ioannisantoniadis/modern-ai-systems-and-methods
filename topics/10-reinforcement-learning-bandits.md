# Reinforcement Learning and Bandits

## Core Idea

Reinforcement learning studies agents that take actions, receive rewards, and affect future states. Bandits are a simpler setting where actions produce rewards without rich long-term state dynamics.

## MDP Vocabulary

- State: information about the current situation.
- Action: choice available to the agent.
- Reward: feedback signal.
- Policy: mapping from states to actions.
- Value function: expected future return from a state or action.
- Transition: how actions change state.
- Discount factor: how future rewards are weighted.

## Bandits

Bandits model exploration versus exploitation without full sequential state.

Types:

- Multi-armed bandit.
- Contextual bandit.
- Thompson sampling.
- UCB.
- Epsilon-greedy.

Use for recommendations, ads, ranking exploration, notifications, and personalization when immediate reward dominates.

## RL Methods

- Dynamic programming when environment is known.
- Monte Carlo and temporal difference learning.
- Q-learning.
- SARSA.
- Policy gradients.
- Actor-critic methods.

## Offline RL

Offline RL learns from logged behavior without new exploration. It is hard because the learned policy may choose actions not well covered by the logged data.

## RLHF Connection

RLHF uses human preference signals to optimize model behavior. In practice, it combines preference modeling, reward modeling, and policy optimization. It is connected to RL, ranking, and alignment.

## Interview Check

Be able to explain when not to use RL: if the action does not affect future state, if supervised labels are enough, if exploration is unsafe, or if reward is poorly defined.
