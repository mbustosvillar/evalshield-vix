"""
ppo_agent.py
============
Custom PPO Implementation for Tactical Asset Allocation.
Optimizes portfolio weights to maximize asymmetric reward.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, action_dim)
        self.sigma = nn.Parameter(torch.zeros(action_dim)) # Vector of log-sigmas

    def forward(self, x):
        x = self.fc(x)
        mu = torch.sigmoid(self.mu(x)) # Weights between 0 and 1
        sigma = torch.exp(self.sigma)
        return mu, sigma

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value.parameters(), 'lr': lr}
        ])
        self.gamma = gamma
        self.eps_clip = eps_clip

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0) # Add batch dim
        mu, sigma = self.policy(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).detach().numpy(), action_log_prob.item()

    def train(self, states, actions, log_probs, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # Standard PPO update
        for _ in range(5):
            mu, sigma = self.policy(states)
            dist = Normal(mu, sigma)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(self.value(states).squeeze(), returns)

            self.optimizer.zero_grad()
            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()
            self.optimizer.step()

def train_rl_agent(sim, epochs=10):
    state_dim = sim.state_dim
    action_dim = sim.action_dim
    agent = PPOAgent(state_dim, action_dim)

    print("[INFO] Starting RL Training (Standalone Simulator)...")
    for epoch in range(epochs):
        state = sim.reset()
        done = False
        states, actions, log_probs, rewards = [], [], [], []

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done = sim.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        # Compute returns and advantages
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + agent.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = np.array(returns)
        values = agent.value(torch.FloatTensor(np.array(states))).detach().numpy().squeeze()
        advantages = returns - values

        agent.train(np.array(states), np.array(actions), np.array(log_probs), returns, advantages)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Avg Reward: {np.mean(rewards):.6f} | Total Value: {sim.portfolio_value:.2f}")

    return agent

if __name__ == "__main__":
    import os
    from allocation_simulator import create_sim
    if os.path.exists("historical_training_data.csv"):
        sim = create_sim("historical_training_data.csv")
        train_rl_agent(sim)
    else:
        print("[ERROR] Training data not found.")
