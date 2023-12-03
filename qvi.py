import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mdp

class QNetwork(nn.Module):
    """
    Build a simple MLP
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FittedQVI:
    def __init__(self, N, K, H, num_iterations, lr=0.001, batch_size=32):
        self.N = N
        self.K = K
        self.H = H
        self.num_iterations = num_iterations
        self.policy_history = []
        self.lr = lr
        self.batch_size = batch_size

        self.init_start_policy()

    def reset(self):
        self.policy_history = []

    def init_start_policy(self):
        num_states = mdp.num_states
        self.policy_history.append(np.random.choice([0, 1], size=num_states))

    def get_p2_policy(self):
        return self.policy_history[-1]  # Could replace with something more fancy later

    def rollout(self, num_trajectories, trajectory_length):
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []

        prev_policy = self.get_p2_policy()
        for _ in range(num_trajectories):
            trajectory = mdp.simulate(self.N, self.K, [1] * mdp.num_states, prev_policy)
            states, actions, rewards, next_states = [], [], [], []

            for i in range(0, len(trajectory) - 2, 3):
                states.append(trajectory[i])
                actions.append(trajectory[i + 1])
                rewards.append(trajectory[i + 2])
                next_states.append(trajectory[i + 3])

            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_next_states.append(next_states)

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), np.array(all_next_states)

    def train_q_network(self, states, actions, q_targets, q_network):
        optimizer = optim.Adam(q_network.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.num_iterations):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                states_batch = torch.FloatTensor(states[batch_indices])
                actions_batch = torch.FloatTensor(actions[batch_indices])
                q_targets_batch = torch.FloatTensor(q_targets[batch_indices])

                q_values = q_network(states_batch, actions_batch)
                loss = criterion(q_values, q_targets_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def iteration_step(self):
        prev_policy = self.get_p2_policy()

        # Rollout to collect data for FQI
        states, actions, rewards, next_states = self.rollout(num_trajectories=1000, trajectory_length=self.H)

        # Flatten the arrays for training
        states_flat = states.reshape(-1, states.shape[-1])
        actions_flat = actions.reshape(-1)
        rewards_flat = rewards.reshape(-1)
        next_states_flat = next_states.reshape(-1, next_states.shape[-1])

        # Train Q-network using FQI

        q_network = QNetwork(state_dim=mdp.num_states, action_dim=1)
        q_targets = rewards_flat + np.amax(q_network(torch.FloatTensor(next_states_flat), torch.FloatTensor(actions_flat[:, None])).detach().numpy(), axis=1)

        self.train_q_network(states_flat, actions_flat, q_targets, q_network)

        # Use the trained Q-network to derive a new policy
        new_policy = np.argmax(q_network(torch.FloatTensor(states_flat), torch.FloatTensor(actions_flat[:, None])).detach().numpy(), axis=1)

        self.policy_history.append(new_policy)

    def full_iteration(self):
        for _ in range(self.num_iterations):
            self.iteration_step()

        return self.policy_history[-1]

# Example usage:
N = 3
K = 2
H = 10
fqvi = FittedQVI(N, K, H, num_iterations=5)
fqvi.full_iteration()
