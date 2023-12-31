import numpy as np
import mdp   
import unittest
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import eval

class DP:
    def __init__(self, MDP):
        self.MDP = MDP
        self.reward_history = []

    def oneStep(self, prev_V):
        value = self.MDP.R + np.sum(self.MDP.P * prev_V.reshape(1, 1, -1), axis=2)
        policy = np.argmax(value, axis = 0)
        new_V = np.amax(value, axis = 0)

        return new_V, policy

    def fullDP(self, horizon):
        prev_V = np.zeros(self.MDP.nStates)
        policy_h = None

        for _ in range(horizon):
            prev_V, policy_h = self.oneStep(prev_V)
            total_reward = np.sum(prev_V)
            self.reward_history.append(total_reward)
            
        # Return the final policy
        return policy_h


class QNetwork(nn.Module):
    """
    Build a simple MLP
    """
    def __init__(self):
        super(QNetwork, self).__init__()
        # Input layer has 3 neurons
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        #x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class FittedQVI:
    def __init__(self, N, K, num_iterations, pi_data, variant, lr=0.0001, batch_size=32):
        self.N = N
        self.K = K
        self.num_iterations = num_iterations
        self.lr = lr
        self.batch_size = batch_size
        self.num_states = len(pi_data)
        self.pi_data = pi_data
        self.variant = variant

    def rollout(self, num_trajectories, prev_policy):
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_horizons = []

        trajectories = mdp.simulate(self.N, self.K, self.pi_data, prev_policy, num_trajectories, self.variant)
        for i in range(num_trajectories):

            trajectory = trajectories[i]

            for i in range(0, len(trajectory)-2, 3):
                all_states.append(trajectory[i])
                all_actions.append(trajectory[i + 1])
                all_rewards.append(trajectory[i + 2])
                all_horizons.append(i)

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), np.array(all_horizons)
    
    def train_q_network(self, states, actions, horizons, q_targets, q_network):
        optimizer = optim.Adam(q_network.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.num_iterations):
            for i in range(len(states)):
                s = states[i]
                a = actions[i]
                h = horizons[i]
                loss = criterion(q_network(torch.tensor([s, a, h], dtype=torch.float32)), torch.tensor([q_targets[i]], dtype=torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def iteration_step(self, prev_policy):
        states, actions, rewards, horizons = self.rollout(480, prev_policy)

        q_network = QNetwork()
        
        outputs = []
        for a, act, c, r in zip(states, actions, horizons, rewards):
            nn_output = np.max([q_network(torch.tensor([a, b, c], dtype=torch.float32)).detach().numpy() for b in [0, 1]])
            outputs.append(r + nn_output)
        q_targets = outputs

        self.train_q_network(states, actions, horizons, q_targets, q_network)

        new_policy = [np.argmax([q_network(torch.tensor([s, a, 0], dtype=torch.float32)).detach().numpy() for a in [0, 1]]) for s in range(self.num_states)]

        return new_policy

    def fullQVI(self, horizon):
        policy_h = self.pi_data
        for _ in tqdm(range(horizon)):
            policy_h = self.iteration_step(policy_h)
                
        return policy_h

class FittedPI:
    def __init__(self, N, K, num_iterations, num_eval_iterations, num_q_iterations, num_states, variant, lr=0.00001, batch_size=32):
        self.N = N
        self.K = K
        self.num_iterations = num_iterations
        self.num_eval_iterations = num_eval_iterations
        self.num_q_iterations = num_q_iterations
        self.lr = lr
        self.batch_size = batch_size
        self.num_states = num_states
        self.variant = variant

    def rollout(self, num_trajectories, prev_policy, curr_policy):
        all_states = []
        all_actions = []
        all_rewards = []
        all_horizons = []

        trajectories = mdp.simulate(self.N, self.K, curr_policy, prev_policy, num_trajectories, self.variant)
        for i in range(num_trajectories):

            trajectory = trajectories[i]

            for i in range(0, len(trajectory)-2, 3):
                all_states.append(trajectory[i])
                all_actions.append(trajectory[i + 1])
                all_rewards.append(trajectory[i + 2])
                all_horizons.append(i)

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), np.array(all_horizons)

    def train_q_network(self, states, actions, horizons, q_targets, q_network):
        optimizer = optim.Adam(q_network.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        q_targets = torch.stack([q_targets], dim=1)

        for i in range(self.num_q_iterations):
            optimizer.zero_grad()
            states, actions, horizons, q_targets = states.detach(), actions.detach(), horizons.detach(), q_targets.detach()
            outputs = q_network(torch.stack([states, actions, horizons], dim=1))
            loss = criterion(outputs, q_targets)
            loss.backward()
            optimizer.step()

    def policy_iteration_step(self, prev_policy):
        new_policy = np.random.choice([0, 1], size=self.num_states)

        for k in range(self.num_eval_iterations):
            states, actions, rewards, horizons = self.rollout(96, new_policy, prev_policy)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            horizons = torch.tensor(horizons, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            stacked_tensor = torch.stack([states, actions, horizons], dim=1)

            q_network = QNetwork()
            q_targets = torch.add(rewards, torch.max(q_network(stacked_tensor), dim=1).values)

            self.train_q_network(states, actions, horizons, q_targets, q_network)

            new_policy = [np.argmax([q_network(torch.tensor([s, a, 0], dtype=torch.float32)).detach().numpy() for a in [0, 1]]) for s in range(self.num_states)]

        trajectory = self.rollout(50, new_policy, prev_policy)
        reward = np.mean(trajectory[2])
        return new_policy, reward

    def fullPI(self):
        total_rewards = []
        policy_h = np.random.choice([0, 1], size=self.num_states)
        for _ in tqdm(range(self.num_iterations)):
            policy_h, reward = self.policy_iteration_step(policy_h)
            total_rewards.append(reward)
        return policy_h, total_rewards

class StrategyIteration:
    def __init__(self, N, K, optimization_method, num_iterations, variant):
        self.N = N
        self.K = K
        self.num_iterations = num_iterations
        self.prev_policy = None
        self.optimization_method = optimization_method
        self.variant = variant

        num_states = N * (K + 1) * (K + 1) * (2 ** N) * (2 ** N)
        self.num_states = num_states+2

        self.prev_policy = np.random.choice([0, 1], size=self.num_states)

    def iteration_step(self):
        if self.optimization_method == "DP":
            MDP = mdp.build_nothanks_mdp(self.N, self.K, self.prev_policy)
            self.MDP = MDP
        new_policy = None
        
        if self.optimization_method == 'DP':
            dp = DP(MDP)
            new_policy = dp.fullDP(MDP.H)
        
        elif self.optimization_method == "QVI":
            qvi = FittedQVI(self.N, self.K, num_iterations = 10, pi_data = self.prev_policy, variant = self.variant)
            new_policy = qvi.fullQVI(int(self.N*(self.K/2+1)))

        elif self.optimization_method == "PI":
            fittedpi = FittedPI(self.N, self.K, num_iterations = 10, num_eval_iterations = 10, num_q_iterations = 20, num_states = len(self.prev_policy), variant = self.variant)
            new_policy, total_rewards = fittedpi.fullPI()
        self.prev_policy = new_policy

    def full_iteration(self):
        metrics = {}
        for _ in range(self.num_iterations):
            print("Iteration ", _)
            self.iteration_step()
            
            try:
                mwrd, mwrt, milks = eval.evaluate_policy(self.N, self.K, self.prev_policy, variant=self.variant, state2idx=self.MDP.state2idx, idx2state=self.MDP.idx2state)
            except:
                mwrd, mwrt, milks = eval.evaluate_policy(self.N, self.K, self.prev_policy, variant=self.variant)
            metrics[_] = (mwrd, mwrt, milks)

        return metrics


if __name__ == "__main__":
    N = 2
    K = 2
    num_iterations = 2
    si = StrategyIteration(N, K, "PI", num_iterations, variant=False)
    rewards = si.full_iteration()
    print(rewards)
