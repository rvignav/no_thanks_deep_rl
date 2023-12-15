import numpy as np
import mdp   
import unittest
import eval
from mdp import get_mappings
import torch
import torch.nn as nn
import torch.optim as optim

class DP:
    def __init__(self, MDP):
        self.MDP = MDP
        
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
        
        # ONLY RETURNING FIRST TIMESTEP ACTIONS. CHECKED, IT'S NOT STATIONARY
        
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
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FittedQVI:
    def __init__(self, N, K, num_iterations, pi_data, lr=0.001, batch_size=32):
        self.N = N
        self.K = K
        self.num_iterations = num_iterations
        self.lr = lr
        self.batch_size = batch_size
        self.num_states = len(pi_data)
        self.pi_data = pi_data

    def rollout(self, num_trajectories, prev_policy):
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_horizons = []

        trajectories = mdp.simulate(self.N, self.K, self.pi_data, prev_policy, num_trajectories)
        for i in range(num_trajectories):

            trajectory = trajectories[i]

            for i in range(0, len(trajectory)-3, 3):
                all_states.append(trajectory[i])
                all_actions.append(trajectory[i + 1])
                all_rewards.append(trajectory[i + 2])
                all_next_states.append(trajectory[i + 3])
                all_horizons.append(i)

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), np.array(all_next_states), np.array(all_horizons)
    
    def train_q_network(self, states, actions, horizons, q_targets, q_network):
        optimizer = optim.Adam(q_network.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.num_iterations):
            for i in range(len(states)):
                s = states[i]
                a = actions[i]
                h = horizons[i]
                # print(q_targets[i])
                loss = criterion(q_network(torch.tensor([s, a, h], dtype=torch.float32)), torch.tensor([q_targets[i]], dtype=torch.float32))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def iteration_step(self, prev_policy):
        # Rollout to collect data for FQI
        states, actions, rewards, next_states, horizons = self.rollout(50, prev_policy)

        # Train Q-network using FQI
        q_network = QNetwork()
        
        outputs = []
        for a, act, c, r in zip(states, actions, horizons, rewards):
            nn_output = np.max([q_network(torch.tensor([a, b, c], dtype=torch.float32)).detach().numpy() for b in [0, 1]])
            outputs.append(r + nn_output)
        q_targets = outputs
        # q_targets = rewards + np.amax(q_network(torch.tensor([states, actions, horizons]).detach().numpy(), axis=1)

        self.train_q_network(states, actions, horizons, q_targets, q_network)

        # Use the trained Q-network to derive a new policy
        new_policy = [np.argmax([q_network(torch.tensor([s, a, 0], dtype=torch.float32)).detach().numpy() for a in [0, 1]]) for s in range(self.num_states)]

        return new_policy

    def fullQVI(self, horizon):
        policy_h = self.pi_data
        for _ in range(horizon):
            policy_h = self.iteration_step(policy_h)
                
        return policy_h

class FittedPI:
    def __init__(self, N, K, num_iterations, num_pi, num_states, lr=0.001, batch_size=32):
        self.N = N
        self.K = K
        self.num_iterations = num_iterations
        self.num_pi = num_pi
        self.lr = lr
        self.batch_size = batch_size
        self.num_states = num_states

    def rollout(self, num_trajectories, prev_policy, curr_policy):
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_horizons = []

        #print(self.N, self.K, self.pi_data, prev_policy, num_trajectories)
        trajectories = mdp.simulate(self.N, self.K, curr_policy, prev_policy, num_trajectories)
        for i in range(num_trajectories):

            trajectory = trajectories[i]

            for i in range(0, len(trajectory)-3, 3):
                all_states.append(trajectory[i])
                all_actions.append(trajectory[i + 1])
                all_rewards.append(trajectory[i + 2])
                all_next_states.append(trajectory[i + 3])
                all_horizons.append(i)

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), np.array(all_next_states), np.array(all_horizons)

    def train_q_network(self, states, actions, horizons, q_targets, q_network):
        optimizer = optim.Adam(q_network.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.num_iterations):
            for i in range(len(states)):
                s = states[i]
                a = actions[i]
                h = horizons[i]
                # print(q_targets[i])
                loss = criterion(q_network(torch.tensor([s, a, h], dtype=torch.float32)), torch.tensor([q_targets[i]], dtype=torch.float32))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def policy_iteration_step(self, prev_policy):
        new_policy = np.random.choice([0, 1], size=self.num_states)

        for k in range(self.num_pi):
            states, actions, rewards, next_states, horizons = self.rollout(50, new_policy, prev_policy)

            q_network = QNetwork()
            outputs = []
            for a, act, c, r in zip(states, actions, horizons, rewards):
                nn_output = np.max([q_network(torch.tensor([a, b, c], dtype=torch.float32)).detach().numpy() for b in [0, 1]])
                outputs.append(r + nn_output)
            q_targets = outputs
            self.train_q_network(states, actions, horizons, q_targets, q_network)

            # Use the trained Q-network to derive a new policy            
            new_policy = [np.argmax([q_network(torch.tensor([s, a, 0], dtype=torch.float32)).detach().numpy() for a in [0, 1]]) for s in range(self.num_states)]

        return new_policy

    def fullPI(self):
        policy_h = np.random.choice([0, 1], size=self.num_states)
        for _ in range(self.num_iterations):
            policy_h = self.policy_iteration_step(policy_h)

        return policy_h

class StrategyIteration:
    def __init__(self, N, K, optimization_method, num_iterations):
        self.N = N
        self.K = K
        self.num_iterations = num_iterations
        self.prev_policy = None
        self.optimization_method = optimization_method

        state2idx, _ = mdp.get_mappings(self.N, self.K)
        self.num_states = len(state2idx)+2

        self.prev_policy = np.random.choice([0, 1], size=self.num_states)

    def iteration_step(self):
        MDP = mdp.build_nothanks_mdp(self.N, self.K, self.prev_policy)
        new_policy = None
        
        if self.optimization_method == 'DP':
            dp = DP(MDP)
            new_policy = dp.fullDP(MDP.H)
        
        elif self.optimization_method == "QVI":
            qvi = FittedQVI(self.N, self.K, num_iterations = 10, pi_data = self.prev_policy)
            new_policy = qvi.fullQVI(MDP.H)

        elif self.optimization_method == "PI":
            fittedpi = FittedPI(self.N, self.K, num_iterations = 10, num_pi = 10, num_states = len(self.prev_policy))
            new_policy = fittedpi.fullPI()

        self.prev_policy = new_policy

    def full_iteration(self):
        for _ in range(self.num_iterations):
            print("Iteration ", _)
            self.iteration_step()

        return self.prev_policy

class TestStrategyIteration(unittest.TestCase):
    def test_strategy_iteration(self):
        """
        Test that strategy iteration works.
        """
        N = 2
        K = 2
        
        num_iterations = 2
        si = StrategyIteration(N, K, 'PI', num_iterations)
        pi_1 = si.full_iteration()
        eval.evaluate_policy(N, K, pi_1)

if __name__ == '__main__':
    unittest.main()