import numpy as np
import mdp   
import unittest
import eval
from mdp import get_mappings

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

    def forward(self, state, action, h):
        x = torch.cat([state, action, h], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FittedQVI:
    def __init__(self, MDP, num_iterations, lr=0.001, batch_size=32):
        self.MDP = MDP
        self.num_iterations = num_iterations
        self.lr = lr
        self.batch_size = batch_size

    def rollout(self, num_trajectories, prev_policy):
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_horizons = []

        for _ in range(num_trajectories):
            # NEED TO CHECK IF THIS TRAJECTORY IS RIGHT
            trajectory = mdp.simulate(N, K, [1] * self.MDP.nStates, prev_policy)
            states, actions, rewards, next_states, horizons = [], [], [], [], []

            for i in range(0, len(trajectory), 3):
                states.append(trajectory[i])
                actions.append(trajectory[i + 1])
                rewards.append(trajectory[i + 2])
                next_states.append(trajectory[i + 3])
                horizons.append(i)

            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_next_states.append(next_states)
            all_horizons.append(horizons)

        return np.array(all_states), np.array(all_actions), np.array(all_rewards), np.array(all_next_states), np.array(all_horizons)
    
    def train_q_network(self, states, actions, horizons, q_targets, q_network):
        optimizer = optim.Adam(q_network.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.num_iterations):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                # THIS BATCH INDICES THING IS KINDA SUS
                states_batch = torch.FloatTensor(states[batch_indices])
                actions_batch = torch.FloatTensor(actions[batch_indices])
                horizons_batch = torch.FloatTensor(horizons[batch_indices])
                q_targets_batch = torch.FloatTensor(q_targets[batch_indices])

                q_values = q_network(states_batch, actions_batch, horizons_batch)
                loss = criterion(q_values, q_targets_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def iteration_step(self):
        prev_policy = self.get_p2_policy() #NEED TO CHANGE THIS

        # Rollout to collect data for FQI
        states, actions, rewards, next_states, horizons = self.rollout(1000, prev_policy)

        # Flatten the arrays for training
        states_flat = states.reshape(-1)
        actions_flat = actions.reshape(-1)
        rewards_flat = rewards.reshape(-1)
        next_states_flat = next_states.reshape(-1)
        horizons_flat = horizons.reshape(-1)

        # Train Q-network using FQI

        q_network = QNetwork()
        q_targets = rewards_flat + np.amax(q_network(torch.FloatTensor(next_states_flat), torch.FloatTensor(actions_flat[:, None])).detach().numpy(), axis=1)

        self.train_q_network(states_flat, actions_flat, horizons_flat, q_targets, q_network)

        # Use the trained Q-network to derive a new policy
        new_policy = np.argmax(q_network(torch.FloatTensor(states_flat), torch.FloatTensor(actions_flat[:, None])).detach().numpy(), axis=1)

        self.policy_history.append(new_policy)

    def fullQVI(self, horizon):
        #THIS NEEDS TO BE CHANGED
        prev_V = np.zeros(self.MDP.nStates)
        policy_h = None

        for _ in range(horizon):
            prev_V, policy_h = self.oneStep(prev_V)
                
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
        
        num_iterations = 20
        si = StrategyIteration(N, K, 'DP', num_iterations)
        pi_1 = si.full_iteration()
        eval.evaluate_policy(N, K, pi_1)

if __name__ == '__main__':
    unittest.main()