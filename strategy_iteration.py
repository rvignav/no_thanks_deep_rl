import numpy as np
import mdp   
import unittest
import eval
from mdp import get_mappings

class DP:
    def __init__(self, MDP, horizon):
        self.MDP = MDP
        self.track = self.MDP.state2idx[(2,1,1,0,0)]
        self.horizon = horizon
        self.step = 0

    def oneStep(self, prev_V):
        value = self.MDP.R + np.sum(self.MDP.P * prev_V.reshape(1, 1, -1), axis=2)
        policy = np.argmax(value, axis = 0)
        new_V = np.amax(value, axis = 0)

        self.step += 1
        print(self.horizon - self.step)
        print(new_V[self.track])

        return new_V, policy

    def fullDP(self):
        prev_V = np.zeros(self.MDP.nStates)
        policy_h = None

        for _ in range(self.horizon):
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
        self.num_states = len(state2idx)

        self.prev_policy = np.random.choice([0, 1], size=self.num_states)

    def iteration_step(self):
        MDP = mdp.build_nothanks_mdp(self.N, self.K, self.prev_policy)
        new_policy = None

        if self.optimization_method == 'DP':
            dp = DP(MDP, MDP.H)
            new_policy = dp.fullDP()
        
        self.prev_policy = new_policy

    def full_iteration(self):
        for _ in range(self.num_iterations):
            self.iteration_step()

        return self.prev_policy

class TestStrategyIteration(unittest.TestCase):
    def test_strategy_iteration(self):
        """
        Test that strategy iteration works.
        """
        N = 3
        K = 2
        num_iterations = 1
        si = StrategyIteration(N, K, 'DP', num_iterations)
        pi_1 = si.full_iteration()
        eval.evaluate_policy(N, K, pi_1)

if __name__ == '__main__':
    unittest.main()