import numpy as np
import mdp   

class DP:
    def __init__(self, MDP):
        self.MDP = MDP

    def oneStep(prev_V):
        value = self.R + np.sum(self.P * prev_V.reshape(1, 1, -1), axis=2)
        new_V = np.amax(value, axis = 0)
        policy = np.argmax(value, axis = 0)

        return new_V, policy

    def fullDP(horizon):
        prev_V = np.zeros(mdp.calc_S(self.MDP.N, self.MDP.K))
        policy_h = None

        for _ in range(horizon):
            prev_V, policy_h = oneStep(prev_V)

        return policy_h

class StrategyIteration:
    def __init__(self, N, K, H, optimization_method, num_iterations):
        self.N = N
        self.K = K
        self.H = H
        self.num_iterations = num_iterations
        self.policy_history = []
        self.optimization_method = optimization_method

        init_start_policy()

    def reset():
        self.policy_history = []

    def init_start_policy():
        num_states = mdp.calc_S(self.N, self.K)
        self.policy_history.append(np.random.choice([0, 1], size=num_states))

    def get_p2_policy():
        return self.policy_history[-1] # Could replace with something more fancy later
    
    def iteration_step():
        prev_policy = get_p2_policy()
        MDP = build_nothanks_mdp(N, K, prev_policy)
        new_policy = None

        if self.optimization_method == 'DP':
            dp = DP(MDP)
            new_policy = dp.fullDP(self.H)
        
        self.policy_history.append(new_policy)

    def full_iteration():
        for _ in range(self.num_iterations):
            iteration_step()

        return self.policy_history[-1]