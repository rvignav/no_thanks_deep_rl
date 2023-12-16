import numpy as np
import matplotlib.pyplot as plt
import utils

class MDP:
    def __init__(self, idx2state, state2idx, N, K, P, R, H):
        """
        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object
        :param P: Transition function: |A| x |S| x |S'| array
        :param R: Reward function: |A| x |S| array
        :param H: time horizon: natural number
        """
        assert K % 2 == 0
        self.idx2state = idx2state # maps number to tuple of form (c, k1, k2, s1, s2) where c is current face-up 
                                   # card, k1 is # tokens p1 has, k2 analogous, s1 is player 1's subset index, s2 analogous
        self.state2idx = state2idx
        self.N = N
        self.K = K
        
        assert P.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = P.shape[0]
        self.nStates = P.shape[1]
        assert P.shape == (self.nActions, self.nStates, self.nStates), "Invalid transition function: it has dimensionality " + repr(P.shape) + ", but it should be (nActions,nStates,nStates)"
        
        assert (abs(P.sum(2) - 1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.P = P
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions"
        assert R.shape == (self.nActions, self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert H >= 1, "Invalid time horizon"
        self.H = H

    def isTerminal(self, state):
        return state == self.nStates-1 # last state is dummy state denoting end of game


def get_mappings(N, K):
    state2idx = {}
    idx2state = {}
    i = 0

    for c in range(N):
        for k1 in range(K+1):
            for k2 in range(K+1):
                for s1 in range(2**N):
                    for s2 in range(2**N):
                        if c in get_subset(s1, N) or c in get_subset(s2, N) or set(get_subset(s2, N)).intersection(set(get_subset(s1, N))) or k1 + k2 > K:
                            continue
                        state2idx[(c, k1, k2, s1, s2)] = i
                        idx2state[i] = (c, k1, k2, s1, s2)
                        i += 1
                    
    return state2idx, idx2state

def build_nothanks_mdp(N, K, pi_2):
    state2idx, idx2state = get_mappings(N, K)
                    
    # Transition function: |A| x |S| x |S| array
    S = len(state2idx)
    A = 2 # 0 = take card, 1 = pass
    
    P = np.zeros([A, S+2, S+2])
    
    for s in range(len(idx2state)):
        c, k1, k2, s1, s2 = idx2state[s]
        
        # take card
        remaining_cards = list(set([i for i in range(N)]) - set(get_subset(s1, N)) - set(get_subset(s2, N)) - set([c]))
        if len(remaining_cards) == 0:
            P[0, s, S+1] = 1
        else:
            for card in remaining_cards:
                new_s1 = get_subset_index(get_subset(s1, N) + [c], N)
                sprime = state2idx[(card, K-k2, k2, new_s1, s2)]
                a = pi_2[sprime]
                if a == 0 or k2 == 0:
                    new_remaining_cards = list(set([i for i in range(N)]) - set(get_subset(new_s1, N)) - set(get_subset(s2, N)) - set([card]))
                    if len(new_remaining_cards) == 0:
                        P[0, s, S+1] = 1
                    else:
                        for card2 in new_remaining_cards:
                            new_s2 = get_subset_index(get_subset(s2, N) + [card], N)
                            sdprime = state2idx[(card2, K-k2, k2, new_s1, new_s2)]
                            P[0, s, sdprime] = 1 / (len(remaining_cards) * len(new_remaining_cards))
                else:
                    P[0, s, state2idx[(card, K-k2, k2-1, new_s1, s2)]] = 1 / len(remaining_cards)
        
        # pass
        if k1 == 0:
            P[1, s, S] = 1
        else:
            sprime = state2idx[(c, k1-1, k2, s1, s2)]
            a = pi_2[sprime]
            if a == 0 or k2 == 0:
                remaining_cards = list(set([i for i in range(N)]) - set(get_subset(s1, N)) - set(get_subset(s2, N)) - set([c]))
                if len(remaining_cards) == 0:
                    P[1, s, S+1] = 1
                else:
                    for card in remaining_cards:
                        new_s2 = get_subset_index(get_subset(s2, N) + [c], N)
                        sdprime = state2idx[(card, k1-1, K-(k1-1), s1, new_s2)]
                        P[1, s, sdprime] = 1 / len(remaining_cards)
            else:
                sprime = state2idx[(c, k1-1, k2-1, s1, s2)]
                P[1, s, sprime] = 1

    P[0, S+1, S+1] = 1
    P[1, S+1, S+1] = 1
    
    P[0, S, S] = 1
    P[1, S, S] = 1

    # Reward function: |A| x |S| array
    R = np.zeros([A, S+2])

    # set rewards
    for s in range(len(idx2state)):
        c, k1, k2, s1, s2 = idx2state[s]
        R[0, s] = K - k1 - k2 - c
    
    for s in range(len(idx2state)):
        R[1, s] = -1
    
    R[0, S] = -1e8
    R[1, S] = -1e8
    
    R[0, S+1] = 0
    R[1, S+1] = 0
    
    # Time horizon
    H = int(N*(K/2+1))

    # MDP object
    mdp = MDP(idx2state, state2idx, N, K, P, R, H)
    return mdp

def pp(state, N):
    return f"c: {state[0]}, k1: {state[1]}, k2: {state[2]}, s1: {get_subset(state[3], N)}, s2: {get_subset(state[4], N)}"

def get_subset(index, N):
    # Map number in range [0, 2^N - 1] to subset of {0, ..., N-1}
    subset = []
    for i in range(N):
        if index & (1 << i):
            subset.append(i)
    return subset

def get_subset_index(subset, N):
    # Map subset of {0, ..., N-1} to number in range [0, 2^N - 1]
    index = 0
    for i in subset:
        index |= (1 << i)
    return index

def get_index(curr_state, N, K):
    c, k1, k2, s1, s2 = curr_state
    # Map state to number in range [0, N * (K+1) * (K+1) * 2^N * 2^N - 1]
    return int(c * ((K+1) * (K+1) * (2**N) * (2**N)) + k1 * ((K+1) * (2**N) * (2**N)) + k2 * ((2**N) * (2**N)) + s1 * (2**N) + s2)

def get_state(index, N, K):
    c = index // ((K+1) * (K+1) * (2**N) * (2**N))
    index -= c * ((K+1) * (K+1) * (2**N) * (2**N))
    k1 = index // ((K+1) * (2**N) * (2**N))
    index -= k1 * ((K+1) * (2**N) * (2**N))
    k2 = index // ((2**N) * (2**N))
    index -= k2 * ((2**N) * (2**N))
    s1 = index // (2**N)
    index -= s1 * (2**N)
    s2 = index
    return (c, k1, k2, s1, s2)

def simulate(N, K, pi_1, pi_2, num_trajectories, variant=False):
    trajs = []

    for _ in range(num_trajectories):
        cards = [i for i in range(N)]
        if variant:
            num_to_remove = min(N-2, 5)
            cards = list(np.random.choice(cards, len(cards)-num_to_remove, replace=False))
        
        trajectory = []
        c = cards[np.random.randint(len(cards))]
        curr_state = (c, K/2, K/2, 0, 0)
        trajectory = [get_index(curr_state, N, K)]
                        
        while get_subset(curr_state[3], N) + get_subset(curr_state[4], N) != cards:
            if curr_state[1] == 0:
                a = 0
            else:
                a = pi_1[get_index(curr_state, N, K)]
                        
            trajectory.append(a)
            
            if a == 0:
                remaining_cards = list(set(cards) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                
                if len(remaining_cards) == 0:
                    trajectory.append(K - curr_state[1] - curr_state[2] - curr_state[0])
                    break
                
                card = remaining_cards[np.random.randint(0, len(remaining_cards))]
                reward = K - curr_state[1] - curr_state[2] - curr_state[0]
                if not variant:
                    next_subset_index = get_subset_index(get_subset(curr_state[3],N)+[curr_state[0]],N)
                
                # Take sequences into account
                if variant:
                    subset = get_subset(curr_state[3],N)
                    if curr_state[0]-1 in subset and curr_state[0]+1 in subset:
                        reward = K - curr_state[1] - curr_state[2] +(curr_state[0]+1)
                    elif curr_state[0]-1 in subset:
                        reward = K - curr_state[1] - curr_state[2]
                    elif curr_state[0]+1 in subset:
                        reward = K - curr_state[1] - curr_state[2]+1
                    next_subset_index = get_subset_index(subset+[curr_state[0]],N)
                
                curr_state = (card, K-curr_state[2], curr_state[2], next_subset_index, curr_state[4])
            else:
                curr_state = (curr_state[0], curr_state[1]-1, curr_state[2], curr_state[3], curr_state[4])
                reward = -1
            
            trajectory.append(reward)
            
            if curr_state[2] == 0:
                a = 0
            else:
                a = pi_2[get_index(curr_state, N, K)]
                            
            if a == 0:
                remaining_cards = list(set(cards) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                if len(remaining_cards) == 0:
                    break
                card = remaining_cards[np.random.randint(0, len(remaining_cards))]
                curr_state = (card, curr_state[1], K-curr_state[1], curr_state[3], get_subset_index(get_subset(curr_state[4],N)+[curr_state[0]],N))
            else:
                curr_state = (curr_state[0], curr_state[1], curr_state[2]-1, curr_state[3], curr_state[4])
            
            trajectory.append(get_index(curr_state, N, K))
                                
        if len(trajectory) % 3 == 1:
            trajectory = trajectory[:-1]
        trajs.append(trajectory)
        
    return trajs # (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_n, a_n, r_n)

def simulate_softmax(N, K, theta_1, theta_2, num_trajectories, variant=False):
    trajs = []

    for _ in range(num_trajectories):
        cards = [i for i in range(N)]
        if variant:
            num_to_remove = min(N-2, 5)
            cards = list(np.random.choice(cards, len(cards)-num_to_remove, replace=False))
        
        trajectory = []
        c = cards[np.random.randint(len(cards))]
        curr_state = (c, K/2, K/2, 0, 0)
        trajectory = [get_index(curr_state, N, K)]
        
        h = -1
                
        while get_subset(curr_state[3], N) + get_subset(curr_state[4], N) != cards:
            h += 1
            if curr_state[1] == 0:
                a = 0
            else:
                phis = utils.extract_features(curr_state, h)
                action_dist = utils.compute_action_distribution(theta_1, phis)
                a = np.random.choice(2, p=action_dist[0])
                        
            trajectory.append(a)
            
            if a == 0:
                remaining_cards = list(set(cards) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                
                if len(remaining_cards) == 0:
                    trajectory.append(K - curr_state[1] - curr_state[2] - curr_state[0])
                    break
                
                card = remaining_cards[np.random.randint(0, len(remaining_cards))]
                reward = K - curr_state[1] - curr_state[2] - curr_state[0]
                if not variant:
                    next_subset_index = get_subset_index(get_subset(curr_state[3],N)+[curr_state[0]],N)
                
                # Take sequences into account
                if variant:
                    subset = get_subset(curr_state[3],N)
                    if curr_state[0]-1 in subset and curr_state[0]+1 in subset:
                        reward = K - curr_state[1] - curr_state[2] +(curr_state[0]+1)
                    elif curr_state[0]-1 in subset:
                        reward = K - curr_state[1] - curr_state[2]
                    elif curr_state[0]+1 in subset:
                        reward = K - curr_state[1] - curr_state[2]+1
                    next_subset_index = get_subset_index(subset+[curr_state[0]],N)
                
                curr_state = (card, K-curr_state[2], curr_state[2], next_subset_index, curr_state[4])
            else:
                curr_state = (curr_state[0], curr_state[1]-1, curr_state[2], curr_state[3], curr_state[4])
                reward = -1
            
            trajectory.append(reward)
            
            if curr_state[2] == 0:
                a = 0
            else:
                phis = utils.extract_features(curr_state, h)
                action_dist = utils.compute_action_distribution(theta_2, phis)
                a = np.random.choice(2, p=action_dist[0])
                            
            if a == 0:
                remaining_cards = list(set(cards) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                if len(remaining_cards) == 0:
                    break
                card = remaining_cards[np.random.randint(0, len(remaining_cards))]
                curr_state = (card, curr_state[1], K-curr_state[1], curr_state[3], get_subset_index(get_subset(curr_state[4],N)+[curr_state[0]],N))
            else:
                curr_state = (curr_state[0], curr_state[1], curr_state[2]-1, curr_state[3], curr_state[4])
            
            trajectory.append(get_index(curr_state, N, K))
                    
        if len(trajectory) % 3 == 1:
            trajectory = trajectory[:-1]
        trajs.append(trajectory)
    return trajs # (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_n, a_n, r_n)
        
if __name__ == "__main__":
    N = 3
    K = 2
    state2idx, idx2state = get_mappings(N, K)
    pi_2 = [0]*len(idx2state)
    print("Testing MDP gen with dummy player 2 policy")
    mdp = build_nothanks_mdp(N, K, pi_2)
    print("Succeeded")
    print(simulate(N, K, [1]*len(idx2state), pi_2, 1, variant = True))