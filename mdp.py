import numpy as np
import matplotlib.pyplot as plt
import utils

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
    return c * K * K * (2 ** N) * (2 ** N) + k1 * K * (2 ** N) * (2 ** N) + k2 * (2 ** N) * (2 ** N) + s1 * (2 ** N) + s2

def get_state(index, N, K):
    c = index // (K * K * (2 ** N) * (2 ** N))
    index %= (K * K * (2 ** N) * (2 ** N))
    k1 = index // (K * (2 ** N) * (2 ** N))
    index %= (K * (2 ** N) * (2 ** N))
    k2 = index // ((2 ** N) * (2 ** N))
    index %= ((2 ** N) * (2 ** N))
    s1 = index // (2 ** N)
    index %= (2 ** N)
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
            
            trajectory.append(get_index(curr_state, N, K)
                                
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
