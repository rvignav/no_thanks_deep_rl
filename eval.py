from mdp import get_subset, get_subset_index, get_mappings
import numpy as np

def run_games(N: int, K: int, pi_1: list, pi_2: list, num_games: int = 100):
    num_games_won = 0
    state2idx, idx2state = get_mappings(N, K)
    for i in range(num_games):
        p1_reward = 0
        p2_reward = 0
        
        c = np.random.randint(N)
        print(c)
        curr_state = (c, K/2, K/2, 0, 0)
        
        traj = []
        
        while get_subset(curr_state[3], N) + get_subset(curr_state[4], N) != [i for i in range(N)]:
            traj.append(curr_state)
            if curr_state[1] == 0:
                a = 0
            else:
                a = pi_1[state2idx[curr_state]]
                                    
            if a == 0:
                remaining_cards = list(set([i for i in range(N)]) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                
                if len(remaining_cards) == 0:
                    traj.append(a)
                    traj.append(K - curr_state[1] - curr_state[2] - curr_state[0])
                    p1_reward += K - curr_state[1] - curr_state[2] - curr_state[0]
                    break
                
                card = remaining_cards[np.random.randint(0, len(remaining_cards))]
                reward = K - curr_state[1] - curr_state[2] - curr_state[0]
                curr_state = (card, K-curr_state[2], curr_state[2], get_subset_index(get_subset(curr_state[3],N)+[curr_state[0]],N), curr_state[4])
            else:
                curr_state = (curr_state[0], curr_state[1]-1, curr_state[2], curr_state[3], curr_state[4])
                reward = -1
            
            traj.append(a)
            traj.append(reward)
            traj.append(curr_state)
            
            p1_reward += reward
            
            if curr_state[2] == 0:
                a = 0
            else:
                a = pi_2[state2idx[curr_state]]
                            
            if a == 0:
                remaining_cards = list(set([i for i in range(N)]) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                
                if len(remaining_cards) == 0:
                    traj.append(a)
                    traj.append(K - curr_state[1] - curr_state[2] - curr_state[0])
                    p2_reward += K - curr_state[1] - curr_state[2] - curr_state[0]
                    break
                
                card = remaining_cards[np.random.randint(0, len(remaining_cards))]
                reward = K - curr_state[1] - curr_state[2] - curr_state[0]
                curr_state = (card, curr_state[1], K-curr_state[1], curr_state[3], get_subset_index(get_subset(curr_state[4],N)+[curr_state[0]],N))
            else:
                curr_state = (curr_state[0], curr_state[1], curr_state[2]-1, curr_state[3], curr_state[4])
                reward = -1
            
            traj.append(a)
            traj.append(reward)
            
            p2_reward += reward
        
        print("Game ", i, " p1_reward ", p1_reward, " p2_reward ", p2_reward)
                                
        num_games_won += (p1_reward > p2_reward)
        # if p1_reward < p2_reward:
        print(traj)
    
    return num_games_won / num_games

def get_thresh_policy(N, K):
    thresh = int(K/2)

    pi_2 = []
    
    state2idx, idx2state = get_mappings(N, K)
    
    for s in range(len(idx2state)):
        if idx2state[s][1] >= thresh:
            pi_2.append(1)
        else:
            pi_2.append(0)
    pi_2.append(0)
    pi_2.append(0)
    return pi_2

def evaluate_policy(N: int, K: int, pi_1: list):
    if N >= 3:
        pi_2 = get_thresh_policy(N, K)
    else:
        # Compute optimal policy
        state2idx, idx2state = get_mappings(N, K)
        print(len(state2idx))
        pi_2 = np.zeros(len(state2idx)+2)
        for c in range(N):
            for k1 in range(K+1):
                for k2 in range(K+1):
                    for s1 in range(2**N):
                        for s2 in range(2**N):
                            s = (c, k1, k2, s1, s2)
                            try:
                                if c == 0:
                                    pi_2[state2idx[s]] = 0
                                else:
                                    if K - k1 - k2 >= 1:
                                        pi_2[state2idx[s]] = 0
                                    else:
                                        pi_2[state2idx[s]] = 1
                            except:
                                pass
    
    state2idx, idx2state = get_mappings(N, K)
    
    print("Percentage of games won ", run_games(N, K, pi_1, pi_2))
    
    # Check if pi_1 does milking + setting a threshold for NoThanks variant
    # for i in range(len(pi_1)-2):
    #     print("State ", idx2state[i], " pi_1 ", pi_1[i])
    
    # check if actions are only dependent on c, k1, k2