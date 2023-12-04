import mdp

def run_games(N: int, K: int, pi_1: list, pi_2: list, num_games: int = 100):
    num_games_won = 0
    for i in range(num_games):
        p1_reward = 0
        p2_reward = 0
        c = np.random.randint(N)
        curr_state = (c, K/2, K/2, 0, 0)
        trajectory = [curr_state]
        while get_subset(curr_state[3], N) + get_subset(curr_state[4], N) != [i for i in range(N)]:
            if curr_state[1] == 0:
                a = 0
            else:
                a = pi_1[curr_state]
                                    
            if a == 0:
                remaining_cards = list(set([i for i in range(N)]) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                card = remaining_cards[np.random.randint(0, len(remaining_cards))]
                reward = K - curr_state[1] - curr_state[2] - curr_state[0]
                curr_state = (card, K-curr_state[2], curr_state[2], get_subset_index(get_subset(curr_state[3],N)+[curr_state[0]],N), curr_state[4])
            else:
                curr_state = (curr_state[0], curr_state[1]-1, curr_state[2], curr_state[3], curr_state[4])
                reward = 0
            
            p1_reward += reward
            
            if curr_state[2] == 0:
                a = 0
            else:
                a = pi_2[curr_state]
                            
            if a == 0:
                remaining_cards = list(set([i for i in range(N)]) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                if len(remaining_cards) == 0:
                    break
                card = remaining_cards[np.random.randint(0, len(remaining_cards))]
                reward = K - curr_state[1] - curr_state[2] - curr_state[0]
                curr_state = (card, curr_state[1], K-curr_state[1], curr_state[3], get_subset_index(get_subset(curr_state[4],N)+[curr_state[0]],N))
            else:
                curr_state = (curr_state[0], curr_state[1], curr_state[2]-1, curr_state[3], curr_state[4])
                reward = 0
            
            p2_reward += reward
                                
        num_games_won += (p1_reward > p2_reward)
    
    return num_games_won / num_games

def evaluate_policy(N: int, K: int, pi_1: list):
    thresh = int(K/2)

    pi_2 = []
    
    state2idx, idx2state = mdp.get_mappings(N, K)
    for s in range(idx2state):
        if idx2state[s][1] >= thresh:
            pi_2.append(1)
        else:
            pi_2.append(0)
    pi_2.append(0)
    
    print("Percentage of games won ", run_games(3, 2, pi_1, pi_2))
    
    # Check if pi_1 does milking + setting a threshold for NoThanks variant