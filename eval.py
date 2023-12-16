from mdp import get_subset, get_subset_index, get_index, get_state
import numpy as np
import utils

def run_games(N: int, K: int, pi_1: list, player2strategy: int = 1, num_games: int = 100, variant=False):
    num_games_won = 0
    for i in range(num_games):
        p1_reward = 0
        p2_reward = 0
        
        cards = [i for i in range(N)]
        if variant:
            num_to_remove = min(N-2, 5)
            cards = list(np.random.choice(cards, len(cards)-num_to_remove, replace=False))
        
        c = cards[np.random.randint(len(cards))]
        print(c)
        curr_state = (c, K/2, K/2, 0, 0)
        
        traj = []
        
        while get_subset(curr_state[3], N) + get_subset(curr_state[4], N) != cards:
            traj.append(curr_state)
            if curr_state[1] == 0:
                a = 0
            else:
                a = pi_1[get_index(curr_state, N, K)]
                                    
            if a == 0:
                remaining_cards = list(set(cards) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                
                if len(remaining_cards) == 0:
                    traj.append(a)
                    
                    if variant:
                        subset = get_subset(curr_state[3],N)
                        if curr_state[0]-1 in subset and curr_state[0]+1 in subset:
                            r = K - curr_state[1] - curr_state[2] +(curr_state[0]+1)
                        elif curr_state[0]-1 in subset:
                            r = K - curr_state[1] - curr_state[2]
                        elif curr_state[0]+1 in subset:
                            r = K - curr_state[1] - curr_state[2]+1
                        else:
                            r = K - curr_state[1] - curr_state[2] - curr_state[0]
                        
                        traj.append(r)
                        p1_reward += r
                        
                    else:
                        traj.append(K - curr_state[1] - curr_state[2] - curr_state[0])
                        p1_reward += K - curr_state[1] - curr_state[2] - curr_state[0]
                    
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
            
            traj.append(a)
            traj.append(reward)
            traj.append(curr_state)
            
            p1_reward += reward
            
            if curr_state[2] == 0:
                a = 0
            else:
                if player2strategy == 1:
                    a = int(get_state(s, N, K)[1] >= int(K/2))
                elif player2strategy == 2:
                    a = np.random.choice(2, p=[0.5, 0.5])
                else:
                    # optimal
                    try:
                        if curr_state[0] == 0:
                            a = 0
                        else:
                            if K - curr_state[1] - curr_state[2] >= 1:
                                a = 0
                            else:
                                a = 1
                    except:
                        a = 0
                    
                    
                            
            if a == 0:
                remaining_cards = list(set(cards) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                
                if len(remaining_cards) == 0:
                    traj.append(a)
                    
                    if variant:
                        subset = get_subset(curr_state[4],N)
                        if curr_state[0]-1 in subset and curr_state[0]+1 in subset:
                            r = K - curr_state[1] - curr_state[2] +(curr_state[0]+1)
                        elif curr_state[0]-1 in subset:
                            r = K - curr_state[1] - curr_state[2]
                        elif curr_state[0]+1 in subset:
                            r = K - curr_state[1] - curr_state[2]+1
                        else:
                            r = K - curr_state[1] - curr_state[2] - curr_state[0]
                        
                        traj.append(r)
                        p2_reward += r
                        
                    else:
                        traj.append(K - curr_state[1] - curr_state[2] - curr_state[0])
                        p2_reward += K - curr_state[1] - curr_state[2] - curr_state[0]
                    
                    break
                
                card = remaining_cards[np.random.randint(0, len(remaining_cards))]
                reward = K - curr_state[1] - curr_state[2] - curr_state[0]
                
                if not variant:
                    next_subset_index = get_subset_index(get_subset(curr_state[4],N)+[curr_state[0]],N)
                
                # Take sequences into account
                if variant:
                    subset = get_subset(curr_state[4],N)
                    if curr_state[0]-1 in subset and curr_state[0]+1 in subset:
                        reward = K - curr_state[1] - curr_state[2] +(curr_state[0]+1)
                    elif curr_state[0]-1 in subset:
                        reward = K - curr_state[1] - curr_state[2]
                    elif curr_state[0]+1 in subset:
                        reward = K - curr_state[1] - curr_state[2]+1
                    next_subset_index = get_subset_index(subset+[curr_state[0]],N)
                
                curr_state = (card, curr_state[1], K-curr_state[1], curr_state[3], next_subset_index)
            else:
                curr_state = (curr_state[0], curr_state[1], curr_state[2]-1, curr_state[3], curr_state[4])
                reward = -1
            
            traj.append(a)
            traj.append(reward)
            
            p2_reward += reward
        
        print("Game ", i, " p1_reward ", p1_reward, " p2_reward ", p2_reward)
                                
        num_games_won += (p1_reward > p2_reward)
        print(traj)
    
    return num_games_won / num_games

def run_games_softmax(N: int, K: int, theta_1: list, player2strategy: int = 1, num_games: int = 100, variant=False):
    num_games_won = 0
    
    for i in range(num_games):
        p1_reward = 0
        p2_reward = 0
        
        cards = [i for i in range(N)]
        if variant:
            num_to_remove = min(N-2, 5)
            cards = list(np.random.choice(cards, len(cards)-num_to_remove, replace=False))
        
        c = cards[np.random.randint(len(cards))]
        print(c)
        curr_state = (c, K/2, K/2, 0, 0)
        
        traj = []
        
        h = -1
        
        while get_subset(curr_state[3], N) + get_subset(curr_state[4], N) != cards:
            h += 1
            traj.append(curr_state)
            if curr_state[1] == 0:
                a = 0
            else:
                phis = utils.extract_features(curr_state, h)
                action_dist = utils.compute_action_distribution(theta_1, phis)
                a = np.random.choice(2, p=action_dist[0])
                                    
            if a == 0:
                remaining_cards = list(set(cards) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                
                if len(remaining_cards) == 0:
                    traj.append(a)
                    
                    if variant:
                        subset = get_subset(curr_state[3],N)
                        if curr_state[0]-1 in subset and curr_state[0]+1 in subset:
                            r = K - curr_state[1] - curr_state[2] +(curr_state[0]+1)
                        elif curr_state[0]-1 in subset:
                            r = K - curr_state[1] - curr_state[2]
                        elif curr_state[0]+1 in subset:
                            r = K - curr_state[1] - curr_state[2]+1
                        else:
                            r = K - curr_state[1] - curr_state[2] - curr_state[0]
                        
                        traj.append(r)
                        p1_reward += r
                        
                    else:
                        traj.append(K - curr_state[1] - curr_state[2] - curr_state[0])
                        p1_reward += K - curr_state[1] - curr_state[2] - curr_state[0]
                    
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
            
            traj.append(a)
            traj.append(reward)
            traj.append(curr_state)
            
            p1_reward += reward
            
            if curr_state[2] == 0:
                a = 0
            else:
                if player2strategy == 1:
                    a = int(get_state(s, N, K)[1] >= int(K/2))
                elif player2strategy == 2:
                    a = np.random.choice(2, p=[0.5, 0.5])
                else:
                    # optimal
                    try:
                        if curr_state[0] == 0:
                            a = 0
                        else:
                            if K - curr_state[1] - curr_state[2] >= 1:
                                a = 0
                            else:
                                a = 1
                    except:
                        a = 0
                            
            if a == 0:
                remaining_cards = list(set(cards) - set(get_subset(curr_state[3], N)) - set(get_subset(curr_state[4], N)) - set([curr_state[0]]))
                
                if len(remaining_cards) == 0:
                    traj.append(a)
                    
                    if variant:
                        subset = get_subset(curr_state[4],N)
                        if curr_state[0]-1 in subset and curr_state[0]+1 in subset:
                            r = K - curr_state[1] - curr_state[2] +(curr_state[0]+1)
                        elif curr_state[0]-1 in subset:
                            r = K - curr_state[1] - curr_state[2]
                        elif curr_state[0]+1 in subset:
                            r = K - curr_state[1] - curr_state[2]+1
                        else:
                            r = K - curr_state[1] - curr_state[2] - curr_state[0]
                        
                        traj.append(r)
                        p2_reward += r
                        
                    else:
                        traj.append(K - curr_state[1] - curr_state[2] - curr_state[0])
                        p2_reward += K - curr_state[1] - curr_state[2] - curr_state[0]
                    
                    break
                
                card = remaining_cards[np.random.randint(0, len(remaining_cards))]
                reward = K - curr_state[1] - curr_state[2] - curr_state[0]
                
                if not variant:
                    next_subset_index = get_subset_index(get_subset(curr_state[4],N)+[curr_state[0]],N)
                
                # Take sequences into account
                if variant:
                    subset = get_subset(curr_state[4],N)
                    if curr_state[0]-1 in subset and curr_state[0]+1 in subset:
                        reward = K - curr_state[1] - curr_state[2] +(curr_state[0]+1)
                    elif curr_state[0]-1 in subset:
                        reward = K - curr_state[1] - curr_state[2]
                    elif curr_state[0]+1 in subset:
                        reward = K - curr_state[1] - curr_state[2]+1
                    next_subset_index = get_subset_index(subset+[curr_state[0]],N)
                
                curr_state = (card, curr_state[1], K-curr_state[1], curr_state[3], next_subset_index)
            else:
                curr_state = (curr_state[0], curr_state[1], curr_state[2]-1, curr_state[3], curr_state[4])
                reward = -1
            
            traj.append(a)
            traj.append(reward)
            
            p2_reward += reward
        
        print("Game ", i, " p1_reward ", p1_reward, " p2_reward ", p2_reward)
                                
        num_games_won += (p1_reward > p2_reward)
        print(traj)
    
    return num_games_won / num_games

def evaluate_policy_softmax(N: int, K: int, theta_1: list, player2strategy: int = 1, variant: bool = False):
    print("Percentage of games won ", run_games_softmax(N, K, theta_1, player2strategy, variant=variant))

def evaluate_policy(N: int, K: int, pi_1: list, variant: bool = False):
    # player2strategy: 1 is threshold, 2 is dummy, 3 is optimal (only use for N=K=2)
    mwrt = run_games(N, K, pi_1, 1, variant=variant)
    mwrd = run_games(N, K, pi_1, 2, variant=variant)
    milks = False # TODO
    
    return mwrd, mwrt, milks