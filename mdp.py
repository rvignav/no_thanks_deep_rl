import numpy as np
import matplotlib.pyplot as plt

class MDP:
	def __init__(self, P, R, H):
		"""
		The constructor verifies that the inputs are valid and sets
		corresponding variables in a MDP object
		:param P: Transition function: |A| x |S| x |S'| array
		:param R: Reward function: |A| x |S| array
		:param H: time horizon: natural number
		"""
		assert P.ndim == 3, "Invalid transition function: it should have 3 dimensions"
		self.nActions = P.shape[0]
		self.nStates = P.shape[1]
		assert P.shape == (self.nActions, self.nStates, self.nStates), "Invalid transition function: it has dimensionality " + repr(P.shape) + ", but it should be (nActions,nStates,nStates)"
		assert (abs(P.sum(2) - 1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
		self.P = P
		assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions"
		assert R.shape == (self.nActions, self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
		self.R = R
		assert 1 <= H, "Invalid time horizon, should be >= 1"
		self.H = H

	def isTerminal(self, state):
		return state == self.nStates-1 # last state is dummy state denoting end of game

def get_index(c, k1, k2, s1, s2):
    return c * K * K * (2 ** N) * (2 ** N) + k1 * K * (2 ** N) * (2 ** N) + k2 * (2 ** N) * (2 ** N) + s1 * (2 ** N) + s2

def get_subset(index, N):
    # map number in range [0, 2^N - 1] to subset of {0, ..., N-1}
    subset = []
    for i in range(N):
        if index % 2 == 1:
            subset.append(i)
        index = index // 2
    return subset

def get_subset_index(subset, N):
    # map subset of {0, ..., N-1} to number in range [0, 2^N - 1]
    index = 0
    for i in subset:
        index += 2 ** i
    return index

def build_nothanks_mdp(N, K, pi_2):
	# Transition function: |A| x |S| x |S'| array
    S = N * K * K * (2 ** N) * (2 ** N)
    A = 2 # 0 = take card, 1 = pass
    
	P = np.zeros([A, S, S])
    for c in range(N):
        for k1 in range(K):
            for k2 in range(K):
                for s1 in range(2**n):
                    for s2 in range(2**n):
                        # take card
                        i1 = get_index(c, k1, k2, s1, s2)
                        remaining_cards = range(N) - get_subset(s1, N) - get_subset(s2, N)
                        for card in remaining_cards:
                            new_s1 = get_subset_index(get_subset(s1, N) + [card])
                            i2 = get_index(card, K-k2, k2, new_s1, s2)
                            a = pi_2(i2)
                            if a == 0:
                                new_remaining_cards = range(N) - get_subset(new_s1, N) - get_subset(s2, N)
                                for card2 in new_remaining_cards:
                                    new_s2 = get_subset_index(get_subset(s2, N) + [card2])
                                    i3 = get_index(card, K-k2, k2, new_s1, new_s2)
                                    P[0, i1, i3] = 1 / (len(remaining_cards) * len(new_remaining_cards))
                            else:
                                P[0, i1, get_index(card, K-k2, k2-1, new_s1, s2)] = 1 / len(remaining_cards)
                        
                        # pass
                        i2 = get_index(c, k1-1, k2, s1, s2)
                        a = pi_2(i2)
                        if a == 0:
                            remaining_cards = range(N) - get_subset(s1, N) - get_subset(s2, N)
                            for card in new_remaining_cards:
                                new_s2 = get_subset_index(get_subset(s2, N) + [card])
                                i3 = get_index(card, k1-1, K-(k1-1), s1, new_s2)
                                P[0, i1, i3] = 1 / len(remaining_cards)
                        else:
                            i2 = get_index(c, k1-1, k2, s1, s2)
                            P[1, i1, i2] = 1

	# Reward function: |A| x |S| array
	R = np.zeros([A, S])

	# set rewards
	for c in range(N):
        for k1 in range(K):
            for k2 in range(K):
                for s1 in range(2**n):
                    for s2 in range(2**n):
                        R[0, get_index(c, k1, k2, s1, s2)] = K - k1 - k2 - c

	# Time horizon
	H = N * (K/2 + 1)

	# MDP object
	mdp = MDP(P, R, H)
	return mdp