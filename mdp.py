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

def build_nothanks_mdp(N, K):
	# Transition function: |A| x |S| x |S'| array
	P = np.zeros([4, 17, 17])
	a = 0.8;  # intended move
	b = 0.1;  # lateral move

	# up (a = 0)

	P[0, 0, 0] = a + b;
	P[0, 0, 1] = b;

	P[0, 1, 0] = b;
	P[0, 1, 1] = a;
	P[0, 1, 2] = b;

	P[0, 2, 1] = b;
	P[0, 2, 2] = a;
	P[0, 2, 3] = b;

	P[0, 3, 2] = b;
	P[0, 3, 3] = a + b;

	P[0, 4, 4] = b;
	P[0, 4, 0] = a;
	P[0, 4, 5] = b;

	P[0, 5, 4] = b;
	P[0, 5, 1] = a;
	P[0, 5, 6] = b;

	P[0, 6, 5] = b;
	P[0, 6, 2] = a;
	P[0, 6, 7] = b;

	P[0, 7, 6] = b;
	P[0, 7, 3] = a;
	P[0, 7, 7] = b;

	P[0, 8, 8] = b;
	P[0, 8, 4] = a;
	P[0, 8, 9] = b;

	P[0, 9, 8] = b;
	P[0, 9, 5] = a;
	P[0, 9, 10] = b;

	P[0, 10, 9] = b;
	P[0, 10, 6] = a;
	P[0, 10, 11] = b;

	P[0, 11, 10] = b;
	P[0, 11, 7] = a;
	P[0, 11, 11] = b;

	P[0, 12, 12] = b;
	P[0, 12, 8] = a;
	P[0, 12, 13] = b;

	P[0, 13, 12] = b;
	P[0, 13, 9] = a;
	P[0, 13, 14] = b;

	P[0, 14, 13] = b;
	P[0, 14, 10] = a;
	P[0, 14, 15] = b;

	P[0, 15, 16] = 1;
	P[0, 16, 16] = 1;

	# down (a = 1)

	P[1, 0, 0] = b;
	P[1, 0, 4] = a;
	P[1, 0, 1] = b;

	P[1, 1, 0] = b;
	P[1, 1, 5] = a;
	P[1, 1, 2] = b;

	P[1, 2, 1] = b;
	P[1, 2, 6] = a;
	P[1, 2, 3] = b;

	P[1, 3, 2] = b;
	P[1, 3, 7] = a;
	P[1, 3, 3] = b;

	P[1, 4, 4] = b;
	P[1, 4, 8] = a;
	P[1, 4, 5] = b;

	P[1, 5, 4] = b;
	P[1, 5, 9] = a;
	P[1, 5, 6] = b;

	P[1, 6, 5] = b;
	P[1, 6, 10] = a;
	P[1, 6, 7] = b;

	P[1, 7, 6] = b;
	P[1, 7, 11] = a;
	P[1, 7, 7] = b;

	P[1, 8, 8] = b;
	P[1, 8, 12] = a;
	P[1, 8, 9] = b;

	P[1, 9, 8] = b;
	P[1, 9, 13] = a;
	P[1, 9, 10] = b;

	P[1, 10, 9] = b;
	P[1, 10, 14] = a;
	P[1, 10, 11] = b;

	P[1, 11, 10] = b;
	P[1, 11, 15] = a;
	P[1, 11, 11] = b;

	P[1, 12, 12] = a + b;
	P[1, 12, 13] = b;

	P[1, 13, 12] = b;
	P[1, 13, 13] = a;
	P[1, 13, 14] = b;

	P[1, 14, 13] = b;
	P[1, 14, 14] = a;
	P[1, 14, 15] = b;

	P[1, 15, 16] = 1;
	P[1, 16, 16] = 1;

	# left (a = 2)

	P[2, 0, 0] = a + b;
	P[2, 0, 4] = b;

	P[2, 1, 1] = b;
	P[2, 1, 0] = a;
	P[2, 1, 5] = b;

	P[2, 2, 2] = b;
	P[2, 2, 1] = a;
	P[2, 2, 6] = b;

	P[2, 3, 3] = b;
	P[2, 3, 2] = a;
	P[2, 3, 7] = b;

	P[2, 4, 0] = b;
	P[2, 4, 4] = a;
	P[2, 4, 8] = b;

	P[2, 5, 1] = b;
	P[2, 5, 4] = a;
	P[2, 5, 9] = b;

	P[2, 6, 2] = b;
	P[2, 6, 5] = a;
	P[2, 6, 10] = b;

	P[2, 7, 3] = b;
	P[2, 7, 6] = a;
	P[2, 7, 11] = b;

	P[2, 8, 4] = b;
	P[2, 8, 8] = a;
	P[2, 8, 12] = b;

	P[2, 9, 5] = b;
	P[2, 9, 8] = a;
	P[2, 9, 13] = b;

	P[2, 10, 6] = b;
	P[2, 10, 9] = a;
	P[2, 10, 14] = b;

	P[2, 11, 7] = b;
	P[2, 11, 10] = a;
	P[2, 11, 15] = b;

	P[2, 12, 8] = b;
	P[2, 12, 12] = a + b;

	P[2, 13, 9] = b;
	P[2, 13, 12] = a;
	P[2, 13, 13] = b;

	P[2, 14, 10] = b;
	P[2, 14, 13] = a;
	P[2, 14, 14] = b;

	P[2, 15, 16] = 1;
	P[2, 16, 16] = 1;

	# right (a = 3)

	P[3, 0, 0] = b;
	P[3, 0, 1] = a;
	P[3, 0, 4] = b;

	P[3, 1, 1] = b;
	P[3, 1, 2] = a;
	P[3, 1, 5] = b;

	P[3, 2, 2] = b;
	P[3, 2, 3] = a;
	P[3, 2, 6] = b;

	P[3, 3, 3] = a + b;
	P[3, 3, 7] = b;

	P[3, 4, 0] = b;
	P[3, 4, 5] = a;
	P[3, 4, 8] = b;

	P[3, 5, 1] = b;
	P[3, 5, 6] = a;
	P[3, 5, 9] = b;

	P[3, 6, 2] = b;
	P[3, 6, 7] = a;
	P[3, 6, 10] = b;

	P[3, 7, 3] = b;
	P[3, 7, 7] = a;
	P[3, 7, 11] = b;

	P[3, 8, 4] = b;
	P[3, 8, 9] = a;
	P[3, 8, 12] = b;

	P[3, 9, 5] = b;
	P[3, 9, 10] = a;
	P[3, 9, 13] = b;

	P[3, 10, 6] = b;
	P[3, 10, 11] = a;
	P[3, 10, 14] = b;

	P[3, 11, 7] = b;
	P[3, 11, 11] = a;
	P[3, 11, 15] = b;

	P[3, 12, 8] = b;
	P[3, 12, 13] = a;
	P[3, 12, 12] = b;

	P[3, 13, 9] = b;
	P[3, 13, 14] = a;
	P[3, 13, 13] = b;

	P[3, 14, 10] = b;
	P[3, 14, 15] = a;
	P[3, 14, 14] = b;

	P[3, 15, 16] = 1;
	P[3, 16, 16] = 1;

	# Reward function: |A| x |S| array
	R = -1 * np.ones([4, 17]);

	# set rewards
	R[:, 15] = 100;  # goal state
	R[:, 9] = -70;  # bad state
	R[:, 5] = -70;  # bad state
	R[:, 16] = 0;  # end state

	# Time horizon
	H = N * (K/2 + 1)

	# MDP object
	mdp = MDP(P, R, H)
	return mdp