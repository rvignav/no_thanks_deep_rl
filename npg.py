# NPG
import utils
import mdp
import eval
import numpy as np

from utils import extract_features, compute_softmax, compute_action_distribution

import matplotlib.pyplot as plt

class NPG:
    def __init__(self, N, K, T, I, J, delta, lamb):
        self.N = N
        self.K = K
        self.T = T
        self.I = I
        self.J = J
        self.delta = delta
        self.lamb = lamb
        self.theta_2 = None

        self.total_rewards = []

    def compute_log_softmax_grad(self, theta, phis, action_idx):
        """ computes the log softmax gradient for the action with index action_idx

        :param theta: model parameter (shape d x 1)
        :param phis: RFF features of the state and actions (shape d x |A|)
        :param action_idx: The index of the action you want to compute the gradient of theta with respect to
        :return: log softmax gradient (shape d x 1)
        """
        return (phis[:, action_idx] - np.sum(phis @ compute_action_distribution(theta, phis).T, axis=1)).reshape(theta.shape)
        return (phis[:, action_idx] - np.sum(phis @ compute_action_distribution(theta, phis).T, axis=1)).reshape(theta.shape)

    def compute_fisher_matrix(self, grads, lamb):
        """ computes the fisher information matrix using the sampled trajectories gradients

        :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
        :param lamb: lambda value used for regularization
        :param lamb: lambda value used for regularization

        :return: fisher information matrix (shape d x d)





        Note: don't forget to take into account that trajectories might have different lengths
        """
        N = len(grads)
        d = grads[0][0].shape[0]
        fisher = np.zeros((d, d))
        for n in range(N):
            grad_sum = np.zeros((d, d))
            for t in range(len(grads[n])):
                grad_sum += grads[n][t] @ grads[n][t].T
            fisher += grad_sum / len(grads[n])
        return fisher / N + lamb * np.eye(d)


    # def calculate_fisher_matrix(grads, lamb=1.0):
    #     """ computes the fisher information matrix using the sampled trajectories gradients

    #     :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
    #     :param lamb: lambda value used for regularization

    #     :return: fisher information matrix (shape d x d)



    #     Note: don't forget to take into account that trajectories might have different lengths
    #     """
    #     N = len(grads)
    #     d = grads[0][0].shape[0]

    #     fisher_sum = np.zeros((d, d))

    #     for n in range(N):
    #         grad_sum = np.sum([np.outer(grad, grad) for grad in grads[n]], axis=0)
    #         fisher_sum += grad_sum / len(grads[n])

    #     fisher = fisher_sum / N + lamb * np.eye(d)
    #     return fisher


    # def calculate_fisher_matrix(grads, lamb=1.0):
    #     """ computes the fisher information matrix using the sampled trajectories gradients

    #     :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
    #     :param lamb: lambda value used for regularization

    #     :return: fisher information matrix (shape d x d)



    #     Note: don't forget to take into account that trajectories might have different lengths
    #     """
    #     N = len(grads)
    #     d = grads[0][0].shape[0]

    #     fisher_sum = np.zeros((d, d))

    #     for n in range(N):
    #         grad_sum = np.sum([np.outer(grad, grad) for grad in grads[n]], axis=0)
    #         fisher_sum += grad_sum / len(grads[n])

    #     fisher = fisher_sum / N + lamb * np.eye(d)
    #     return fisher

    def compute_value_gradient(self, grads, rewards):
        """ computes the value function gradient with respect to the sampled gradients and rewards

        :param grads: ist of list of gradients, where each sublist represents a trajectory
        :param rewards: list of list of rewards, where each sublist represents a trajectory
        :return: value function gradient with respect to theta (shape d x 1)
        """
        reward_totals = [np.sum(r) for r in rewards]
        b = np.sum(reward_totals) / len(reward_totals)

        value_grad = np.zeros(grads[0][0].shape)
        for n in range(len(grads)):
            grad_sum = np.zeros(grads[0][0].shape)
            for t in range(len(grads[n])):
                grad_sum += grads[n][t] * (np.sum(rewards[n][t:]) - b)
            value_grad += grad_sum / len(grads[n])
        return value_grad / len(grads)

    def compute_eta(self, delta, fisher, v_grad):
        """ computes the learning rate for gradient descent

        :param delta: trust region size
        :param fisher: fisher information matrix (shape d x d)
        :param v_grad: value function gradient with respect to theta (shape d x 1)
        :return: the maximum learning rate that respects the trust region size delta
        """
        eps = 1e-6
        eta = np.sqrt(delta / (v_grad.T @ np.linalg.inv(fisher) @ v_grad + eps))
        return eta

    def sample(self, theta, J):
        """ samples J trajectories using the current policy

        :param theta: the model parameters (shape d x 1)
        :param J: number of trajectories to sample
        :return:
            trajectories_gradients: lists with sublists for the gradients for each trajectory rollout (should be a 2-D list)
            trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout (should be a 2-D list)

        Note: the maximum trajectory length is 200 steps
        """
        trajectories_gradients = []
        trajectories_rewards = []

        trajectories = mdp.simulate_softmax(self.N, self.K, theta, self.theta_2, J)

        for trajectory in trajectories:
            n = len(trajectory) // 3
            trajectories_rewards.append([trajectory[3 * i + 2] for i in range(n)])

            gradients = []
            for time in range(n):
                state, action = mdp.get_state(trajectory[3 * time], self.N, self.K), trajectory[3 * time + 1]
                phis = extract_features(state, time)
                gradients.append(self.compute_log_softmax_grad(theta, phis, action))

            trajectories_gradients.append(gradients)

        return trajectories_gradients, trajectories_rewards


    def train(self, I, J, delta, lamb):
        """

        :param I: number of iterations to train the model
        :param J: number of trajectories to sample in each time step
        :param delta: trust region size
        :param lamb: lambda for fisher matrix computation
        :return:
            theta: the trained model parameters
            avg_episodes_rewards: list of average rewards for each time step
        """
        # theta = self.theta_2
        theta = np.random.rand(100,1)

        episode_rewards = []

        for iter in range(I):
            grads, rewards = self.sample(theta, J)
            fisher = self.compute_fisher_matrix(grads, lamb)
            v_grad = self.compute_value_gradient(grads, rewards)
            eta = self.compute_eta(delta, fisher, v_grad)


            theta += eta * np.linalg.inv(fisher) @ v_grad


            episode_rewards.append(np.mean([np.sum(r) for r in rewards]))


        return theta, episode_rewards

    def strategy_iteration(self):
        self.theta_2 = np.random.rand(100,1)
        # self.theta_2 = get_thresh_policy(self.N, self.K)
        self.theta_2 = np.random.rand(100,1)
        # self.theta_2 = get_thresh_policy(self.N, self.K)
        self.total_rewards = []

        for iter in range(self.T):
            print("Iteration: ", iter)
            self.theta_2, episode_rewards = self.train(self.I, self.J, self.delta, self.lamb)
            self.total_rewards.extend(episode_rewards)

        return self.theta_2, self.total_rewards

if __name__ == '__main__':
    np.random.seed(1234)
    N = 2
    K = 2
    npg = NPG(N, K, 2, 10, 20, 1e-2, 1e-3) # N, K, T = num strategy iterations, I = num NPG iterations, J = num rollouts
    theta, total_rewards = npg.strategy_iteration()

    eval.evaluate_policy_softmax(N, K, theta, 3)

    # print(total_rewards)
    # plt.plot(total_rewards)
    # plt.title("avg rewards per timestep")
    # plt.xlabel("timestep")
    # plt.ylabel("avg rewards")
    # plt.show()