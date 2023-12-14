import numpy as np

def extract_features(s, h):
    feats = []
    for a in [0, 1]:
        # copy s
        sc = list(s)
        sc.append(a)
        sc.append(h)
        feats.append(sc)
    return np.array(feats).T

def compute_softmax(logits, axis):
    """ computes the softmax of the logits

    :param logits: the vector to compute the softmax over
    :param axis: the axis we are summing over
    :return: the softmax of the vector

    Hint: to make the softmax more stable, subtract the max from the vector before applying softmax
    """

    max_logit = np.max(logits, axis=axis, keepdims=True)
    return np.exp(logits - max_logit) / np.sum(np.exp(logits - max_logit), axis=axis)


def compute_action_distribution(theta, phis):
    """ compute probability distrubtion over actions

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :return: softmax probability distribution over actions (shape 1 x |A|)
    """

    logits = theta.T @ phis
    return compute_softmax(logits, axis=1)

if __name__ == "__main__":
    s = [1, 2, 3, 4, 5]
    h = 6
    print(extract_features(s, h))