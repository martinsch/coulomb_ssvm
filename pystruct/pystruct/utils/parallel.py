import numpy as np


def eval_func_tuple(f_args):
    """Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])


def find_most_violated(m, X, Y, batch, model, n_samples, w):
    X_b = X[batch]
    Y_b = Y[batch]
    assert len(X_b) == len(Y_b)
    assert len(X_b) == n_samples

    if n_samples == 0:
        return 0, 0, n_samples, None
    Y_hat = model.batch_loss_augmented_inference(
        X_b, Y_b, w, relaxed=True)
    delta_joint_feature = (model.batch_joint_feature(X_b, Y_b)
                 - model.batch_joint_feature(X_b, Y_hat))
    loss = np.sum(model.batch_loss(Y_b, Y_hat))

    violation = np.maximum(0, loss - np.dot(w, delta_joint_feature))
    return violation, loss, n_samples, delta_joint_feature


