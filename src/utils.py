import numpy as np
from scipy.special import expit
import aesara.tensor as at

"""
Functions for PyMC4 (using aesara instead of theano)
"""



def log_likelihood(w, link, Phi_n, Phi_gq, w_gq, Mj=None, T=None):
    """
    Log likelihood for RelU link with histogram basis
    :param w: K x (KB + 1) array (weights)
    :param Phi_n: (KB + 1) x N_T array (values of basis functions at each dimension and point)
    :param link:
    :param points_hawkes:
    :param T_phi:
    :param T:
    :return:
    """

    if np.abs(w).any() > 10e2:
        return -1.0 * np.Inf

    number_of_dimensions = len(Phi_n)
    number_of_basis = (Phi_n[0].shape[-1] -  1) // number_of_dimensions

    if w.ndim == 1:
        w = w.reshape(number_of_dimensions, number_of_dimensions * number_of_basis + 1)

    logl = 0.0
    for i in range(number_of_dimensions):

        # Linear intensity
        h_n = w[i].dot(Phi_n[i].T)
        h_gq = w[i].dot(Phi_gq.T)
        # Nonlinear intensity
        lamda_n = link.eval(h_n)
        lamda_gq = link.eval(h_gq)

        if link.model == 'relu' and w[i].all() > 0 and Mj is not None:
            compensator = (link.theta + link.lamda * link.beta * (
                        w[i][0] - link.eta)) * T + link.lamda * link.beta * np.sum(
                w[i][1:] * Mj)

        else:
            compensator = lamda_gq.dot(w_gq)

        logl += np.sum(np.log(np.maximum(lamda_n, 1e-4))) - 1.0 * compensator

    return logl

