import numpy as np
from scipy.special import expit
import aesara.tensor as at

"""
Functions for PyMC4 (using aesara instead of theano)
"""


def grad_log_likelihood(w,  link, Phi_n, Phi_gq, w_gq, Mj=None, T=None):
    """
    Computes gradient of likelihood function at a parameter f (w)
    
    :param w (array): parameter f 
    :param link (LinkFunction): sigmoid, softplus or relu
    :param Phi_n (list of arrays): values of basis functions at process points (see class NonLinHawkesBasis)
    :param Phi_gq (array): values of basis functions at GQ points 
    :param w_gq (array): weights in GQ
    :param Mj (array): contribution of each bin for ReLU link
    :param T (float): time horizon
    :return: 1D array
    """

    number_of_dimensions = len(Phi_n)
    number_of_basis = ( Phi_n[0].shape[-1] - 1) // number_of_dimensions

    if w.ndim == 1:
        w = w.reshape(number_of_dimensions, number_of_dimensions * number_of_basis + 1)

    grad = np.empty_like(w)

    for i in range(number_of_dimensions):

        # Linear intensity
        h_n = w[i].dot(Phi_n[i].T)
        h_gq = w[i].dot(Phi_gq.T)

        # Nonlinear intensity
        lamda_n = link.eval(h_n)
        lamda_gq = link.eval(h_gq)

        if link.model == 'logit':
            # avoids numerical errors with large values
            tmp_gq = link.beta * h_gq
            tmp_n = link.beta * h_n
            l_q =  1.0 + ( tmp_gq < 50.0) * np.exp(- 1.0 * tmp_gq ) + (tmp_gq >= 50.0) * (1.0  - tmp_gq)
            l_n = 1.0 + (tmp_n < 50.0) * np.exp(- 1.0 * tmp_n) + (tmp_n >= 50.0) * (
                        1.0 - tmp_n)

            t_n = np.maximum(lamda_n * l_n, 1e-4)
            l_q = np.maximum(l_q, 1e-4)

            grad[i] = link.beta * Phi_n[i].T.dot(1.0 / t_n) - link.beta * link.lamda * w_gq.dot(Phi_gq / l_q.reshape(-1,1))

        elif link.model == 'relu':
            if w[i].all() > 0 and Mj is not None:
                # can compute compensator exactly with histograms and only excitation
                t1 =  link.beta * link.lamda * np.insert(Mj, 0, T)
            else:
                t1 = link.lamda * link.beta * w_gq.dot (Phi_gq * (h_gq.reshape(-1,1) > 0))

            t2 = np.sum(link.lamda * link.beta * Phi_n[i] / np.maximum(lamda_n, 1e-4).reshape(-1,1), axis=0)

            grad[i] = np.array(t2 - t1)


        elif link.model == 'sigmoid':
            grad[i] = link.beta * np.dot(expit(- link.beta * (h_n - link.eta)), Phi_n[i]) \
                   - (w_gq * lamda_gq * expit(- link.beta * (h_gq - link.eta))).dot(link.beta * Phi_gq)

        else:
            raise ValueError("Gradient not implemented")


    return grad.reshape(-1)



class LogLike_at(at.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [at.dmatrix]  # expects a matrix of parameter values when called
    otypes = [at.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike,  link, Phi_n, Phi_gq, w_gq, Mj=None, T=None):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.Phi_n = Phi_n
        self.Phi_gq = Phi_gq
        self.w_gq = w_gq
        self.link = link
        self.Mj = Mj
        self.T = T

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.link, self.Phi_n, self.Phi_gq, self.w_gq,
                               Mj=self.Mj, T=self.T)

        outputs[0][0] = np.array(logl)  # output the log-likelihood



class LogLikeWithGrad_at(at.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [at.dvector]  # expects a vector of parameter values when called
    otypes = [at.dscalar]  # outputs a scalar

    def __init__(self, loglike,  link, Phi_n, Phi_gq, w_gq, Mj=None, T=None):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.Phi_n = Phi_n
        self.Phi_gq = Phi_gq
        self.w_gq = w_gq
        self.link = link
        self.Mj = Mj
        self.T = T

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad_at(self.link, self.Phi_n, self.Phi_gq, self.w_gq,  Mj=Mj, T=T)


    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.link, self.Phi_n, self.Phi_gq, self.w_gq, Mj=self.Mj,
                               T=self.T)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad_at(at.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, link, Phi_n, Phi_gq, w_gq, Mj=None, T=None):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.Phi_n = Phi_n
        self.Phi_gq = Phi_gq
        self.w_gq = w_gq
        self.link = link
        self.Mj = Mj
        self.T = T

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # calculate gradients
        grads = grad_log_likelihood(theta, self.link, self.Phi_n, self.Phi_gq, self.w_gq,  Mj=self.Mj, T=self.T)

        outputs[0][0] = grads




