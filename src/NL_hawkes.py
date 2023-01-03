import numpy as np
import copy
from scipy.stats import beta, gamma, norm
from scipy.special import expit  # logistic function for arrays
from scipy.stats import expon
from scipy.stats import uniform
from scipy.stats import multinomial
from numpy.polynomial import legendre
import matplotlib.pyplot as plt
from scipy.special import digamma, gammaln
from polyagamma import random_polyagamma
import math
from utils import log_likelihood
import scipy
from sample import NHPPP_simulation
import itertools
from skfda.representation.basis import Fourier

""""

Class for simulation and inference on nonlinear Hawkes processes

3 Link functions:
    - ReLU
    - Sigmoid
    - Softplus

Basis for interaction functions:
    - histograms
    - Fourier basis

"""



class LinkFunction:
    """
    Class for link functions phi(x) = \theta + lamda * \psi(beta * (x - eta))
    """

    def __init__(self, model='relu', lamda=1.0, theta=0.0, beta=1.0, eta=0.0):
        self.model = model
        self.lamda = lamda
        self.theta = theta
        self.beta = beta
        self.eta = eta

        if model == 'sigmoid':
            fun = lambda x: theta + lamda * expit(beta * (x - eta))
        elif model == 'relu':
            fun = lambda x: theta + lamda * np.maximum(beta * (x - eta), 0.)
        elif model == 'softplus':
            # avoids overflow error with exp
            fun = lambda x: theta + lamda * (((beta * (x - eta)) < 50.0) * np.log(1.0 + np.exp(beta * (x - eta))) + \
                                             ((beta * (x - eta)) >= 50.0) * beta * (x - eta))
        elif model == 'exp':
            fun = lambda x: theta + np.minimum(lamda, np.exp(beta * (x - eta)))
        else:
            raise ValueError("Link function not implemented")

        self.fun = fun

    def eval(self, x):
        return self.fun(x)


class NonlinHawkesBasis:
    """
    Class for multivariate nonlinear Hawkes processes

    Interaction functions are specified via a basis function decomposition: histogram, Fourier or Beta densities

    Features:
        - Simulation

        - Inference

    """

    def __init__(self, number_of_dimensions, number_of_basis, A, model='sigmoid', basis='histogram', theta=0.0,
                 beta=1.0, eta=0.0, lamda=1.0):
        """

        :param number_of_dimensions (int): number of dimensions
        :param number_of_basis (int): number of basis functions
        :param T_phi (float): upper bound of function support
        :param model (string): type of link function (sigmoid, relu or softplus)
        :param basis (string): type of basis (histogram or Fourier)
        :param theta (float): parameter of link function (see LinkFunction)
        :param beta (float): parameter of link function (see LinkFunction)
        :param eta (float): parameter of link function (see LinkFunction)
        :param lamda (float): parameter of link function (see LinkFunction)
        """

        # General properties
        self.number_of_dimensions = number_of_dimensions  # K
        self.number_of_basis = number_of_basis  # B
        self.beta_ab = np.zeros((number_of_basis, 3))  # 3 HPs per beta density: shape, scale, location
        self.A = A
        self.basis = basis

        if basis == 'fourier':
            self.fourier_basis = Fourier(domain_range=[0, A], n_basis=number_of_basis, period=A)

        self.bins = np.linspace(0, A, number_of_basis + 1)
        self.model = model
        self.link = LinkFunction(model=model, lamda=lamda, theta=theta, beta=beta, eta=eta)

        # Parameters and estimates
        self.lamda_ub = np.zeros(number_of_dimensions)  # upper bound lambda star
        self.lamda_ub_estimated = None
        self.base_activation = np.zeros(number_of_dimensions)  # background rate nu
        self.base_activation_estimated = None
        self.weight = np.zeros((number_of_dimensions, number_of_dimensions,
                                number_of_basis))  # weight matrix of interaction functions K x K x B
        self.weight_estimated = None




    def set_hawkes_parameters(self, base_activation, weight, lamda_ub=None):
        """

        Set parameter of Hawkes process

        :param base_activation (1D numpy array): background rate
        :param weight (3D numpy array): weights of interaction functions in basis decomposition  number_of_dimensions x number_of_dimensions x number_of_basis
        :param lamda_ub: upper bound parameter in link function (only for sigmoid model)

        """

        # Raise ValueError if the given parameters do not have the right shape
        if (not lamda_ub is None) and (np.shape(lamda_ub) != (self.number_of_dimensions,)):
            raise ValueError('Intensity upperbounds have incorrect shape')
        if np.shape(base_activation) != (self.number_of_dimensions,):
            raise ValueError('Background rates have incorrect shape')
        if np.shape(weight) != (self.number_of_dimensions, self.number_of_dimensions, self.number_of_basis):
            raise ValueError('Weights have incorrect shape')

        # For link functions other than sigmoid, set to all one array
        if lamda_ub is None:
            self.lamda_ub = np.ones(self.number_of_dimensions)
        else:
            self.lamda_ub = copy.copy(lamda_ub)

        self.base_activation = copy.copy(base_activation)
        self.weight = copy.copy(weight)




    def intensity(self, t, target_dimension, timestamps_history):
        """
        Compute conditional intensity function for a component at a single timestamp, given the history of points.

        :param t (float): time stamp
        :param target_dimension (int): component at which the intensity is computed
        :param timestamps_history (list of floats): history of points
        :return (float): intensity value
        """

        # Raise ValueError if the given historical timestamps do not have the right shape
        if len(timestamps_history) != self.number_of_dimensions:
            raise ValueError('History of points have incorrect shape')

        # Parameter of interest for computing intensity
        base_activation_target_dimension = self.base_activation_estimated[target_dimension]
        weight_target_dimension = self.weight_estimated[target_dimension]

        # Find source components
        sources = np.arange(self.number_of_dimensions)[np.sum(np.abs(weight_target_dimension), axis=1) > 0]

        # Compute linear intensity
        linear_intensity = 0
        for n in sources:

            active_timestamps = np.array(timestamps_history[n])
            active_timestamps = active_timestamps[(active_timestamps < t) * (t - active_timestamps <= self.A)]

            if self.basis == 'histogram':

                histo = np.histogram(t - active_timestamps, bins=self.bins)[0]
                linear_intensity += np.sum(weight_target_dimension[n] * histo)

            elif self.basis == 'fourier':
                if len(active_timestamps) > 0:
                    f_t = np.squeeze(self.fourier_basis(t - active_timestamps))
                    linear_intensity += np.sum(weight_target_dimension[n].reshape(-1, 1) * f_t)

            else:
                raise ValueError("Basis not implemented")

        return self.link.eval(base_activation_target_dimension + linear_intensity)



    def simulation(self, T):
        """
        Function to simulate the process on the horizon [0,T]
        :param T (float): horizon
        :return (list of numpy arrays): observation of the process (points at each component)
        """

        t = 0

        # Initialise history
        points_hawkes = []
        for i in range(self.number_of_dimensions):
            points_hawkes.append([])

        while (t < T):

            # upper bound for thinning sampling
            if self.model == 'sigmoid':
                intensity_sup = self.number_of_dimensions * self.link.lamda
            else:
                h_max = [np.maximum(0, np.max(self.weight[d])) for d in range(self.number_of_dimensions)]
                N_eve = sum(
                    [np.sum(np.array(points_hawkes[d]) > (t - self.A)) for d in range(self.number_of_dimensions)])

                intensity_sup = np.sum(self.link.eval(np.array([self.base_activation[d] + N_eve * h_max[d]
                                                                for d in range(self.number_of_dimensions)])))

            r = expon.rvs(scale=1 / intensity_sup)
            t += r
            lambda_t = np.array([self.intensity(t, m, points_hawkes) for m in range(self.number_of_dimensions)])
            sum_intensity = np.sum(lambda_t)

            if self.link == 'sigmoid':
                assert sum_intensity <= intensity_sup, "intensity exceeds the upper bound"

            D = uniform.rvs(loc=0, scale=1)
            if D * intensity_sup <= sum_intensity:
                k = list(multinomial.rvs(1, list(lambda_t / sum_intensity))).index(1)
                points_hawkes[k].append(t)

        # Delete points outside of [0,T] if needed and convert to array
        for i in range(len(points_hawkes)):
            if (len(points_hawkes[i]) > 0) and (points_hawkes[i][-1] > T):
                del points_hawkes[i][-1]
            points_hawkes[i] = np.array(points_hawkes[i])

        return points_hawkes


    def Phi_t(self, t, points_hawkes, n_basis=None):
        """
        Compute  \Phi(t)=[1,\Phi_{11}(t),...,\Phi_{MB}(t)] where \Phi_{jb}(t) is the value at time stamp t
        of the b-th basis function at the j-th dimension.

        :param t (float): time stamp
        :param points_hawkes (list of floats): history of points
        :return (array)
        """

        # Raise ValueError if the given timestamps do not have the right shape
        if len(points_hawkes) != self.number_of_dimensions:
            raise ValueError('History has incorrect shape')
        Phi_t = [1]

        if n_basis is None:
            n_basis = self.number_of_basis

        bins = np.linspace(0, self.A, n_basis + 1)

        for i in range(self.number_of_dimensions):

            index = (np.array(points_hawkes[i]) < t) & ((t - np.array(points_hawkes[i])) <= self.A)

            if self.basis == 'fourier':
                f_t = self.fourier_basis(t - np.array(points_hawkes[i])[index])
                Phi_t = Phi_t + list(f_t[:, :, 0])

            elif self.basis == 'histogram':

                for j in range(n_basis):
                    b = np.digitize(t - np.array(points_hawkes[i])[index], bins) - 1
                    Phi_t.append(sum(b == j))

            else:
                raise ValueError("Basis not implemented")

        return np.array(Phi_t)



    def precompute_basis_values(self, points_hawkes, T, n_basis=None, num_gq=5000):
        """
        Function to precompute the values of the basis function at each point of the process and of the Gaussian quadrature

        :param points_hawkes (list): history of points
        :param T (float): horizon time
        :param num_gq (int): number of points in Gaussian quadrature
        :return list, array, array: Phi_n, Phi_wq, w_gq (weights of GQ)
        """

        # number of dimensions
        K = self.number_of_dimensions

        # number of basis
        if n_basis is None:
            n_basis = self.number_of_basis

        # number of points on each dimension
        N = np.array([len(points_hawkes[i]) for i in range(K)])

        # points and weights for GQ
        p_gq, w_gq = gq_points_weights(0, T, num_gq)

        # Precompute vector Phi at each process point
        Phi_n = [np.zeros((N[d], K * n_basis + 1)) for d in range(K)]
        for d in range(K):
            for n in range(N[d]):
                Phi_n[d][n] = self.Phi_t(points_hawkes[d][n], points_hawkes, n_basis=n_basis)

        # Precompute vector Phi at each point of Gaussian quadrature
        Phi_gq = np.zeros((num_gq, K * n_basis + 1))
        for m in range(num_gq):
            Phi_gq[m] = self.Phi_t(p_gq[m], points_hawkes, n_basis=n_basis)

        return Phi_n, Phi_gq, w_gq



    def log_likelihood(self, Phi_n, Phi_gq, points_hawkes, w_gq):
        """
        Compute Log likelihood function of the observation

        :param Phi_n: precomputed values of functions at points
        :param Phi_gq: precomputed values of functions at GQ points
        :param w_gq: GQ weights
        :return: float
        """

        # Raise ValueError if the given timestamps do not have the right shape
        if len(points_hawkes) != self.number_of_dimensions:
            raise ValueError('History has incorrect shape')
        if np.shape(Phi_gq) != (len(w_gq), self.number_of_dimensions * self.number_of_basis + 1):
            raise ValueError('Dimensions of Phi_gq or w_gq are incorrect')
        for i in range(self.number_of_dimensions):
            if len(Phi_n[i]) != len(points_hawkes[i]):
                raise ValueError('Dimension of Phi_n is incorrect')

        logl = 0.0
        for i in range(self.number_of_dimensions):
            w = self.weight[i]

            # Linear intensity
            h_n = w.dot(Phi_n[i].T)
            h_gq = w.dot(Phi_gq.T)

            # Nonlinear intensity
            lamda_n = self.link.eval(h_n)
            lamda_gq = self.link.eval(h_gq)

            compensator = lamda_gq.dot(w_gq)

            logl += np.sum(np.log(lamda_n)) - 1.0 * compensator

        return logl


    ####################### Inference methods


    # Gibbs sampler for sigmoid Hawkes processes

    def Gibbs_sampler(self, points_hawkes, T, num_iter, init_w=None, mu=None, K=None):
        """
        Gibbs sampler with data augmentation in the sigmoid Hawkes model

        :param points_hawkes  (list of floats): observed points
        :param T  (float): horizon
        :param num_iter  (int): number of Gibbs iterations
        :param init_w (numpy.ndarray): initial value of h_k in the chain
        :param mu (numpy.ndarray): mean vector of normal prior on h_k dim: number_of_dimension*number_of_basis + 1
        :param K  (numpy.ndarray): covariance matrix of normal prior on w dim: (number_of_dimension*number_of_basis + 1) x (number_of_dimension*number_of_basis + 1)
        :return (numpy.ndarray): posterior samples of f = (nu, h).
        """

        # Raise ValueError if the link function is not the sigmoid
        if self.model != 'sigmoid':
            raise ValueError('Gibbs sampler cannot be applied to nonlinear Hawkes processes with ReLU or softplus links')

        print("Initialisation...")

        # number of points on each dimension
        N = np.array([len(points_hawkes[i]) for i in range(self.number_of_dimensions)])

        # Precompute vector Phi at each process point
        Phi_n = [np.zeros((N[d], self.number_of_dimensions * self.number_of_basis + 1)) for d in
                 range(self.number_of_dimensions)]
        for d in range(self.number_of_dimensions):
            for n in range(N[d]):
                Phi_n[d][n] = self.Phi_t(points_hawkes[d][n], points_hawkes)

        # Default prior mean and covariance if not specified
        if mu is None:
            mu = np.zeros(self.number_of_dimensions * self.number_of_basis + 1)
        if K is None:
            K = np.eye(self.number_of_dimensions * self.number_of_basis + 1)

        # Initialization of w
        if init_w is None:  # from prior
            w = np.random.multivariate_normal(mean=mu,
                                              cov=K,
                                              size=self.number_of_dimensions)  # dim = K * (KB + 1)
        else:
            w = copy.copy(init_w)

        # Initialisation of linear intensity: at the process points
        H_n = [w[d].dot(Phi_n[d].T) for d in range(self.number_of_dimensions)]  # H_n[d] has dim N[d]

        # Initialisation of auxiliary variables
        omega_n = [np.array([])] * self.number_of_dimensions  # marks of original process at event points

        # Initialisation of auxiliary marked PPP
        ppp_times = [[]] * self.number_of_dimensions
        ppp_marks = [[]] * self.number_of_dimensions

        # Posterior samples
        post_w = [copy.copy(w)]

        print("Starting Gibbs iterations...")

        # Gibbs iterations
        for ite in range(num_iter):

            for d in range(self.number_of_dimensions):

                # sample auxiliary marks
                H_n[d] = w[d].dot(Phi_n[d].T)
                omega_n[d] = random_polyagamma(h=np.ones(N[d]), z=self.link.beta * (H_n[d] - self.link.eta))

                # sample Poisson Point Process via thinning
                ppp_times[d] = []
                # Sample a number of points
                J = np.random.poisson(self.link.lamda * T)

                # Sample J times uniformly in [0,T]
                times = np.random.uniform(0, T, size=J)

                # Accept or reject points
                rdm = np.random.random(size=J)
                intensity = []
                for j, t in enumerate(times):
                    linear_intensity = 0
                    for i in range(len(points_hawkes[d])):
                        if points_hawkes[d][i] >= t:
                            break
                        elif t - points_hawkes[d][i] > self.A:
                            continue
                        if self.basis == 'histogram':
                            b = np.digitize(t - points_hawkes[d][i], self.bins) - 1
                            linear_intensity += w[d][b + 1]

                        elif self.basis == 'fourier':
                            f_t = np.squeeze(self.fourier_basis(t - points_hawkes[d][i]))
                            linear_intensity += np.sum(w[d].reshape(-1, 1) * f_t)

                        else:
                            raise ValueError("Basis not implemented")

                    intensity.append(expit(- 1.0 * self.link.beta * (w[d][0] + linear_intensity - self.link.eta)))

                intensity = np.array(intensity)
                ppp_times[d] = times[intensity > rdm]

                R = len(ppp_times[d])

                # Compute values of functions at each point of the PPP
                Phi_times = np.zeros((R, self.number_of_dimensions * self.number_of_basis + 1))
                for t in range(R):
                    Phi_times[t] = self.Phi_t(ppp_times[d][t], points_hawkes)
                H_times = w[d].dot(Phi_times.T)
                H_times.reshape(R)

                if Phi_times.shape[0] < 2:
                    H_times = np.array([H_times])

                # Sample marks
                ppp_marks[d] = random_polyagamma(h=np.ones(R), z=self.link.beta * (H_times - self.link.eta).reshape(-1))

                # Compute mu and sigma (mean and covariance of posterior over w)
                if len(H_times) > 0:
                    Phi_times = np.repeat(Phi_times, 1, axis=0)
                    ppp_marks[d] = np.repeat(ppp_marks[d], 1)
                    Phi = np.transpose(np.concatenate([Phi_n[d], Phi_times], axis=0), (1, 0))
                    D = np.diag(np.concatenate([omega_n[d], ppp_marks[d]], axis=0))
                    v = 0.5 * self.link.beta * np.concatenate([np.ones(N[d]), - np.ones(R)], axis=0) \
                        + (self.link.beta ** 2) * self.link.eta * np.concatenate([omega_n[d], ppp_marks[d]], axis=0)

                else:
                    Phi = np.transpose(Phi_n[d], (1, 0))
                    D = np.diag(omega_n[d])
                    v = 0.5 * self.link.beta * np.ones(N[d]) + (self.link.beta ** 2) * self.link.eta * omega_n[d]

                Sigma = np.linalg.inv((self.link.beta ** 2) * Phi.dot(D).dot(Phi.T) + np.linalg.inv(K))
                Mu = Sigma.dot(Phi.dot(v) + np.dot(np.linalg.inv(K), mu))

                # Sample w
                w[d] = np.random.multivariate_normal(mean=Mu, cov=Sigma)

                if ite % 50 == 0:
                    print("Iteration ", ite)
                    print(f"w: {w}")
                    print(f"Number of points in N: {N[d]} and in PPP: {R}")

            post_w.append(copy.copy(w))

        return post_w



    # Adaptive Mean-Field Variational algorithm: also only for sigmoid Hawkes processes


    def adaptive_MF_VI(self, points_hawkes, T, num_gq, num_iter=100, mean=0.0, sigma=5.0, depth=3, graph=None):
        """
        Compute mean-field Gaussian variational posterior

        :param points_hawkes  (list of floats): observed points
        :param T  (float): horizon
        :param num_gq (int): number of Gaussian quadrature points in [0,T]
        :param num_iter  (int): number of VI iterations
        :param mean (float): mean of normal prior on each parameter
        :param sigma (float): variance of normal prior on each parameter
        :param depth (int): number of levels of the nested histogram prior
        :paran graph (numpy.ndarray): graph of interaction of the process dim: number_of_dimensions x number_of_dimensions
        :return: nothing (everything saved in Hawkes model)
        """

        # Raise ValueError if the link function is not the sigmoid
        if self.model != 'sigmoid':
            raise ValueError(
                'Gibbs sampler cannot be applied to nonlinear Hawkes processes with ReLU or softplus links')

        print("Initialisation...")

        # Enumerate "models" (graph and number of basis function)
        if graph is None:
            S = np.array(np.meshgrid(*[np.array([0, 1])] * (self.number_of_dimensions ** 2))).reshape(
                self.number_of_dimensions, self.number_of_dimensions, -1)
            S = np.transpose(S, axes=(2, 0, 1))  # 3D array of graph parameters (n_models x K x K)
            n_graphs = S.shape[0]
            # first graph parameter is empty -> needs to exclude it in the cross product with the number of basis
            set_of_models = list(itertools.product(list(S[1:, :, :]), list(2 ** np.arange(depth + 1))))
            set_of_models = [(S[0], 0)] + set_of_models

        else:
            set_of_models = list(itertools.product([graph], list(2 ** np.arange(depth + 1))))
            n_graphs = 1

        n_models = len(set_of_models)

        print(f"Number of graph parameters {n_graphs} and 'models' {n_models}")

        self.set_of_models = set_of_models

        # points and weights for GQ for all numbers of basis functions
        list_phi_n, list_phi_gq = [], []

        # Precompute basis functions
        for i in range(depth + 1):
            Phi_n, Phi_gq, w_gq = self.precompute_basis_values(self, points_hawkes, T, n_basis=2 ** i, num_gq=num_gq)
            list_phi_n.append(Phi_n)
            list_phi_gq.append(Phi_gq)

            # print("Shape should be nb of points x number of dimensions * number of basis functions + 1")
            # for i in range(self.number_of_dimensions):
            #     print(Phi_n[i].shape)
            # print("Shape should be nb of GQ points x number of dimensions * number of basis functions + 1")
            # print(Phi_gq.shape)


        # Save values along iterations
        self.hist_vi_int_intensity = []
        self.hist_vi_mean_nu_vposterior = []
        self.hist_vi_mean_weight_vposterior = []
        self.hist_vi_cov_weight_vposterior = []
        self.hist_elbo = []
        self.hist_loglik = []

        # Final values at each 'model'
        self.int_intensity = []
        self.mean_nu_vposterior = []
        self.mean_weight_vposterior = []
        self.cov_vi = []
        self.elbo = []  # values of ELBO for each model
        self.vpost_proba = []  # proba a posteriori for each model
        self.log_lik = []

        print("Starting MF-VI...")

        for i, m in enumerate(set_of_models):

            n_basis = m[1]
            graph = m[0]

            print(f"Model with graph parameter: {graph} and {n_basis} basis functions")

            if n_basis > 0:
                Phi_n, Phi_gq = list_phi_n[np.log2(n_basis).astype(np.int64)], list_phi_gq[
                    np.log2(n_basis).astype(np.int64)]
            else:
                Phi_n, Phi_gq, w_gq = self.precompute_basis_values(self, points_hawkes, T, n_basis=0,
                                                                   num_gq=num_gq)

            # History of algorithm for model m
            hist_vi_int_intensity = []
            hist_vi_mean_nu_vposterior = []
            hist_vi_mean_weight_vposterior = []
            hist_vi_cov_weight_vposterior = []
            hist_elbo = []
            hist_llik = []

            masks = []
            for d in range(self.number_of_dimensions):
                if n_basis > 1:
                    mask = graph[d].reshape(-1, 1) * np.ones((self.number_of_dimensions, n_basis))  # non-null weights
                    # print(mask)
                    mask = np.insert(mask.reshape(-1), 0, 1)
                elif n_basis == 1:
                    mask = graph[d] * np.ones(self.number_of_dimensions)
                    # print(mask)
                    mask = np.insert(mask.reshape(-1), 0, 1)
                else:
                    mask = np.ones(1)
                masks.append(mask)

            # initial mean and covariance of w (=h_k), i.e. prior mean and variance
            mu = mean * np.zeros(self.number_of_dimensions * n_basis + 1)
            K = sigma * np.eye(self.number_of_dimensions * n_basis + 1)
            E_w = [masks[d] * mu for d in range(self.number_of_dimensions)]
            Cov_w = [np.outer(masks[d], masks[d]) * K for d in range(self.number_of_dimensions)]

            H_n_lin = [E_w[d].dot(Phi_n[d].T) for d in range(self.number_of_dimensions)]
            H_gq_lin = [E_w[d].dot(Phi_gq.T) for d in range(self.number_of_dimensions)]

            H_n_bar = [self.link.beta * (H_n_lin[d] - self.link.eta) for d in
                       range(self.number_of_dimensions)]
            H_gq_bar = [self.link.beta * (H_gq_lin[d] - self.link.eta) for d in range(self.number_of_dimensions)]

            E_H2_n = [np.square(H_n_lin[d]) - 2.0 * self.link.eta * H_n_lin[d] + self.link.eta ** 2
                      + np.diag(Phi_n[d].dot(Cov_w[d].dot(Phi_n[d].T))) for d in
                      range(self.number_of_dimensions)]  # square expectation
            E_H2_gq = [np.square(H_gq_lin[d]) - 2.0 * self.link.eta * H_gq_lin[d] + self.link.eta ** 2 \
                       + np.diag(Phi_gq.dot(Cov_w[d].dot(Phi_gq.T))) for d in range(self.number_of_dimensions)]

            H_gq_tilde = [self.link.beta * np.sqrt(E_H2_gq[d]) for d in range(self.number_of_dimensions)]
            H_n_tilde = [self.link.beta * np.sqrt(E_H2_n[d]) for d in range(self.number_of_dimensions)]

            E_omega_n = [0.5 / H_n_tilde[d] * np.tanh(H_n_tilde[d] / 2.) for d in range(self.number_of_dimensions)]
            E_omega_gq = [0.5 / H_gq_tilde[d] * np.tanh(H_gq_tilde[d] / 2.) for d in range(self.number_of_dimensions)]

            # compensator of the augmented Poisson process over [0,T]
            int_intensity = np.zeros(self.number_of_dimensions)

            # Initialise ELBO
            elbo = -1.0 * np.Inf

            # Start iterations
            for ite in range(num_iter):

                if ite % 1 == 0:
                    print("Iteration ", ite)
                    print(f"ELBO: {elbo}")

                elbo = 0.0

                for d in range(self.number_of_dimensions):

                    # update H_n and omega_n: value of linear intensity and marke at the process points
                    H_n_lin[d] = (masks[d] * E_w[d]).dot(Phi_n[d].T)
                    E_H2_n[d] = np.square(H_n_lin[d]) - 2.0 * self.link.eta * H_n_lin[d] + (
                                self.link.eta ** 2) + np.diag(
                        Phi_n[d].dot(Cov_w[d].dot(Phi_n[d].T)))
                    H_n_tilde[d] = self.link.beta * np.sqrt(E_H2_n[d])
                    H_n_bar[d] = self.link.beta * (H_n_lin[d] - self.link.eta)

                    # expectation of augmented marks at each point (expectation of Polya Gamma dist)
                    E_omega_n[d] = 0.5 / H_n_tilde[d] * np.tanh(0.5 * H_n_tilde[d])

                    # update Poisson intensity: values at the GQ points
                    H_gq_lin[d] = E_w[d].dot(Phi_gq.T)
                    H_gq_bar[d] = self.link.beta * (H_gq_lin[d] - self.link.eta)
                    E_H2_gq[d] = np.square(H_gq_lin[d]) - 2.0 * self.link.eta * H_gq_lin[
                        d] + self.link.eta ** 2 + np.diag(Phi_gq.dot(Cov_w[d].dot(Phi_gq.T)))
                    H_gq_tilde[d] = self.link.beta * np.sqrt(E_H2_gq[d])
                    E_omega_gq[d] = 0.5 / H_gq_tilde[d] * np.tanh(0.5 * H_gq_tilde[d])

                    # Evaluate ELBO
                    if sum(graph[d]) == 0.0:
                        idx = np.zeros(1).astype(int)
                    else:
                        idx = np.where(masks[d] == 1)[0]
                    K_model = K[idx, :][:, idx]
                    Cov_w_model = Cov_w[d][idx, :][:, idx]

                    # update mean and covariances of weights
                    int_A = np.zeros((self.number_of_dimensions * n_basis + 1,
                                      self.number_of_dimensions * n_basis + 1))

                    int_A += self.link.beta ** 2 * np.sum(E_omega_n[d].reshape(-1, 1, 1) * (
                                np.expand_dims(Phi_n[d], axis=2) * np.expand_dims(Phi_n[d], axis=1)), axis=0)
                    int_A += self.link.lamda * (self.link.beta ** 2) * np.sum(w_gq.reshape(-1, 1, 1) * (
                            E_omega_gq[d].reshape(-1, 1, 1) * expit(-H_gq_tilde[d].reshape(-1, 1, 1)) * np.exp(
                        0.5 * (H_gq_tilde[d] - H_gq_bar[d]).reshape(-1, 1, 1))
                            * (np.expand_dims(Phi_gq, axis=2) * np.expand_dims(Phi_gq, axis=1))), axis=0)

                    Cov_w[d] = np.linalg.inv(int_A + np.linalg.inv(K)) * np.outer(masks[d], masks[d])

                    int_B = np.zeros(self.number_of_dimensions * n_basis + 1)
                    int_B += 0.5 * self.link.beta * np.sum(
                        Phi_n[d] * (1.0 + 2.0 * self.link.eta * self.link.beta * E_omega_n[d]).reshape(-1, 1),
                        axis=0)  # first term of integral of B
                    int_B += 0.5 * self.link.beta * self.link.lamda * np.sum(w_gq.reshape(-1, 1) * (
                                (- 1.0 + 2.0 * self.link.beta * self.link.eta * E_omega_gq[d].reshape(-1, 1))
                                * expit(-H_gq_tilde[d].reshape(-1, 1))
                                * np.exp(0.5 * (H_gq_tilde[d] - H_gq_bar[d]).reshape(-1, 1)) * Phi_gq), axis=0)

                    # third term (prior)
                    int_B += np.dot(np.linalg.inv(K), mu)

                    E_w[d] = scipy.linalg.solve(int_A + np.linalg.inv(K), int_B) * masks[d]  # Cov_w[d].dot(int_B)

                    # Update ELBO
                    elbo += 0.5 * K_model.shape[0]
                    elbo -= 0.5 * np.log(np.linalg.det(2 * math.pi * K_model))
                    elbo += 0.5 * np.log(np.linalg.det(2 * math.pi * Cov_w_model))
                    elbo -= 0.5 * (np.trace(np.linalg.inv(K_model).dot(Cov_w_model)) + E_w[d].dot(np.linalg.inv(K)).dot(
                        E_w[d]))

                    elbo += np.sum(0.5 * H_n_bar[d] - np.log(2))
                    elbo += np.sum(np.log(self.link.lamda) - np.log(np.cosh(0.5 * H_n_tilde[d])))

                    # check these computations
                    int_ppp_gq = np.exp(- 0.5 * H_gq_bar[d]) / 2.0 / np.cosh(0.5 * H_gq_tilde[d])

                    elbo += self.link.lamda * w_gq.dot(int_ppp_gq)
                    elbo -= self.link.lamda * T

                loglik = log_likelihood(np.array(E_w), self.link, Phi_n, Phi_gq, w_gq)

                # Save values at iteration
                hist_vi_int_intensity.append(copy.copy(int_intensity))
                hist_vi_mean_nu_vposterior.append(copy.copy(np.array(E_w)[:, 0]))
                hist_vi_mean_weight_vposterior.append(copy.copy(np.array(E_w)[:, 1:]))
                hist_vi_cov_weight_vposterior.append(copy.copy(Cov_w))
                hist_elbo.append(copy.copy(elbo))
                hist_llik.append(loglik)


            # save final values for this model
            self.int_intensity.append(hist_vi_int_intensity[-1])
            self.mean_nu_vposterior.append(hist_vi_mean_nu_vposterior[-1])
            self.mean_weight_vposterior.append(hist_vi_mean_weight_vposterior[-1])
            self.cov_vi.append(hist_vi_cov_weight_vposterior[-1])
            self.elbo.append(hist_elbo[-1])
            self.log_lik.append(hist_llik[-1])

            self.hist_vi_int_intensity.append(hist_vi_int_intensity)
            self.hist_vi_mean_nu_vposterior.append(hist_vi_mean_nu_vposterior)
            self.hist_vi_mean_weight_vposterior.append(hist_vi_mean_weight_vposterior)
            self.hist_vi_cov_weight_vposterior.append(hist_vi_cov_weight_vposterior)
            self.hist_elbo.append(hist_elbo)
            self.hist_loglik.append(hist_llik)

        self.elbo = np.array(self.elbo)
        self.vpost_proba = np.exp(self.elbo - np.max(self.elbo))
        self.vpost_proba = self.vpost_proba / np.sum(self.vpost_proba)


