"""

Sigmoidal Hawkes model in 2D with unknown number of basis

"2-step adaptive" MF-VI algorithm

1. Estimate kernel norms with full graph

2. Threshold to select graph and re-run VI

"""

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
module_path_2 = os.path.abspath(os.path.join('../..'))
module_path_3 = os.path.abspath(os.path.join('./Scalable_VB_Hawkes/'))
module_path_4 = os.path.abspath(os.path.join('./NHGPS/'))
module_path_5 = os.path.abspath(os.path.join('/'))
if module_path not in sys.path:
    sys.path.append(module_path)
if module_path_2 not in sys.path:
    sys.path.append(module_path_2)
if module_path_3 not in sys.path:
    sys.path.append(module_path_3)
if module_path_4 not in sys.path:
    sys.path.append(module_path_4)
if module_path_5 not in sys.path:
    sys.path.append(module_path_5)
print(os.getcwd())

from src.NL_hawkes import NonlinHawkesBasis, LinkFunction
from src.utils import nb_excursions, gq_points_weights, nd_block_diag, intensity_process, NpEncoder, log_likelihood, \
    exp_f_norm_histo, f_norm_histo
from src.plot import plot_graph, plot_functions_norms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # for plot
from scipy.special import expit
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
from datetime import datetime
import json
from pathlib import PosixPath
from types import SimpleNamespace
from time import time
import pickle
import aesara.tensor as at
import arviz as az


if __name__ == "__main__":

    ############ General simulation parameters

    list_dimensions = [10]
    number_of_dimensions = 10
    number_of_bins=2 # choose larger than 2 for fourier basis
    A=0.1 # support of basis functions
    basis_true='histogram'
    basis_estimation='histogram'
    #list_T=[50, 150, 300]
    list_T = [300]
    T_init = 50 # time for burning
    T_max = list_T[-1] + T_init
    list_links = ['relu', 'softplus', 'unknown_sigmoid'] # choices: relu, softplus, sigmoid, unknown_sigmoid for simulating data in the misspecified case
    #list_links = ['sigmoid'] # well-specified case
    estimate_lamda = False

    ######### Link function settings

    #model = {"theta": 0.0, "beta": 1.0, "eta": 0.0, "lamda":1.0 }
    nl_models = {
        "relu": {"theta": 0.0001, "beta": 1., "eta": 0.0, "lamda": 1.0},
        "softplus": {"theta": 0.0001, "beta": 1.0, "eta": 0.0, "lamda":1.0 },
        "sigmoid": {"theta": 0.0, "beta": 0.2, "eta": 10.0, "lamda": 20.0},
        "unknown_sigmoid": {"theta": 0.0, "beta": 0.2, "eta": 10.0, \
                            "lamda": np.random.uniform(low=10.0, high=30.0, size=number_of_dimensions)},
    }

    ############ Inference parameters
    num_gq = 20000 #np.minimum(2 * np.int64(number_of_bins * T / A) , 20000) # nb of points in Gaussian quadrature
    num_iter = 150 # number of iterations of the MF-VI algo
    mu = 0.0 # mean of normal prior
    sigma = 5.0 # variance of normal prior
    depth = 3 # number of nested histograms tested (depth + 2)
    #threshold = 0.07 # for thresholding l1 norms

    w01 = np.linspace(5, 2, num=number_of_bins).reshape(1, -1)
    w02 = np.linspace(4, 2, num=number_of_bins).reshape(1, -1)
    w03 = np.linspace(-4, -1, num=number_of_bins).reshape(1, -1)

    #for T in list_T:
    for j in range(len(list_dimensions)):

        #number_of_dimensions = list_dimensions[j]

        base_activation = np.random.uniform(low=5.0, high=7.0, size=number_of_dimensions)

        # connected graph with the 2 first main diagonals
        graph = np.eye(number_of_dimensions) + np.eye(number_of_dimensions, k=1)

        # plot true graph
        #plot_graph(graph, save_dir=save_dir / f'true_graph.pdf')

        w_exc = np.zeros((number_of_dimensions, number_of_dimensions, number_of_bins))
        w_inh = np.zeros((number_of_dimensions, number_of_dimensions, number_of_bins))
        for i in range(number_of_dimensions):
            w_exc[i, i, :] = w01
            w_inh[i, i, :] = w03
            if i < number_of_dimensions - 1:
                w_exc[i, i + 1, :] = w02
                w_inh[i, i + 1, :] = w02

        full_graph = np.ones((number_of_dimensions, number_of_dimensions), dtype=np.int64)

        weight = {
            'exc': w_exc,
            'inh': w_inh,
        }

        N_param = len(base_activation) + (number_of_dimensions * number_of_dimensions * number_of_bins)
        print("Number of unknown parameters : ", N_param)

        for link in list_links:

            # if estimate_lamda:
            #     lamda_ub = np.random.uniform(low=1.0, high=2.0, size=number_of_dimensions)
            # else:
            #     lamda_ub = true_link.lamda * np.ones(number_of_dimensions)

            true_link = nl_models[link]  # link function for simulating the process
            inference_link = nl_models['sigmoid']  # link function for inference

            true_link = SimpleNamespace(**true_link)
            inference_link = SimpleNamespace(**inference_link)

            for scen in weight:

                sparse_weight = np.expand_dims(graph, axis=2) * weight[scen]

                t0 = time()

                ########## Initialise and simulate from model
                true_model = NonlinHawkesBasis(number_of_dimensions, number_of_bins, A, model=link, basis=basis_true,
                                              theta=true_link.theta, beta=true_link.beta, lamda=true_link.lamda,\
                                               eta=true_link.eta)
                true_model.set_hawkes_parameters(base_activation, sparse_weight)

                # Simulate process
                points_hawkes_tot = true_model.simulation(T=T_max)
                # Remove "initial condition" to observe stationary distribution
                T_max = T_max - T_init
                for d in range(len(points_hawkes_tot)):
                    points_hawkes_tot[d] = points_hawkes_tot[d][points_hawkes_tot[d] > T_init] - T_init
                # time for simulating the process
                t_simu = time() - t0
                print(f'Time for simulating the process: {t_simu} sec')

                for r,T in enumerate(list_T):

                    # Create subdirectory for results
                    save_dir = (
                        f'/data/hylacola/sulem/PycharmProjects/Scalable_VB_Hawkes/results/synthetic_data/{datetime.utcnow().strftime("%m_%d_%H:%M:%S")}_{link}_D{number_of_dimensions}_T_{T}_{scen}/')
                    save_dir = PosixPath(save_dir).expanduser()
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)

                    points_hawkes = []
                    for d in range(number_of_dimensions):
                        points_hawkes.append( points_hawkes_tot[d][(points_hawkes_tot[d] < T)])

                    # number of points on each dimension
                    N = np.array([len(points_hawkes[i]) for i in range(number_of_dimensions)])
                    N_exc = nb_excursions(points_hawkes, A)

                    print(f"T = {T}")
                    print(f"Number of events on each dimension: ", N)
                    print(f"Number of excursions : ", N_exc)

                    # Create subdirectory for data
                    save_dir_data = (
                        f'/data/hylacola/sulem/PycharmProjects/Scalable_VB_Hawkes/data/synthetic_data/dataset_{link}_histo_dim{number_of_dimensions}_{scen}/')
                    save_dir_data = PosixPath(save_dir_data).expanduser()
                    if not os.path.isdir(save_dir_data):
                        os.makedirs(save_dir_data)

                    # save points
                    with open(save_dir_data / f'points.p', 'wb') as fp:
                        pickle.dump(points_hawkes, fp)

                    data_config = {
                        "n_dim": number_of_dimensions,
                        "n_bas": number_of_bins,
                        "basis_true": basis_true,
                        "graph": graph,
                        "basis_estimation": basis_estimation,
                        "link": link,
                        "theta": true_link.theta,
                        "lamda_ub": true_link.lamda,
                        "eta": true_link.eta,
                        "beta": true_link.beta,
                        "base_activation": base_activation,
                        "weights": sparse_weight,
                        "T": T,
                        "N_eve": N,
                        "A": A,
                        "N_excursions": N_exc,
                        "num_gq": num_gq,
                        "t_simu": t_simu
                    }

                    with open(save_dir_data / f'data_config.json',
                              'w') as fp:
                        json.dump(data_config, fp, indent=2, cls=NpEncoder)

                    ###### ADAPTIVE VI IN FULL GRAPH
                    t0 = time()
                    hawkes_model = NonlinHawkesBasis(number_of_dimensions, number_of_bins, A, model='sigmoid', basis=basis_estimation,
                                                  theta=inference_link.theta, beta=inference_link.beta, lamda=inference_link.lamda,
                                                     eta=inference_link.eta)
                    hawkes_model.set_hawkes_parameters(base_activation, sparse_weight)

                    if estimate_lamda:
                        mean_nu_1, mean_weight_1, cov_1, lamda_ub_1 = hawkes_model.AdaptiveVB(points_hawkes, T=T,
                                                                                                 num_gq=num_gq, num_iter=num_iter,
                                                                                                 sigma=sigma, depth=depth,
                                                                                                 estimate_lambda=True,
                                                                                                 graph=full_graph)
                    else:
                        mean_nu_1, mean_weight_1, cov_1 = hawkes_model.AdaptiveVB(points_hawkes, T=T,
                                                                                                 num_gq=num_gq, num_iter=num_iter,
                                                                                                 sigma=sigma, depth=depth,
                                                                                                 estimate_lambda=False,
                                                                                                 graph=full_graph)

                    t_vi = time() - t0
                    mean_weight_1 = np.array(mean_weight_1).reshape(number_of_dimensions, number_of_dimensions, -1)
                    model_1 = hawkes_model.set_of_models[np.argmax(hawkes_model.vpost_proba)]

                    # ELBO along iterations
                    burning = 5
                    fig, ax = plt.subplots(1, len(hawkes_model.hist_elbo),
                                           figsize=(4 * len(hawkes_model.hist_elbo), 3))
                    for i in range(len(hawkes_model.hist_elbo)):
                        if len(hawkes_model.hist_elbo) == 1:
                            a = ax
                        else:
                            a = ax[i]
                        a.set_title(f'B = {2 ** i}')
                        a.plot(range(burning, len(hawkes_model.hist_elbo[i])), hawkes_model.hist_elbo[i][burning:])
                        a.set_ylabel(r"ELBO")
                        a.set_xlabel("iteration")
                    plt.savefig(save_dir / f"elbo.pdf")

                    # Estimated norms
                    estimated_norms = exp_f_norm_histo(mean_weight_1, cov_1, A)
                    ordered_norms = np.sort(estimated_norms.flatten())
                    true_norms = f_norm_histo(sparse_weight, A)
                    vmax = max(np.max(true_norms), np.max(estimated_norms))

                    # plot diff with truth
                    # plot_functions_norms(np.abs(estimated_norms - true_norms))
                    # plt.savefig(save_dir / f"sigmoid_{scen}_D{number_of_dimensions}_abs_err_norms.pdf")

                    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

                    a = ax[0]
                    heatmap = a.pcolor(true_norms, cmap='Oranges', vmax=vmax)
                    node_names = range(1, number_of_dimensions + 1)
                    column_labels = [i for i in node_names]
                    row_labels = [i for i in node_names]
                    a.set_xticks(np.arange(estimated_norms.shape[0]) + 0.5, minor=False)
                    a.set_yticks(np.arange(estimated_norms.shape[1]) + 0.5, minor=False)
                    a.invert_yaxis()
                    a.xaxis.tick_top()
                    a.set_xticklabels(row_labels, minor=False, fontsize=8)
                    a.set_yticklabels(column_labels, minor=False, fontsize=8)
                    fig.colorbar(heatmap, ax=a)

                    a = ax[1]
                    heatmap = a.pcolor(estimated_norms, cmap='Oranges', vmax=vmax)
                    node_names = range(1, number_of_dimensions + 1)
                    column_labels = [i for i in node_names]
                    row_labels = [i for i in node_names]
                    a.set_xticks(np.arange(estimated_norms.shape[0]) + 0.5, minor=False)
                    a.set_yticks(np.arange(estimated_norms.shape[1]) + 0.5, minor=False)
                    a.invert_yaxis()
                    a.xaxis.tick_top()
                    a.set_xticklabels(row_labels, minor=False, fontsize=8)
                    a.set_yticklabels(column_labels, minor=False, fontsize=8)
                    fig.colorbar(heatmap, ax=a)

                    a = ax[2]
                    a.set_ylabel(r"$L_1$-norm")
                    a.set_xlabel("function index")
                    a.set_xticks([])
                    a.scatter(range(len(ordered_norms)), ordered_norms, marker='+', s=50)
                    # a.hlines(xmin=0, xmax=len(ordered_norms), y=threshold, label=r"$\eta_0$", linestyles='dotted', color='r',
                    #          linewidth=3)
                    a.grid(visible=True)
                    a.legend()

                    plt.savefig(save_dir / f"L1_norms.pdf")

                    # Re run VI with selected graph
                    # hawkes_model.adaptive_MF_VI(points_hawkes, T=T, num_gq=num_gq, num_iter=num_iter, sigma=sigma, depth=depth,
                    #                             graph=selected_graph)
                    # t_vi = time() - t0
                    # print(f'Time of VI algo: {t_vi} sec')

                    # find MAP model
                    # m_hat = np.log2(hawkes_model.set_of_models[np.argmax(hawkes_model.vpost_proba)][1]).astype(np.int64)
                    #
                    # t0 = time()
                    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
                    # ax.set_title("ELBO along iterations for several models")
                    # ax.set_xlabel('iteration')
                    # ax.set_ylabel('ELBO')
                    # ax.grid()
                    # for i in range(len(hawkes_model.hist_elbo)):
                    #     ax.plot(hawkes_model.hist_elbo[i], label=f"m{i}")
                    # ax.legend()
                    # plt.savefig(save_dir / f"sigmoid_{scen}_D{number_of_dimensions}_diff_{r}_elbo.pdf")
                    #
                    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
                    # ax.set_title("Variational posterior probabilities on the models")
                    # x = [f'm{i}' for i in range(len(hawkes_model.elbo))]
                    # ax.bar(x=x, height=hawkes_model.vpost_proba)
                    # plt.savefig(save_dir / f"sigmoid_{scen}_D{number_of_dimensions}_diff_{r}_vpost_probas.pdf")
                    #
                    # print('Posterior probabilities on the models : ', hawkes_model.vpost_proba)

                    # Save results
                    with open(save_dir / f'selected_model.json', 'w') as fp:
                        json.dump(model_1, fp, cls=NpEncoder)

                    with open(save_dir / f'elbo.json', 'w') as fp:
                        json.dump(hawkes_model.elbo, fp, cls=NpEncoder)

                    with open(save_dir / f'model_probas.json', 'w') as fp:
                        json.dump(hawkes_model.vpost_proba, fp, cls=NpEncoder)

                    with open(save_dir / f'est_norms.json', 'w') as fp:
                        json.dump(estimated_norms, fp, cls=NpEncoder)

                    with open(save_dir / f'true_norms.json', 'w') as fp:
                        json.dump(true_norms, fp, cls=NpEncoder)

                    # save estimated parameters in all models
                    with open(save_dir / f'estimates.json', 'w') as fp:
                        json.dump([hawkes_model.mean_nu_vposterior, hawkes_model.mean_weight_vposterior, hawkes_model.cov_vi], fp, cls=NpEncoder)

                    # estimated parameters only in MAP model
                    with open(save_dir / f'MAP_estimates.json', 'w') as fp:
                        json.dump([mean_nu_1, mean_weight_1, cov_1], fp, cls=NpEncoder)


                    model_config = {
                        "n_dim": number_of_dimensions,
                        "n_bas": number_of_bins,
                        "basis_true":basis_true,
                        "graph":graph,
                        "basis_estimation": basis_estimation,
                        "link": link,
                         "theta":true_link.theta,
                        "lamda_ub": true_link.lamda,
                        "estimate_lamda": estimate_lamda,
                        "eta": true_link.eta,
                        "beta": true_link.beta,
                         "base_activation": base_activation,
                         "weights": sparse_weight,
                        "T": T,
                        "N_eve": np.sum(N),
                        "A": A,
                        "N_exc": N_exc,
                        "sigma_prior": sigma,
                        "num_gq": num_gq,
                        "num_iter": num_iter,
                        "depth": depth,
                        "time_vi": t_vi,
                    }

                    with open(save_dir / f'config.json', 'w') as fp:
                        json.dump(model_config, fp, indent=2, cls=NpEncoder)






