import os, sys
import pandas as pd
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.optimize import minimize, basinhopping, dual_annealing, differential_evolution
from numba import jit, njit
import pickle
from random import random
import datetime
import argparse

sys.path.insert(0, "../")
from utils import util

############################################################
# SECTION I: SETUP
############################################################

config = util.load_config()

toy_data_path = config['paths']['root_dir'] + config['paths']['toy_data_path_agg']
solutions_path = config['paths']['root_dir'] + config['paths']['solutions_path']

quarters = config['setup']['quarters']
behaviors = config['setup']['behaviors']

start_month = config['parameters']['start_month']
K = config['parameters']['K']
M_tot = config['parameters']['M_tot']
obs_start = config['parameters']['obs_start']
obs_end = config['parameters']['obs_end']
q = config['parameters']['q']

############################################################
# SECTION II: DATA
############################################################
########################################
##### Load in toy data
########################################

# The format of this data is assumed to follow the notation in the paper (Eq. 6)
nonpanel = pickle.load(open(toy_data_path + "y_non_panel_members_apr_23.pkl", "rb"))
panel = pickle.load(open(toy_data_path + "y_panel_members_apr_23.pkl", "rb"))


########################################
##### Split data into train data
########################################


total_months = len(panel[0]) / 4
nonpanel_train, nonpanel_test = util.train_test_split(nonpanel, total_months, M_tot)
panel_train, panel_test = util.train_test_split(panel, total_months, M_tot)


########################################
##### ETL data for integrating into likelihood
########################################

y_pa = panel_train  # this is just to be consistent w/ notation in paper
Y = panel_train + nonpanel_train
y_sum = sum(Y)
d = (util.matrix_K(q, M_tot, duration=3, ADD=True, LOSS=True, END=True)).dot(y_sum)

## The following line is to convert the arrays (corresponding to paper's notation) to a time series
## i.e. the conversion will take a sequence of length 4*M_tot to a sequence of length M_tot
## The results is a d pandas df that represents y_pa in time series format

toy_data_panel = util.convert_to_series(nonpanel_train, panel_train, M_tot)


############################################################
# SECTION III: LIKELIHOOD
############################################################


def l_z(beta_z, beta_0z, eta, N_pi, N_pa, N_np):
    """
    This represents the selection likelihood of consumers given a set of eta
    :param beta_z: Interaction coefficient for the lambda_i in the selection probability (Eq. 4 of paper).
    :param beta_0z: Intercept for the selection probability (Eq. 4 of paper).
    :param eta: Equivalent to "lambda" in the paper.
    :param toy_data_panel: A sequence of 1's & 0's at the consumer level, generated under Weibull+PH assumption.
    :return: A scalar representing the selection likelihood.
    """
    selection_prob = util.selection_probability(beta_z, beta_0z, eta)

    panel = (N_pi + N_pa) * np.log(np.mean(selection_prob))
    nonpanel = N_np * np.log(1 - np.mean(selection_prob))
    likelihood = panel + nonpanel

    return likelihood


def l_y(eta, beta, c, quarters, toy_data_panel, beta_z, beta_0z, start_month, N_pi, pi_IA, pi_RA, obs_start, obs_end):
    """
    This represents the likelihood of seeing the observed consumer behavior given a set of eta
    :param eta: Equivalent to "lambda" in the paper.
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param quarters: Setup configuration that equates months to quarter.
    :param toy_data_panel: A sequence of 1's & 0's at the consumer level, generated under Weibull+PH assumption.
    :param beta_z: Interaction coefficient for the lambda_i in the selection probability (Eq. 4 of paper).
    :param beta_0z: Intercept for the selection probability (Eq. 4 of paper).
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param pi_IA: Dropout probability in IA phase (see Fig. 1 of paper).
    :param pi_RA: Dropout probability in IA phase (see Fig. 1 of paper).
    :return: A scalar representing the observed panel likelihood.
    """

    selection_prob = util.selection_probability(beta_z, beta_0z, eta)
    a = 1 / np.sum(selection_prob)
    # Panel_actives

    toy_data_panel_active = toy_data_panel[toy_data_panel['panel'] == 'pa']

    scaled_probs = toy_data_panel_active.apply(
        lambda x: util.ly_actives(x, eta, beta, c, start_month, pi_IA, pi_RA, selection_prob),
        axis=1)

    summed_probs = scaled_probs.sum(axis=1)
    summed_probs = summed_probs[summed_probs != 0]
    likelihood_actives = np.log((float(a) * summed_probs)).sum()

    # Panel-inactives
    scaled_probs = util.ly_inactives(eta, beta, beta_z, beta_0z, c, selection_prob, obs_start, obs_end)

    summed_probs = scaled_probs.sum(axis=1)
    summed_probs = summed_probs[summed_probs != 0]

    likelihood_inactives = N_pi*np.log((float(a) * summed_probs)).sum()

    return likelihood_actives + likelihood_inactives



def l_d(mu, cov, c, M_tot, N_np, N_pa, N_pi, beta, beta_z, beta_0z, y_pa, q, d, K, sig_theta):
    '''
    This represents the likelihood of describing the aggregated data with given parameters
    "param mu: list of length 4, [mu_IA, mu_IC, mu_RA, mu_RC]
    :param sig_eta: nd.array, covariance matrix 4 by 4
    :param M_tot: int, total months.
    :param N_np: int, number of non-panel members
    :param N_pa: int, number of panel-active members
    :param N_pi: int, number of panel-inactive members
    :param beta_z: DataFrame, beta_z in selection pdf
    :param beta_0z: float, beta_0 in selection pdf
    :param y_pa: nd.array, N_pa rows, 4*M columns each row represents
                y_i = [IA_i1, IAi2, ...IA_iM, IC_i1,...IC_iM, RA_i1,...,RA_iM, RC_i1,...,RC_1M]
                for panel member i
    :param q: int, end quarter of calibration period; should satisfy q<= M / 3
    :param d: nd.array, one row,  3 * q columns: [ADD_1, LOSS_1, END_1,..., ADD_q, LOSS_q, END_q]
    :param K: Number of times to repeat sampling over eta (pg 16 of paper)
    :param sig_theta: covariance of theta defaulted to identity matrix first
    :return: A scalar representing the approximate aggregate likelihood
    '''
    mu = util.calc_mu(K, mu, cov, c, M_tot, N_np, N_pa, N_pi, beta, beta_z, beta_0z, y_pa, q)
    tmp = np.matrix(d) - mu

    return -0.5 * float(tmp * (sig_theta) * tmp.T)


def total_likelihood(params, M_tot, quarters, behaviors, toy_data_panel, d, y_pa, q, start_month, K):
    """
    This function just packs up the 3-likelihood terms
    :param params: All of the parameters to be searched over in the optimization phase
    :param M_tot: int, total months.
    :param quarters: Dict used to equate month to quarter
    :param behaviors: The 4 distinct behaviors for subscriptions business: IA, IC, RA, RC
    :param toy_data_panel: A pandas df that represents y_pa in time series format
    :param d: A vector of length 3*q that represents quarterly information (Eq. 7 of paper)
    :param y_pa: nd.array, N_pa rows, 4*M columns each row represents
                y_i = [IA_i1, IAi2, ...IA_iM, IC_i1,...IC_iM, RA_i1,...,RA_iM, RC_i1,...,RC_1M]
                for panel member i
    :param q: int, end quarter of calibration period; should satisfy q<= M / 3
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param K: Number of times to repeat sampling over eta (pg 16 of paper)
    :return: A scalar representing the approximate total approximate likelihood
    """
    mu, b, c, beta, beta_z, beta_0z, pi_IA, pi_RA = util.unpack_params(params, behaviors, quarters)
    N_pi, N_pa, N_np = util.unpack_data_dims(toy_data_panel)

    cov = (b + b.T) / 2
    eta = util.draw_eta(mu, cov, K)

    sig_theta = np.identity(d.shape[1])

    # For now, we set pi_IA =1 & pi_RA = 1
    # These terms represent secondary corrections to the l_y term

    lz = l_z(beta_z, beta_0z, eta, N_pi, N_pa, N_np)
    ly = l_y(eta, beta, c, quarters, toy_data_panel, beta_z, beta_0z, start_month, N_pi, pi_IA, pi_RA, obs_start, obs_end)
    ld = l_d(mu, cov, c, M_tot, N_np, N_pa, N_pi, beta, beta_z, beta_0z, y_pa, q, d, K, sig_theta)
    likelihood = -(lz + ly + ld)

    print(likelihood)

    return np.array(likelihood)


############################################################
# SECTION IV: OPTIMIZATION
############################################################
def optimization(M_tot, quarters, behaviors, toy_data_panel, d, y_pa, q, start_month, K, method):
    """
    :param M_tot: int, total months.
    :param quarters: Dict used to equate month to quarter
    :param behaviors: The 4 distinct behaviors for subscriptions business: IA, IC, RA, RC
    :param toy_data_panel: A pandas df that represents y_pa in time series format
    :param d: A vector of length 3*q that represents quarterly information (Eq. 7 of paper)
    :param y_pa: nd.array, N_pa rows, 4*M columns each row represents
                y_i = [IA_i1, IAi2, ...IA_iM, IC_i1,...IC_iM, RA_i1,...,RA_iM, RC_i1,...,RC_1M]
                for panel member i
    :param q: int, end quarter of calibration period; should satisfy q<= M / 3
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param K: Number of times to repeat sampling over eta (pg 16 of paper)
    :param method: Optimization method.  Chosen to be 'anneal' or 'evolve'
    :return: An OptimizeResult object that is saved to pickle in the solutions/ directory
    """
    mu, b, c, beta, beta_z, beta_0z, pi_IA, pi_RA = util.init_params()

    params = np.array(util.pack_params(mu, b, c, beta, beta_z, beta_0z, pi_IA, pi_RA))

    b1 = (0, 5)
    b2 = (-5, 5)
    b3 = (0.99, 1)
    bounds1 = [b1 for x in range(4)]
    bounds2 = [b2 for x in range(16)]
    bounds3 = [b2 for x in range(4)]
    bounds = bounds1 + bounds2 + bounds1 + bounds2 + bounds3 + [b2] + [b3] + [b3]

    args = (M_tot, quarters, behaviors, toy_data_panel, d, y_pa, q, start_month, K)

    if method == 'evolution':
        solution = differential_evolution(total_likelihood,
                                          bounds=bounds,
                                          args=args,
                                          polish=True,
                                          popsize=2,
                                          maxiter=2,
                                          workers=-1)

    elif method == 'anneal':
        minimizer_kwargs = {"method": "Powell"}
        solution = dual_annealing(total_likelihood,
                                  bounds=bounds,
                                  args=args,
                                  local_search_options=minimizer_kwargs,
                                  maxiter=2,
                                  maxfun=2)

    return solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimization_method', type=str, help='anneal or evolve', default='anneal')
    parser.add_argument('--run', type=str, help='1, 2, etc.. or any string that will be appended to outfile',
                        default='1')
    args = parser.parse_args()
    method = args.optimization_method
    run = args.run
    stamp = datetime.date.today().strftime("%Y-%m-%d")

    solution = optimization(M_tot, quarters, behaviors, toy_data_panel, d, y_pa, q, start_month, K,
                            method=method)

    pickle.dump(solution, open(solutions_path + f"optimal_params_{stamp}_{method}_run{run}.pickle", "wb"))
