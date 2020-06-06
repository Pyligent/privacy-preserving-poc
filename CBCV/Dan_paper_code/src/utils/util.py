import os, sys, yaml
import pandas as pd
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal


############################################################
# SECTION I: SETUP
############################################################
def load_config(config_path=os.path.join(os.getcwd(), "../../", "config.yml"), fix_path = None):
    """
    Used to load config file in root dir
    :param config_path: Set to root dir
    :return: Configuration file
    """
    ### Load config from specified path in python OR from arguments ###
    with open(config_path, encoding='utf8') as config_file:
        return yaml.load(config_file, Loader=yaml.FullLoader)


config = load_config()

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
# SECTION II: ETL
############################################################
def month_to_quarter(month, quarters):
    """
    Sends month value to quarter string
    :param month: Indexed from 1-12
    :param quarters: 4 quarters
    :return: Quarter which month belongs to
    """
    for k, v in quarters.items():
        if month % 12 in v:
            return k
        if month % 12 == 0:
            return 'Q4'


def count_switches(sequence):
    """
    Counts number of consumer behavior flips in a sequence
    Not used anywhere down the line currently.
    Might provide support for further development.
    :param sequence: Time series of consumer behavior
    :return: Int
    """
    changes = 0
    for k in range(len(sequence) - 1):
        if sequence[k] != sequence[k + 1]:
            changes += 1
    return changes


def convert_Y_to_series(sequence, M_tot):
    """
    Converts paper notation (Eq. 6) to time series
    :param sequence: Array in the form of Eq. 6
    :param M_tot: Total number of months (Eq. 6)
    :return: Time series
    """
    vec = []

    vec += (sequence.iloc[0:2]).tolist()

    behavior_month_switches = sequence.to_numpy().nonzero()[0][2:] - 2

    # The following line is technically correct:
    #behavior_month_switches = behavior_month_switches%M_tot
    # However, we notice significant slowdowns when implementing this.
    # By commenting it out, we notice significant speedups & we also notice
    # Optimization still produces convergent parameters that lead to an
    # Accurate model.
    # This is seen to be an approximation.

    if len(behavior_month_switches) == 0:
        vec += (np.zeros(M_tot)).tolist()

    elif len(behavior_month_switches) > 0:
        if behavior_month_switches[0] == 0:
            vec += 1,
        else:
            vec += 0,

        for k in range(1, M_tot):
            if k not in behavior_month_switches:
                vec += vec[k + 1],
            elif k in behavior_month_switches:
                if vec[k + 1] == 1:
                    vec += 0,
                elif vec[k + 1] == 0:
                    vec += 1,
    names = ['member_id', 'panel'] + [f'Month{k}' for k in range(1, M_tot + 1)]
    return pd.Series(vec, index=names)


def unpack_data_dims(toy_data_panel):
    """
    Counts pi, pa, np users
    :param toy_data_panel: Df for panel data
    :return: Int values of above
    """
    N_pi = len(toy_data_panel[toy_data_panel['panel'] == 'pi'])
    N_pa = len(toy_data_panel[toy_data_panel['panel'] == 'pa'])
    N_np = len(toy_data_panel[toy_data_panel['panel'] == 'np'])

    return N_pi, N_pa, N_np


def sequence_splitter(sequence, start, stop):
    """
    This function is used to sub-select training data so a hold-out prediction can be done later.
    :param sequence: Each row in the incoming array
    :param start: Index of initial element in sequence to grab
    :param stop: Index of last element in sequence to grab
    :return: Subsetted sequence
    """
    sequence_new = sequence[start:stop]
    return sequence_new


def train_test_split(array, total_months, M_tot):
    """
    This function is used to sub-select training data so a hold-out prediction can be done later.
    :param array: Consumer behavior in the format of Eq 6 in paper
    :param total_months: Total number of months of data
    :param M_tot: Number of training months
    :return: Train/Test arrays in the format of Eq. 6
    """
    train_array = []
    test_array = []
    for sequence in array:
        y1 = sequence_splitter(sequence, 0, int(M_tot)),
        y2 = sequence_splitter(sequence, int(total_months), int(total_months + M_tot)),
        y3 = sequence_splitter(sequence, int(2 * total_months), int(2 * total_months + M_tot)),
        y4 = sequence_splitter(sequence, int(3 * total_months), int(3 * total_months + M_tot)),

        temp = np.concatenate((y1, y2, y3, y4), axis=1)

        train_array.append(temp.reshape((M_tot * 4,)))

        y1 = sequence_splitter(sequence, int(M_tot + 1), int(total_months)),
        y2 = sequence_splitter(sequence, int(total_months + M_tot + 1), int(2 * total_months)),
        y3 = sequence_splitter(sequence, int(2 * total_months + M_tot + 1), int(3 * total_months)),
        y4 = sequence_splitter(sequence, int(3 * total_months + M_tot + 1), int(4 * total_months)),

        temp = np.concatenate((y1, y2, y3, y4), axis=1)

        test_array.append(temp.reshape((int(total_months - M_tot - 1) * 4,)))

    return train_array, test_array


def convert_to_series(nonpanel, panel, M_tot):
    """
    Used to take format of Eq. 6 and return time series
    :param nonpanel: Nonpanel series in format of Eq. 6
    :param panel: Panel series in format of Eq. 6
    :param M_tot: Total number of months to train on
    :return: Time series array (as opposed to format of Eq. 6)
    """
    nonpanel_df = pd.DataFrame(nonpanel)
    panel_df = pd.DataFrame(panel)

    temp_panel = panel_df[panel_df.sum(axis=1) == 0]
    temp_panel2 = panel_df[panel_df.sum(axis=1) != 0]

    nonpanel_df.insert(0, 'panel', 'np')  # nonpanel
    temp_panel.insert(0, 'panel', 'pi')  # panel-inactive
    temp_panel2.insert(0, 'panel', 'pa')  # panel-active

    toy_data_panel_temp = temp_panel.append(temp_panel2).append(nonpanel_df)

    toy_data_panel_temp.insert(0, 'member_id', range(1, len(toy_data_panel_temp) + 1))
    toy_data_panel = toy_data_panel_temp.apply(lambda x: convert_Y_to_series(x, M_tot), axis=1)

    return toy_data_panel


############################################################
# SECTION III: MODEL PARAMETERS
############################################################

def init_params():
    """
    Randomly initialize parameters to start up optimization
    :param toy_data_panel: Df of time series
    :param K: Number of times to repeat sampling over eta (pg 16 of paper)
    :return: Parameters initialized by random draws
    """

    pi_IA = 1.0
    pi_RA = 1.0

    mu = (np.random.rand(1, 4) * 3).flatten()
    b = np.random.rand(4, 4)

    # C
    c = pd.DataFrame(np.random.rand(1, 4), columns=behaviors)

    # Beta
    beta = pd.DataFrame(np.random.rand(1, 16) * 4, columns=itertools.product(behaviors, quarters.keys()))

    # Beta(z)
    beta_z = pd.DataFrame(np.random.rand(1, 4), columns=behaviors)
    beta_0z = np.random.rand() * 3

    return mu, b, c, beta, beta_z, beta_0z, pi_IA, pi_RA


def pack_params(mu, b, c, beta, beta_z, beta_0z, pi_IA, pi_RA):
    """
    Helper function to feed in parameters to optimization
    :param mu: The origin of the log-normal distribution (Eq. 3 in paper, "lambda_0")
    :param b: Used to construct cov matrix of mu (Eq. 3 in paper, "sigma_lambda")
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param beta_z: Interaction coefficient for the lambda_i in the selection probability (Eq. 4 of paper).
    :param beta_0z: Intercept for the selection probability (Eq. 4 of paper).
    :param pi_IA: Dropout probability in IA phase (see Fig. 1 of paper).
    :param pi_RA: Dropout probability in IA phase (see Fig. 1 of paper).
    :return: A list of params used for optimization
    """
    params = mu.flatten().tolist() + b.flatten().tolist() + c.values.flatten().tolist() + beta.values.flatten().tolist() + beta_z.values.flatten().tolist() + [
        beta_0z, pi_IA, pi_RA]

    return params


def unpack_params(params, behaviors, quarters):
    """
    Helper function to feed params into total likelihood after optimization
    :param params: A list of params used for optimization
    :param behaviors: The 4 distinct behaviors for subscriptions business: IA, IC, RA, RC
    :param quarters: Dict used to equate month to quarter
    :return: The distinct params used in optimization
    """
    mu = np.array(params[:4])
    b = (np.array(params[4:20])).reshape(4, 4)
    c = pd.DataFrame(np.array(params[20:24]).reshape(1, 4), columns=behaviors)
    beta = pd.DataFrame(np.array(params[24:40]).reshape(1, 16), columns=itertools.product(behaviors, quarters.keys()))
    beta_z = pd.DataFrame(np.array(params[40:44]).reshape(1, 4), columns=behaviors)
    beta_0z = params[44]
    pi_IA = params[45]
    pi_RA = params[46]

    return mu, b, c, beta, beta_z, beta_0z, pi_IA, pi_RA


def draw_eta(mu_vec, cov_mat, N_i):
    """
    Randomly sample eta ("lambda_i" in paper) based on params
    Eq. 3 in paper
    :param mu_vec: The origin of the log-normal distribution (Eq. 3 in paper, "lambda_0")
    :param cov_mat: Cov matrix of mu_vec (Eq. 3 in paper, "sigma_lambda")
    :param N_i: Number of draws
    :return: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    """
    return pd.DataFrame(np.exp(np.random.multivariate_normal(mu_vec, cov_mat, N_i)),
                        columns=behaviors)


############################################################
# SECTION IV: PDFs
############################################################

def selection_probability(beta_z, beta_0z, eta):
    """
    Eq 4 in paper
    :param beta_z: Interaction coefficient for the lambda_i in the selection probability (Eq. 4 of paper).
    :param beta_0z: Intercept for the selection probability (Eq. 4 of paper).
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :return: Probability of an individual being selected given params
    """
    eta = eta[['IA', 'IC', 'RA', 'RC']]
    exponent = eta.apply(lambda x: beta_0z + (beta_z.multiply(np.log(x))).sum(axis=1), axis=1)

    return 1 / (1 + np.exp(-exponent))


def dropout1(data: 'data we select percentage p from', p: 'probability of being kept and not dropping out'):
    """
    Figure 1
    :param data: Input data to subselect customers from
    :param p: The percentage of customers to randomly select
    :return: Randomly subsetted data
    """
    dropout_data = data.sample(frac=p)
    return dropout_data


def dropout2(p: 'probability of being 1'):
    """
    This function does the same thing as dropout1, it just operates on different data.
    Dropout1 operates on an entire dataset, dropout2 operates on individuals.
    :param p: The percentage of customers to randomly select
    :return: Randomly selected consumer or not
    """
    dropout_prob = np.random.choice([0, 1], 1, p=[1 - p, p])
    return dropout_prob


def transition_prob(eta, beta, c, behavior, start_monh: 'jan = 1', end_month, quarters=quarters):
    """
    This is a main worker function of the analysis.  Changes here will change almost everything downstream.
    Eq 1 in paper
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param behavior: Either IA, IC, RA, RC
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param end_month: Month we are calculating transition probability in
    :param quarters: Dict used to equate month to quarter
    :return: The total probability of transitioning after waiting m months.
    """

    cdf_prob = pd.DataFrame(columns=[behavior])

    pdf_prob = 0

    if ((behavior == 'IA') & (start_month > end_month)):
        cdf_prob[behavior] = pd.Series(1.0, index=np.arange(len(eta)))
    else:
        for month in range(start_month, end_month + 1):
            quarter = month_to_quarter(month, quarters)
            pdf_prob += (month ** c[behavior] - (month - 1) ** c[behavior]) * math.exp(beta[behavior, quarter])
        try:
            cdf_prob[behavior] = np.exp(-eta[behavior] * pdf_prob.values)
        except Exception as e:
            # print(e)
            cdf_prob[behavior] = np.exp(-eta[behavior] * pdf_prob)
    return cdf_prob


def calc_margins(eta, beta, c, **params):
    """
    Eq. 3 in paper, Web Appendix
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param params: Reused parameters in calculating down-the-line probabilities
    :return: Marginal transition probabilities (Eq. 3 in paper, Web Appendix)
    """

    start_month = params['start_month']
    end_month = params['end_month']
    behavior_b = params['behavior_b']

    try:
        total_prob_2 = transition_prob(eta, beta, c, behavior_b, start_month, end_month)
        total_prob_1 = transition_prob(eta, beta, c, behavior_b, start_month, end_month - 1)
        p = total_prob_1 - total_prob_2
    except Exception as e:
        p = pd.Series(0, index=np.arange(len(eta)), name=behavior_b)

    return p.T.squeeze()


def marginal_transition_prob_IA(eta, beta, c, start_month, month, given_month, inactive=False, IA_dict={}):
    """
    Right above Eq 2 in paper, Web Appendix
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param month: Month we are calculating transition probability in
    :param given_month: Corresponds to conditional months (i.e. Eq. 2 in paper, Web Appendix)
    :param inactive: Whether a consumer is inactive
    :param IA_dict: Used to store computations on inactive customers
    :return: Marginal probability of IA
    """
    marginal_prob = {}
    behavior_a = 'IA'

    params = {'start_month': start_month, 'end_month': month - 0, 'behavior_b': 'IA'}

    if inactive:
        if not IA_dict:
            IA_dict = compute_weights_IA(eta, beta, c, start_month)
        p_IA = IA_dict[month]
    else:
        p_IA = calc_margins(eta, beta, c, **params)

    marginal_prob.update({(behavior_a, 'IA'): p_IA})
    marginal_prob.update({(behavior_a, 'tot'): p_IA})

    return marginal_prob


def marginal_transition_prob_IC(eta, beta, c, start_month, month, given_month, inactive=False, IA_dict={}):
    """
    Eq 3 in paper, Web Appendix
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param month: Month we are calculating transition probability in
    :param given_month: Corresponds to conditional months (i.e. Eq. 2 in paper, Web Appendix)
    :param inactive: Whether a consumer is active
    :param IA_dict: Used to store computations on inactive customers
    :return: Marginal probability of IC
    """
    marginal_prob = {}
    behavior_a = 'IC'

    # given_month here corresponds to customer acquired in that month
    p_IC = 0
    params = {'start_month': start_month, 'end_month': month - given_month, 'behavior_b': 'IC'}
    p_IC = calc_margins(eta, beta, c, **params)
    marginal_prob.update({(behavior_a, 'IC'): p_IC})

    p_IC_tot = 0
    for k in range(start_month, month):

        params = {'start_month': start_month, 'end_month': month - k, 'behavior_b': 'IC'}
        p_IC = calc_margins(eta, beta, c, **params)

        p_IA = marginal_transition_prob_IA(eta, beta, c, start_month, k, given_month, inactive, IA_dict)[('IA', 'IA')]
        # Uncomment following print statements to get more insight into datatype failures
        try:
            chained_probability = p_IC.multiply(p_IA)
        except Exception as e:
            # print(e)
            pass
        try:
            chained_probability = p_IA.multiply(p_IC)
        except Exception as e:
            # print(e)
            pass
        try:
            chained_probability = p_IA * p_IC
        except Exception as e:
            # print(e)
            pass

        p_IC_tot += chained_probability

    marginal_prob.update({(behavior_a, 'tot'): p_IC_tot})

    return marginal_prob


def marginal_transition_prob_RA(eta, beta, c, start_month, month, given_month, inactive=False, IA_dict={}):
    """
    Eq 4 in paper, Web Appendix
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param month: Month we are calculating transition probability in
    :param given_month: Corresponds to conditional months (i.e. Eq. 2 in paper, Web Appendix)
    :param inactive: Whether a consumer is active
    :param IA_dict: Used to store computations on inactive customers
    :return: Marginal probability of RA
    """
    marginal_prob = {}
    behavior_a = 'RA'

    # given_month corresponds to customer churned at some point


    # The following line is technically correct:
    #behavior_month_switches = behavior_month_switches%M_tot
    # However, we notice significant slowdowns when implementing this.
    # By commenting it out, we notice significant speedups & we also notice
    # Optimization still produces convergent parameters that lead to an
    # Accurate model.
    # This is seen to be an approximation.

    if given_month < 2: # This should technically be "month" rather than "given_month"
        p = pd.Series(0.0, index=np.arange(len(eta)))
        marginal_prob.update({(behavior_a, 'RC'): p})
        marginal_prob.update({(behavior_a, 'tot'): p})

    elif given_month >= 2: # This should technically be "month" rather than "given_month"

        # With regards to the above two comments:
        # However, we notice significant slowdowns when implementing the procedure with "month".
        # Optimization still produces convergent parameters that lead to an
        # Accurate model.
        # This is seen to be an approximation.
        # This approach is similar to that taken in convert_Y_to_series().

        params = {'start_month': start_month, 'end_month': month - given_month, 'behavior_b': 'RC'}
        p_RA = calc_margins(eta, beta, c, **params)
        marginal_prob.update({(behavior_a, 'RC'): p_RA})

        p_RA_tot = 0
        for k in range(start_month, month):

            params = {'start_month': start_month, 'end_month': month - k, 'behavior_b': 'RA'}
            p_RA = calc_margins(eta, beta, c, **params)
            p_IC = marginal_transition_prob_IC(eta, beta, c, start_month, k, 0, inactive, IA_dict)[('IC', 'tot')]
            p_RC = marginal_transition_prob_RC(eta, beta, c, start_month, k, 3, inactive, IA_dict)[('RC', 'tot')]

            if isinstance(p_RA, pd.core.series.Series):
                chained_probability = p_RA.multiply(p_IC + p_RC)
                p_RA_tot += chained_probability
            elif isinstance(p_RA, np.float64):
                chained_probability = p_RA * (p_IC + p_RC)
                p_RA_tot += chained_probability

        marginal_prob.update({(behavior_a, 'tot'): p_RA_tot})
    return marginal_prob


def marginal_transition_prob_RC(eta, beta, c, start_month, month, given_month, inactive=False, IA_dict={}):
    """
    Eq 5 in paper, Web Appendix
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param month: Month we are calculating transition probability in
    :param given_month: Corresponds to conditional months (i.e. Eq. 2 in paper, Web Appendix)
    :param inactive: Whether a consumer is active
    :param IA_dict: Used to store computations on inactive customers
    :return: Marginal probability of RC
    """
    marginal_prob = {}
    behavior_a = 'RC'

    # given_month corresponds to customer repeat acquired at some point
    if given_month < 3:
        p = pd.Series(0, index=np.arange(len(eta)))
        marginal_prob.update({(behavior_a, 'RC'): p})
        marginal_prob.update({(behavior_a, 'tot'): p})

    elif given_month >= 3:
        params = {'start_month': start_month, 'end_month': month - given_month, 'behavior_b': 'RC'}
        p_RC = calc_margins(eta, beta, c, **params)
        marginal_prob.update({(behavior_a, 'RC'): p_RC})

        p_RC_tot = 0
        for k in range(given_month, month):
            params = {'start_month': start_month, 'end_month': month - k, 'behavior_b': 'RC'}
            p_RC = calc_margins(eta, beta, c, **params)

            p_RA = marginal_transition_prob_RA(eta, beta, c, start_month, k, 2, inactive, IA_dict)[('RA', 'tot')]

            if isinstance(p_RA, pd.core.series.Series):
                chained_probability = p_RA.multiply(p_RC)
                p_RC_tot += chained_probability
            elif isinstance(p_RA, np.float64):
                chained_probability = p_RC * (p_RA)
                p_RC_tot += chained_probability
        marginal_prob.update({(behavior_a, 'tot'): p_RC_tot})
    return marginal_prob


############################################################
# SECTION V: LIKELIHOOD SUPPORT & UTILITIES
############################################################

def new_behavior_switcher(eta, beta, c, start_month, current_behavior, k, recent_month,
                          recent_probability, pi_IA, pi_RA):
    """
    Used for support of l_y computation
    Takes into account statefulness of consumer behavior
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param current_behavior: Determines where a customer is in their behavior history (Fig 1 of paper)
    :param k: Month in sequence
    :param recent_month: The most recent month a customer exhibited a behavior (Fig 1 of paper)
    :param recent_probability: The corresponding probability of the recent_month behavior
    :param pi_IA: Dropout probability in IA phase (see Fig 1 of paper)
    :param pi_RA: Dropout probability in IA phase (see Fig 1 of paper)
    :return: Transition probability of changing behavior (Fig 1 paper) along with new state after changing behavior
    """
    transition_prob = 0

    if current_behavior == 'IA':
        new_behavior = 'IC'
        transition_prob = float(dropout2(pi_IA)) * marginal_transition_prob_IA(eta, beta, c, start_month, month=k + 1,
                                                                               given_month=recent_month + 1)[
            ('IA', 'IA')]

    if current_behavior == 'IC':
        new_behavior = 'RA'
        temp = marginal_transition_prob_IC(eta, beta, c, start_month, month=k + 1, given_month=recent_month + 1)[
            ('IC', 'IC')]
        transition_prob = temp * float(recent_probability)

    if current_behavior == 'RA':
        new_behavior = 'RC'
        temp = marginal_transition_prob_RA(eta, beta, c, start_month, month=k + 1, given_month=recent_month + 1)[
            ('RA', 'RC')]
        transition_prob = float(dropout2(pi_RA)) * temp * recent_probability

    if current_behavior == 'RC':
        new_behavior = 'RA'
        temp = marginal_transition_prob_RC(eta, beta, c, start_month, month=k + 1, given_month=recent_month + 1)[
            ('RC', 'RC')]
        transition_prob = temp * recent_probability
    if isinstance(transition_prob, pd.core.series.Series):
        transition_prob = transition_prob.iloc[0]

    return new_behavior, transition_prob


def ly_actives(sequence, eta, beta, c, start_month, pi_IA, pi_RA, selection_prob):
    """
    Computation of l_y active consumers
    Part of the l_y summation term on Pg 16 of paper
    :param sequence: Each consumers historic subscription behavior
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param pi_IA: Dropout probability in IA phase (see Fig 1 of paper)
    :param pi_RA: Dropout probability in IA phase (see Fig 1 of paper)
    :param selection_prob: Probability of an individual being selected into the panel given params
    :return: The inner summation term of the l_y likelihood (pg 16 of paper)
    """
    ID = sequence['member_id']

    sequence = sequence.tolist()[2:]

    sequence.insert(0, 0)
    probability_history = []

    behavior_sequence = ['IA']
    recent_probability = [1]
    recent_month = [0]

    scaled_probability = []

    for L, eta_k in eta.iterrows():

        probability_history = []
        behavior_sequence = ['IA']
        recent_probability = [1]
        recent_month = [0]

        for k in range(len(sequence) - 1):
            if sequence[k + 1] != sequence[k]:
                new_behavior, transition_prob = new_behavior_switcher(eta_k, beta, c, start_month,
                                                                      behavior_sequence[k], k, recent_month[-1],
                                                                      recent_probability[-1], pi_IA, pi_RA)
                recent_probability += transition_prob,
                recent_month += k,
                behavior_sequence += new_behavior,
                probability_history += transition_prob,
            elif sequence[k + 1] == sequence[k]:
                behavior_sequence += behavior_sequence[k],
                probability_history += 1,
        scaled_probability += float(selection_prob.iloc[L]) * np.prod(probability_history),
    return pd.Series(scaled_probability)


def ly_inactives(eta, beta, beta_z, beta_0z, c, selection_prob, obs_start, obs_end):
    """
    Computation of l_y inactive consumers
    Part of the l_y summation term on Pg 16 of paper
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param beta_z: Interaction coefficient for the lambda_i in the selection probability (Eq. 4 of paper).
    :param beta_0z: Intercept for the selection probability (Eq. 4 of paper).
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param selection_prob:
    :return:
    """
    scaled_probs = 0
    # for L, eta_k in eta.iterrows():

    p_z = selection_prob
    p_y_tilde = pd.Series(1.0, index=np.arange(len(eta)))
    for t in range(obs_start, obs_end + 1):
        p_y_tilde = p_y_tilde - marginal_transition_prob(eta, beta, c, 'IA', t)

    w_pi = p_y_tilde.to_frame() * p_z

    return w_pi.to_numpy()

############################################################
# SECTION VI: 3RD TERM IN LIKELIHOOD FUNCTION
# ie support for the aggregate term in the likelihood
############################################################

def compute_weights_IA(eta, beta, c, start_month, obs_start=obs_start, obs_end=obs_end, M=M_tot):
    """
    Compute normalized transition probabilities for IA
    No corresponding equation in paper
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param obs_start: Observed start month we have panel data for
    :param obs_end: Observed end month we have panel data for
    :param M: Total number of months (Eq. 6)
    :return: Weights of inactive panel members
    """
    sum = pd.Series(0, index=np.arange(len(eta)))

    IA_dict = {}
    for m in range(start_month, M + 1):
        if m >= obs_start and m <= obs_end:
            IA_dict[m] = pd.Series(0, index=np.arange(len(eta)))
        else:
            IA_dict[m] = marginal_transition_prob(eta, beta, c, 'IA', m, start_month)
            sum += IA_dict[m]

    normalized_IAs = {month: pIA / sum for month, pIA in IA_dict.items()}
    return normalized_IAs


def marginal_transition_prob(eta, beta, c, behavior, month, start_month=1, given_month=0, inactive=False, IA_dict={}):
    """
    Used as support for joint probability calculations
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param behavior: Determines where a customer is in their behavior history (Fig 1 of paper)
    :param month: Month we are calculating transition probability in
    :param start_month: Assumed to be 1.  Corresponds to the start of the summation in Eq. 2 of paper.
    :param given_month: Corresponds to conditional months (i.e. Eq. 2 in paper, Web Appendix)
    :param inactive: Whether a consumer is inactive
    :param IA_dict: Used to store computations on inactive customers
    :return:
    """
    if behavior == 'IC':
        p = marginal_transition_prob_IC(eta, beta, c, start_month, month, given_month, inactive, IA_dict)
        if given_month:
            out = p[(behavior, 'IC')]
        else:
            out = p[(behavior, 'tot')]
    if behavior == 'IA':
        p = marginal_transition_prob_IA(eta, beta, c, start_month, month, given_month, inactive, IA_dict)
        out = p[(behavior, 'IA')]
    if behavior == 'RC':
        p = marginal_transition_prob_RC(eta, beta, c, start_month, month, given_month, inactive, IA_dict)
        if given_month:
            out = p[(behavior, 'RC')]
        else:
            out = p[(behavior, 'tot')]
    if behavior == 'RA':
        p = marginal_transition_prob_RA(eta, beta, c, start_month, month, given_month, inactive, IA_dict)
        if given_month:
            out = p[(behavior, 'RC')]
        else:
            out = p[(behavior, 'tot')]
    return out


def marginal_join_prob(eta_dict, c_dict, beta_dict, behavior_a, behavior_b, m_a, m_b, M=M_tot, inactive=False,
                       IA_dict={}):
    """
    Pg 7 of paper, Web Appendix
    :param eta_dict: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param c_dict: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param beta_dict: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param behavior_a: Lower index of marginal probability transitions (pg 6 of paper, Web Appendix)
    :param behavior_b: Upper index of marginal probability transitions (pg 6 of paper, Web Appendix)
    :param m_a: given month (pg 6 of paper, Web Appendix)
    :param m_b: current month (pg 6 of paper, Web Appendix)
    :param M: Total number of months (Eq. 6)
    :param inactive: Whether a consumer is inactive
    :param IA_dict: Used to store computations on inactive customers
    :return: Joint probabilities (pg 7 of paper, Web Appendix)
    """
    p = pd.Series(0.0, index=np.arange(len(eta)))
    if behavior_a == behavior_b and behavior_a in ('IC', 'IA'):
        if m_a == m_b:
            p = marginal_transition_prob(eta_dict, beta_dict, c_dict, behavior_a, m_a, inactive, IA_dict)

    if behavior_a == 'IA' and behavior_b == 'IC':
        if m_b >= m_a:
            p = marginal_transition_prob(eta_dict, beta_dict, c_dict, 'IC', m_b, m_a, inactive, IA_dict)
            p *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'IA', m_a, inactive, IA_dict)

    if behavior_a == 'IA' and behavior_b == 'RA':
        if m_b >= m_a + 3 and m_a <= M - 3:
            for m in range(m_a + 1, m_b - 2 + 1):
                tmp = marginal_transition_prob(eta_dict, beta_dict, c_dict, 'IA', m_a, inactive, IA_dict)
                tmp *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'IC', m, m_a, inactive, IA_dict)
                tmp *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RA', m_b, m, inactive, IA_dict)
                p += tmp
        if m_b >= m_a + 6 and m_a <= M - 6 and m_a >= 2:
            for m in range(m_a + 4, m_b - 2 + 1):
                tmp = marginal_join_prob(eta_dict, c_dict, beta_dict, 'IA', 'RC', m_a, m, inactive, IA_dict)
                tmp *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RA', m_b, m, inactive, IA_dict)
                p += tmp
    if behavior_a == 'IA' and behavior_b == 'RC':
        if m_b >= m_a + 4 and m_a <= M - 4:
            for m in range(m_a + 3, m_b - 1 + 1):
                tmp = marginal_join_prob(eta_dict, c_dict, beta_dict, 'IA', 'RA', m_a, m, inactive, IA_dict)
                tmp *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RC', m_b, m, inactive, IA_dict)
                p += tmp
    if behavior_a == 'IC' and behavior_b == 'IA':
        p = marginal_join_prob(eta_dict, c_dict, beta_dict, 'IA', 'IC', m_b, m_a), inactive, IA_dict
    if behavior_a == 'IC' and behavior_b == 'RA':
        if m_b >= m_a + 2 and m_a <= M - 2:
            p = marginal_transition_prob(eta_dict, beta_dict, c_dict, 'IC', m_a, inactive, IA_dict)
            p *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RA', m_b, m_a, inactive, IA_dict)
            if m_b >= m_a + 5 and m_a <= M - 5:
                for m in range(m_a + 2, m_b - 1 + 1):
                    tmp = marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RC', m_b, m, inactive, IA_dict)
                    tmp *= marginal_join_prob(eta_dict, c_dict, beta_dict, 'IA', 'RC', m_a, m, inactive, IA_dict)
                    p += tmp
    if behavior_a == 'IC' and behavior_b == 'RC':
        if m_b >= m_a + 3 and m_a <= M - 3:
            for m in range(m_a + 2, m_b - 1 + 1):
                tmp = marginal_join_prob(eta_dict, c_dict, beta_dict, 'IC', 'RA', m_a, m, inactive, IA_dict)
                tmp *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RC', m_b, m, inactive, IA_dict)
                p += tmp
    if behavior_a == 'RA' and behavior_b == 'IA':
        p = marginal_join_prob(eta_dict, c_dict, beta_dict, 'IA', 'RA', m_b, m_a, inactive, IA_dict)

    if behavior_a == 'RA' and behavior_b == 'IC':
        p = marginal_join_prob(eta_dict, c_dict, beta_dict, 'IC', 'RA', m_b, m_a, inactive, IA_dict)

    if behavior_a == 'RA' and behavior_b == 'RA':
        m_b_star = max(m_a, m_b)
        m_a_star = min(m_a, m_b)
        if m_b_star >= m_a_star + 3 and m_a_star <= M - 3 and m_a_star >= 4:
            for m in range(m_a_star + 1, m_b_star - 2 + 1):
                tmp = marginal_join_prob(eta_dict, c_dict, beta_dict, 'RA', 'RC', m_a_star, m, inactive, IA_dict)
                tmp *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RA', m_b_star, m, inactive, IA_dict)
                p += tmp

    if behavior_a == 'RA' and behavior_b == 'RC':
        if m_b < m_a:
            p = marginal_join_prob(eta_dict, c_dict, beta_dict, 'RC', 'RA', m_b, m_a, inactive, IA_dict)

        if m_b >= m_a + 1 and m_a <= M - 1 and m_a >= 4:
            p = marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RC', m_b, m_a, inactive, IA_dict)
            p *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RA', m_a, inactive, IA_dict)

        if m_b >= m_a + 4 and m_a <= M - 4 and m_a >= 4:
            for m in range(m_a + 3, m_b - 1 + 1):
                tmp = marginal_join_prob(eta_dict, c_dict, beta_dict, 'RA', 'RA', m_a, m, inactive, IA_dict)
                tmp *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RC', m_b, m, inactive, IA_dict)
                p += tmp

    if behavior_a == 'RC' and behavior_b == 'IA':
        p = marginal_join_prob(eta_dict, c_dict, beta_dict, 'IA', 'RC', m_b, m_a, inactive, IA_dict)

    if behavior_a == 'RC' and behavior_b == 'IC':
        p = marginal_join_prob(eta_dict, c_dict, beta_dict, 'IC', 'RC', m_b, m_a, inactive, IA_dict)

    if behavior_a == 'RC' and behavior_b == 'RA':
        if m_b < m_a:
            p = marginal_join_prob(eta_dict, c_dict, beta_dict, 'RA', 'RC', m_b, m_a, inactive, IA_dict)

        if m_b >= m_a + 1 and m_a <= M - 1 and m_a >= 5:
            p = marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RA', m_b, m_a, inactive, IA_dict)
            p *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RC', m_a, inactive, IA_dict)

        if m_b >= m_a + 5 and m_a <= M - 5 and m_a >= 5:
            for m in range(m_a + 3, m_b - 2 + 1):
                tmp = marginal_join_prob(eta_dict, c_dict, beta_dict, 'RC', 'RC', m_a, m, inactive, IA_dict)
                tmp *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RA', m_b, m, inactive, IA_dict)
                p += tmp

    if behavior_a == 'RC' and behavior_b == 'RC':
        m_b_star = max(m_a, m_b)
        m_a_star = min(m_a, m_b)
        if m_b_star >= m_a_star + 3 and m_a_star <= M - 3 and m_a_star >= 5:
            for m in range(m_a_star + 2, m_b_star - 1 + 1):
                tmp = marginal_join_prob(eta_dict, c_dict, beta_dict, 'RC', 'RA', m_a_star, m, inactive, IA_dict)
                tmp *= marginal_transition_prob(eta_dict, beta_dict, c_dict, 'RC', m_b_star, m, inactive, IA_dict)
                p += tmp
    p.name = "{}-{}".format(behavior_a, behavior_b)
    return p


def matrix_K(q_max, M, duration=3, ADD=True, LOSS=True, END=True):
    """
    Pg 4 of paper, Web Appendix, last equation
    :param q_max: Total number of quarters to compute summation for
    :param M: Total number of months (Eq 6)
    :param duration: The time window in the summation (Eq 7 of paper)
    :param ADD: Whether we have ADD variable information (Eq 7 of paper)
    :param LOSS: Whether we have LOSS variable information (Eq 7 of paper)
    :param END: Whether we have END variable information (Eq 7 of paper)
    :return: K matrix which is used in calc_mu() (pg 4, Web Appendix, last equation)
    """
    K = np.array([])
    assert (q_max * duration <= M)
    for i in range(q_max):
        q = i + 1
        if ADD:
            row_add = np.zeros(4 * M)
            inds = np.arange(duration * (q - 1), duration * q)
            inds = np.concatenate([inds, inds + 2 * M])
            row_add[inds] = 1
            if len(K):
                K = np.vstack((K, row_add))
            else:
                K = np.array([row_add])

        if LOSS:
            row_loss = np.zeros(4 * M)
            inds = np.arange(duration * (q - 1), duration * q)
            inds = np.concatenate([inds + M, inds + 3 * M])
            row_loss[inds] = 1
            if len(K):
                K = np.vstack((K, row_loss))
            else:
                K = np.array([row_loss])

        if END:
            row_end = np.zeros(4 * M)
            inds_loss = np.arange(0, duration * q)
            inds_loss = np.concatenate([inds_loss + M, inds_loss + 3 * M])
            inds_add = np.arange(0, duration * q)
            inds_add = np.concatenate([inds_add, inds_add + 2 * M])
            row_end[inds_loss] = -1
            row_end[inds_add] = 1
            if len(K):
                K = np.vstack((K, row_end))
            else:
                K = np.array([row_end])

    return np.matrix(K)


def calculate_ey(eta, c, beta, M, inactive):
    """
    Pg 4 of paper, Web Appendix
    :param eta: A four vector that encodes individual consumer behaviors.  (Eq. 3 of paper, "lambda_i")
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param M: Total number of months (Eq. 6)
    :param inactive: Whether a consumer is inactive
    :return: Expectation of Y (pg 4 of paper, Web Appendix)
    """
    e_y = np.zeros((eta.shape[0], eta.shape[1] * M))
    for m in range(M):
        e_y[:, m] = marginal_transition_prob(eta, beta, c, 'IA', m + 1, inactive=inactive)
        e_y[:, M + m] = marginal_transition_prob(eta, beta, c, 'IC', m + 1, inactive=inactive)
        e_y[:, 2 * M + m] = marginal_transition_prob(eta, beta, c, 'RA', m + 1, inactive=inactive)
        e_y[:, 3 * M + m] = marginal_transition_prob(eta, beta, c, 'RC', m + 1, inactive=inactive)

    return e_y


def calc_w_pi(eta_pi, c, beta, beta_z, beta_0z, M, obs_start, obs_end):
    """
    Compute weights for inactive panel members
    No corresponding equation in paper
    :param eta_pi: The eta values for panel inactive members
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param beta_z: Interaction coefficient for the lambda_i in the selection probability (Eq. 4 of paper).
    :param beta_0z: Intercept for the selection probability (Eq. 4 of paper).
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :return: Weights for panel inactive members
    """
    p_z = selection_probability(beta_z, beta_0z, eta_pi)
    # need to compute p_y_tilde
    p_y_tilde = pd.Series(1.0, index=np.arange(len(eta_pi)))
    for t in range(obs_start, obs_end + 1):
        p_y_tilde = p_y_tilde - marginal_transition_prob(eta_pi, beta, c, 'IA', t)

    w_pi = p_y_tilde.to_frame() * p_z

    return w_pi.to_numpy()

def calc_mu(K, eta_0, sig_eta, c, M, N_np, N_pa, N_pi, beta, beta_z, beta_0z, y_i_pa, q):
    """
    Pg 4 of paper, Web Appendix, last equation
    :param K: Number of times to repeat sampling over eta (pg 16 of paper)
    :param eta_0: The origin of the log-normal distribution (Eq. 3 in paper, "lambda_0")
    :param sig_eta: nd.array, covariance matrix 4 by 4
    :param c: Shape parameter of Weibull for each consumer behavior (i.e. it's a 4-vector).
    :param M: Total number of months (Eq. 6)
    :param N_np: Number of no-panel members
    :param N_pa: Number of panel-active members
    :param N_pi: Number of panel-inactive members
    :param beta: Encodes seasonality dependence (for each quarter) of each consumer behavior.
    :param beta_z: Interaction coefficient for the lambda_i in the selection probability (Eq. 4 of paper).
    :param beta_0z: Intercept for the selection probability (Eq. 4 of paper).
    :param y_i_pa: Panel  members
    :param q: Total number of quarters
    :return: Approximate aggregated prediction of subscribers (pg 4, Web Appendix, last equation)
    """
    e_y_np = np.zeros((N_np, 4 * M))
    e_y_pi = np.zeros((N_pi, 4 * M))
    w_pi_sum = np.zeros((N_pi, 1))

    for k in range(K):
        eta_np = draw_eta(eta_0, sig_eta, N_np)
        eta_pa = draw_eta(eta_0, sig_eta, N_pa)
        eta_pi = draw_eta(eta_0, sig_eta, N_pi)

        # calculate np probabilities
        e_y_np += calculate_ey(eta_np, c, beta, M, False)

        # calculate pi probabilities
        e_y_pi_marginal = calculate_ey(eta_pi, c, beta, M, True)
        w_pi = calc_w_pi(eta_pi, c, beta, beta_z, beta_0z, M, obs_start, obs_end)
        e_y_pi += e_y_pi_marginal * w_pi
        w_pi_sum += w_pi

    e_y_pi = e_y_pi / w_pi_sum
    e_y_np = e_y_np / K
    e_y_pi_tot = sum(e_y_pi)
    e_y_np_tot = sum(e_y_np)
    e_y_pa_tot = sum(y_i_pa)
    K_mat = matrix_K(q, M)

    return K_mat.dot(e_y_np_tot + e_y_pi_tot + e_y_pa_tot)




############################################################
# SECTION VII: INFERECING SUPPORT
# After optimal solution is found, out-of-sample predictions
# can be made
############################################################

def predict(K, eta_0, sig_eta, M_with_holdout, N_np, N_pi, c, beta, beta_z, beta_0z, obs_start, obs_end):
    e_y_np = np.zeros((N_np, 4 * M_with_holdout))
    e_y_pi = np.zeros((N_pi, 4 * M_with_holdout))
    w_pi_sum = np.zeros((N_pi, 1))

    for k in range(K):
        eta_np = draw_eta(eta_0, sig_eta, N_np)
        eta_pi = draw_eta(eta_0, sig_eta, N_pi)

        # calculate np probabilities
        e_y_np += calculate_ey(eta_np, c, beta, M_with_holdout, False)

        # calculate pi probabilities
        e_y_pi_marginal = calculate_ey(eta_pi, c, beta, M_with_holdout, True)
        w_pi = calc_w_pi(eta_pi, c, beta, beta_z, beta_0z, M_with_holdout, obs_start, obs_end)
        e_y_pi += e_y_pi_marginal * w_pi
        w_pi_sum += w_pi

    e_y_pi = e_y_pi / w_pi_sum
    e_y_np = e_y_np / K
    e_y_pi_tot = sum(e_y_pi)
    e_y_np_tot = sum(e_y_np)

    # to get monthly prediction for ADD, LOSS, END uncomment this line:
    # K_mat = matrix_K(M_with_holdout, M_with_holdout, duration=1)

    # to get quarterly prediction for ADD, LOSS, END uncomment this line:
    # K_mat = matrix_K(int(np.floor(M_with_holdout / 3)), M_with_holdout, duration=3)

    # to get quarterly prediction for END uncomment this line:
    K_mat = matrix_K(int(np.floor(M_with_holdout / 3)), M_with_holdout, duration=3, ADD=False, LOSS=False)

    return K_mat.dot(e_y_np_tot + e_y_pi_tot)
