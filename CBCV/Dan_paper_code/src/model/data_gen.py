import numpy as np
import pickle
import sys
import datetime

sys.path.insert(0, "../")
sys.path.insert(0, "../src/")
from utils.util import matrix_K, load_config

####################################
# Parameters of data generation
####################################
# Proportional Hazard Model Parameters
# Mean lambda_0 in Eq 3
root_dir = '../'


mu_lam_ac1, mu_lam_rac, mu_lam_cc, mu_lam_rcc = 0.1, 0.2, 0.2, 0.2 # lambda for IA, RA, IC, and RC

# Beta in Eq 2
b_ac1, b_cc, b_rac, b_rcc = 0, 0, 0, 0 # beta for IA, RA, IC, and RC. Here we assume no covariates and therefore beta=0

# C shape of distribution in Eq 1
c_ac1, c_cc, c_rac, c_rcc = 0.8, 0.6, 0.6, 0.6  # c for IA, RA, IC, and RC

# Covariance elements in Eq3
sigsq_lambda_ac1 = 1
sig_ac1_rac = 0
sigsq_lambda_rac = 1
sig_ac1_cc = 0
sig_rac_cc = 0
sigsq_lambda_cc = 1
sig_ac1_rcc = 0
sig_rac_rcc = 0
sig_cc_rcc = 0
sigsq_lambda_rcc = 1

# Selection Probability Parameters in Eq 4
sel_b_ac1, sel_b_rac, sel_b_cc, sel_b_rcc = 0.5, 0.5, -0.5, -0.5
sel_b_0 = -1.066582  # we optimize this later to get us as close as to the percentage of panel vs non-panel

# No, of in individuals (total size of prospect pool)
N_i = 5000
# Probability of initial acquisition \pi^{IA}
pi_1 = 0.9

# Probability of re-acquisition  \pi^{RA}
pi_2 = 0.9

# No. of months and corresponding no. of quarters
M = 60
q = int(np.floor(M / 3.0))

###################################

def calc_sel_prob(ind_params, sel_par):
    '''
    This function calculates the selection probabilities according to Eq. (4) in paper
    :param ind_params: an array including individual level parameters i.e. mu_lam_ac1, mu_lam_ac2, mu_lam_rac, mu_lam_cc, mu_lam_rcc
    :param sel_par: selection paramteres \beta_0 and \beta^z for (IA1, IA2, IC, RA, RC)
    :return: probability of panel selection
    '''
    tmp = sel_par * np.append(ind_params, np.ones([np.shape(ind_params)[0], 1]), axis=1)
    logit = np.sum(tmp, axis=1)
    return 1.0 / (1.0 + np.exp(-logit))

def panel_selection_split(ind_params, sel_par):
    '''
    :param ind_params: an array including individual level parameters i.e. mu_lam_ac1, mu_lam_ac2, mu_lam_rac, mu_lam_cc, mu_lam_rcc
    :param sel_par: sel_par: selection paramteres \beta_0 and \beta^z
    :return: tuple of parameters for panel and non-panel members
    '''
    sel_prob = calc_sel_prob(ind_params, sel_par)
    selections = np.random.binomial(1, sel_prob)
    panel_params = ind_params[selections == 1]
    non_panel_params = ind_params[selections == 0]
    return panel_params, non_panel_params


def B(m, c, beta, x_t, hist=False):
    '''
    functions that compute matrix B defined in equation 2
    :param m: month (int)
    :param c: c > 0 is a shape factor (float)
    :param beta: regression coefficients, should be of the same size as x_i
    :param x_t: x_1:m, [x_1, ... x_m] where x_i is covariate at time i
    :param hist: if True it returns the computed B for all months from 1 to m if False returns B for month m
    :return: B (float if hist=False and list of floats if hist=True)
    '''
    # initialize the summation
    s = 0.0
    # initialize the output list
    out = []

    # term1 is t^c and term2 is (t-1)^c I'll save the parameter to avoid re-calculation
    term2 = 0.0
    for t in range(m):
        term1 = np.power(t + 1, c)
        if beta:
            s += (term1 - term2) * np.exp(np.array(beta).dot(x_t[t]))
        else:
            s += (term1 - term2)
        out.append(s)
        term2 = np.copy(term1)
    if hist:
        return out
    return s


def inverse_B(B_list, val):
    '''
    This function exhaustively computes the inverse of B for a value and returns m such that m = B(val)
    :param B_list: B calculated for all months i.e. 1:M
    :param val: input
    :return: m such that m = B(val)
    '''
    if val > B_list[-1]:
        return len(B_list) + 1

    # find the index
    res = list(map(lambda i: i >= val, B_list)).index(True)

    # check if the value is closer to index or index + 1
    if res > 0 and np.abs(val - B_list[res]) > np.abs(val - B_list[res - 1]):
        res += - 1

    return res + 1


def weibull_draw(B_list, lam):
    '''
    This function uses inverse CDF method to sample from Weibull distribution
    :param B_list: B computed for all m:1...M (used for B inverse)
    :param lam: scale parameter of weibull distribution
    :return: samples (int)
    '''
    u = np.random.uniform()
    val = -np.log((1 - u))/lam
    m = inverse_B(B_list, val)
    return m


def generate_data(ind_params, pi_ia, pi_ra, c_params, b_params, x_t, M_tot):
    '''
    function simulates data generation algo outlined in Algo 1 of Web Appendix
    :param ind_params: individual level parameters; each row has mu_x for x in (IA,IC,RA,RC) for one individual
    :param pi_ia: pi_ia defined in paper, probability that a prospect will ever be acquired
    :param pi_ra: pi_ra defined in paper; probability that a churned customer will ever be acquired
    :param c_params: c for x in (IA, IC,RA, RC) for all individuals
    :param b_params: beta for x in (IA, IC,RA, RC) for all individuals
    :param x_t: covariates
    :param M_tot: all months under investigation
    :return: list of list of tuples; each list of tuples represents one individual, for example
    [(a, b), (c, d)] means that corresponding individual initially gets accquired after a month and then after b
    months it  gets churned and then after c months it gets re-acquired and then re-churned after d months
    '''
    out = []
    B_ac1 = B(M_tot, c_params['ac1'], b_params['ac1'], x_t, hist=True)
    B_cc = B(M_tot, c_params['cc'], b_params['cc'], x_t, hist=True)
    B_rac = B(M_tot, c_params['rac'], b_params['rac'], x_t, hist=True)
    B_rcc = B(M_tot, c_params['rcc'], b_params['rcc'], x_t, hist=True)

    for param in ind_params:
        ind_sim_data = []
        if np.random.binomial(1,pi_ia) == 0:
            out.append(ind_sim_data)
            continue

        t_ia = weibull_draw(B_ac1, param[0])
        t_ic = weibull_draw(B_cc, param[1])
        ind_sim_data.append((t_ia, t_ic))

        r_i = np.random.geometric(pi_ra, 1)
        for r in range(r_i[0]):
            t_ra = weibull_draw(B_rac, param[2])
            t_rc = weibull_draw(B_rcc, param[3])
            ind_sim_data.append((t_ra, t_rc))

        out.append(ind_sim_data)
    return out

def convert_behavior_to_y(actions, M):
    '''
    This function converst a list of actions generated by generate_Data function into y format defined  in
    Eq 5, page 10 of the paper
    :param actions: list of list of tuples, list
    :param M: total no. of months, int
    :return: list of np.arrays
    '''
    out = []
    for action_tuple in actions:
        x = np.zeros(4 * M)
        rep = 1
        actul_month_acc, actul_month_cc = 0, 0
        for acc_time, cc_time in action_tuple:
            actul_month_acc = actul_month_cc + acc_time
            actul_month_cc = actul_month_acc + cc_time
            if actul_month_acc <= M:
                if rep == 1:
                    x[actul_month_acc - 1] = 1

                else:
                    x[2*M + actul_month_acc - 1] = 1

            if actul_month_cc <= M:
                if rep == 1:
                    x[M + actul_month_cc - 1] = 1
                else:
                    x[3*M + actul_month_cc - 1] = 1
            rep += 1
        out.append(x)
    return out


def run(y_p_file, y_np_file, agg_q_file, agg_m_file, end_q_file):
    # get the log of mean of \lambda_0 for all processes
    mu_vec = np.log(np.array([mu_lam_ac1, mu_lam_rac, mu_lam_cc, mu_lam_rcc]))

    # create covariance matrix of \sigma_\lambda
    cov_vec = [sigsq_lambda_ac1, sig_ac1_rac,  sigsq_lambda_rac, sig_ac1_cc]
    cov_vec += [ sig_rac_cc, sigsq_lambda_cc, sig_ac1_rcc, sig_rac_rcc, sig_cc_rcc]
    cov_vec += [sigsq_lambda_rcc]
    cov_mat = np.zeros((4, 4))
    u_inds = np.triu_indices(len(cov_mat))
    l_inds = np.tril_indices(len(cov_mat))
    cov_mat[u_inds] = cov_vec
    cov_mat[l_inds] = cov_vec

    # vector of covariates (here as mentioned in the paper we assume no covariates)
    X_ctime = np.zeros(M)

    # draw lambdas_i for all individuals and all processes, see Eq 3
    lambda_i = np.exp(np.random.multivariate_normal(mu_vec, cov_mat, N_i))

    # draw panel member and non_panel members using Eq 4, value of \beta_0 could be changed to get different ratio
    # of panelvs nn-panel members

    sel_param = np.array([sel_b_ac1, sel_b_rac, sel_b_cc, sel_b_rcc, sel_b_0])
    lambda_p, lambda_np = panel_selection_split(lambda_i, sel_param)

    # simulate weibull distribution to for panel and non-panel members
    c_params = {"ac1": c_ac1, "cc": c_cc, "rac": c_rac, "rcc": c_rcc}
    b_params = {"ac1": b_ac1, "cc": b_cc, "rac": b_rac, "rcc": b_rcc}
    panel_actions = generate_data(lambda_p, pi_1, pi_2, c_params, b_params, X_ctime, M)
    non_panel_actions = generate_data(lambda_np, pi_1, pi_2, c_params, b_params, X_ctime, M)

    # transform actions into y format
    seq_panel = convert_behavior_to_y(panel_actions, M)
    seq_non_panel = convert_behavior_to_y(non_panel_actions, M)
    y = seq_non_panel + seq_panel

    # change y to aggregate format using matrix K, implementation of Eq 8
    k_mat = matrix_K(q, M)
    agg_data = k_mat.dot(sum(y)) # vector D_N at quarterly level [ADD_1, LOSS_1, END_1,...., ADD_q, LOSS_q, END_q]

    k_mat_all_months = matrix_K(M, M, duration=1, ADD=False, LOSS=False)
    agg_monthly = k_mat_all_months.dot(sum(seq_panel)) # aggregate data monthly level [ADD_1, LOSS_1, END_1,...., ADD_M, LOSS_M, END_M]

    k_mat_end = matrix_K(q, M, ADD=False, LOSS=False) # aggregate END data at quarterly level [END_1,..END_q]]
    end_quarterly = k_mat_end.dot(sum(y))


    # save data to file

    pickle.dump(seq_panel, open(y_p_file, 'wb'))
    pickle.dump(seq_non_panel, open(y_np_file, 'wb'))
    pickle.dump(agg_data.tolist()[0], open(agg_q_file, 'wb'))
    pickle.dump(end_quarterly.tolist()[0], open(end_q_file, 'wb'))
    pickle.dump(agg_monthly.tolist()[0], open(agg_m_file, 'wb'))



if __name__ == '__main__':
    config = load_config()
    stamp = datetime.date.today().strftime("%Y-%m-%d")
    y_panel_data = config['paths']['root_dir']+config['paths']['toy_data_path_agg']+f'y_panel_members_{stamp}.pkl'
    y_non_panel_data = config['paths']['root_dir']+config['paths']['toy_data_path_agg']+f'y_non_panel_members_{stamp}.pkl'
    agg_data_quarterly = config['paths']['root_dir']+config['paths']['toy_data_path_agg']+f'agg_data_{stamp}.pkl'
    agg_data_monthly = config['paths']['root_dir']+config['paths']['toy_data_path_agg']+f'agg_monthly_data_{stamp}.pkl'
    end_data_quarterly = config['paths']['root_dir']+config['paths']['toy_data_path_agg']+f'end_quarterly_data_{stamp}.pkl'
    run(y_panel_data, y_non_panel_data, agg_data_quarterly,
        agg_data_monthly, end_data_quarterly)
