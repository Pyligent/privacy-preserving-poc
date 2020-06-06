library(tidyverse)
library(zoo)
library(scales)
library(knitr)
library(gridExtra)
library(grid)

#propietary package contains all the code for computing objective function
install.packages('MPL_0.1.0.tar.gz', repos = NULL, type = 'source')
library(MPL)





# User can specify whether to only run 2 parameter settings for which N=20K (=TRUE), or all scenarios (=FALSE). Default to TRUE
do_abridged_code = TRUE




########################################
## DATA SETTINGS - we set a baseline scenario and then marginally perturb the baseline, one dimension at a time

# length of calibration period (in months) that we consider
M_companys = c(36,60,84,108)
n_Ms = length(M_companys)
baseline_M_setting = 2			# this is the index within 'M_companys' corresponding to the baseline scenario for M
n_Ms_nonbaseline = n_Ms - 1

# sizes of population that we consider in simulation
Ns = c(20*10^3, 100*10^3, .5*10^6, 2.5*10^6)		
n_Ns = length(Ns)
baseline_N_setting = 2			# this is the index within 'Ns' corresponding to the baseline scenario for N
n_Ns_nonbaseline = n_Ns - 1

# length of holdout period is 18 months (6 quarters)
M_holdout = 1.5*12										
Q_holdout = M_holdout/3

# extent of selection bias in the panel data set that we consider
# these correspond to the coefficients associated with (\beta^Z for IA, IC, RA, RC)
sel_bias = rbind("none"=c(0,0,0,0),
				"middle"=c(1,1,-1,-1)/2,
				"high"=c(1,1,-1,-1))
sel_bias_levels = rownames(sel_bias)				
colnames(sel_bias) = c("sel_b_ac1","sel_b_rac","sel_b_cc","sel_b_rcc")				
n_sel_bias_settings = nrow(sel_bias)
baseline_sel_bias_setting = 2			# this is the index within 'sel_bias' corresponding to the baseline scenario
n_sel_bias_settings_nonbaseline = n_sel_bias_settings - 1

# panel size as a percent of the total population observed in the panel that we consider
panel_pcts = c(.01, .05, .10)					# we convert this into sel_b_0 via an optimization step below
n_panel_settings = length(panel_pcts)
baseline_panel_pct_setting = 2
n_panel_settings_nonbaseline = n_panel_settings - 1




########################################
## PARAMETER SETTINGS - these settings vary according to a full-factorial design

# (lambda_0,c) settings for IA, IC, RA, RC (all can vary independently)

# for initial acquisitions, the first scenario allows for a sharply increasing pattern which we only see with initial acquisitions
	# we do not see this pattern for the other processes
IA_params = rbind(c(.001,1.2),c(.1,.8))		
colnames(IA_params) = c("lam","c")
n_IA_settings = nrow(IA_params)

IC_params = rbind(c(.1,.8),c(.2,.6))
colnames(IC_params) = c("lam","c")
n_IC_settings = nrow(IC_params)

RA_params = rbind(c(.1,.8),c(.2,.6))
colnames(RA_params) = c("lam","c")
n_RA_settings = nrow(RA_params)

RC_params = rbind(c(.1,.8),c(.2,.6))
colnames(RC_params) = c("lam","c")
n_RC_settings = nrow(RC_params)


# \Sigma_0: square root of diagonals of covariance matrix for IA and IC versus for RA and RC
sd_IA_ICs = c(1,2)
n_sd_mean_IA_IC_settings = length(sd_IA_ICs)

sd_RA_RCs = c(1,2)
n_sd_RA_RC_settings = length(sd_RA_RCs)


# \Sigma_0: off-diagonal correlations within process 
# (i.e., for IA/RA and IC/RA) versus across process (i.e., IA/IC, IA/RC, ...) settings
covmat_correl_within_process = c(0,.2)		# IA/RA and IC/RC
n_covmat_correl_within_process_settings = length(covmat_correl_within_process)

covmat_correl_across_process = c(0,.2)		# everything else
n_covmat_correl_across_process_settings = length(covmat_correl_across_process)


##################
# fixed parameters

pi_1 = .9			# Percentage of the population that will eventually be acquired
pi_NA = .1			# after initially churning, 1 minus the probability that they will eventually be reacquired

b_ac1 = 0			# This allows users to specify a covariate if they so desire, but in the simulation, covariate effects are shut off
b_ac2 = 0			
b_rac = 0
b_cc = 0			
b_rcc = 0			

#parameters including "_ac2" are irrelevant:
  #code allows for a 2-segment mixture in the initial acquisition process,
  #but we do not use it in our process because it is not empirically identified.
  #instead we fix the 2nd segment's parameters and set the size of the segment to 0 in our optimizations
pi_2 = 0			
mu_lam_ac2 = 1
c_ac2 = 1
sig_ac1_ac2 = 0
sigsq_lambda_ac2 = 1		# The fact that the size of the segment is equal to 0 means all other parameters associated with ac2 are ignored
sig_ac2_cc = 0
sig_ac2_rac = 0
sig_ac2_rcc = 0
sel_b_ac2 = 0


##################
# fixed data settings
left_trunc_time = 0 		# assume no left truncation
m_star_ctime = 0			# assume credit card panel begins at the beginning of commercial operations



########################################
## GATHER ALL SCENARIOS FOR BOTH DATA AND PARAMETERS

# full factorial design for the parameter settings
parameter_settings_mat = expand.grid(
						IA_lam_c=1:n_IA_settings,
						RA_lam_c=1:n_RA_settings,
						IC_lam_c=1:n_IC_settings,
						RC_lam_c=1:n_RC_settings,
						sd_mean_adj_IA_IC=1:n_sd_mean_IA_IC_settings,
						sd_mean_adj_RA_RC=1:n_sd_RA_RC_settings,
						covmat_corr_w_in=1:n_covmat_correl_within_process_settings,
						covmat_corr_across=1:n_covmat_correl_across_process_settings
						)
nr_parameter_settings_mat = nrow(parameter_settings_mat)						

# marginal design for data settings	- baseline, then perturb baseline one element at a time along each process				
baseline_data_settings = c(M=baseline_M_setting,
					  N=baseline_N_setting,
				      sel_bias=baseline_sel_bias_setting,
					  panel_pct=baseline_panel_pct_setting)
n_data_settings = length(baseline_data_settings)
n_nonbase_settings_total = n_Ms_nonbaseline + n_Ns_nonbaseline + n_sel_bias_settings_nonbaseline + n_panel_settings_nonbaseline
data_settings_mat = many_rows(baseline_data_settings,n_nonbase_settings_total+1)

data_settings_mat[1+1:n_Ms_nonbaseline,1] = (1:n_Ms)[-baseline_M_setting]
data_settings_mat[1+n_Ms_nonbaseline + 1:n_Ns_nonbaseline,2] = (1:n_Ns)[-baseline_N_setting]
data_settings_mat[1+n_Ms_nonbaseline+n_Ns_nonbaseline +1:n_sel_bias_settings_nonbaseline,3] = (1:n_sel_bias_settings)[-baseline_sel_bias_setting]
data_settings_mat[1+n_Ms_nonbaseline+n_Ns_nonbaseline+n_sel_bias_settings_nonbaseline + 1:n_panel_settings_nonbaseline,4] = (1:n_panel_settings)[-baseline_panel_pct_setting]
nr_data_settings_mat = nrow(data_settings_mat)

# populate the full matrix of all possibilities, full factorial for parameters, marginal for data settings
all_settings_mat = cbind(data_settings_mat[rep(1:nr_data_settings_mat,nr_parameter_settings_mat),],
						parameter_settings_mat[rep(1:nr_parameter_settings_mat,each=nr_data_settings_mat),])
nsims = nrow(all_settings_mat)

# create variant of 'all_settings_mat' where entries represent the actual data setting values
all_settings_vals = all_settings_mat %>%
				mutate(M = M_companys[M]) %>%
				mutate(N = Ns[N]) %>%
				mutate(sel_bias = c("none","middle","high")[sel_bias]) %>%
				mutate(panel_pct = panel_pcts[panel_pct])



######################### SCENARIOS START HERE

# Three methods are MPL, PAN, and AGG, noting that MPL and AGG use two-stage estimation procedures
methods = c("proposed_2_stage","panel","agg_2_stage")
n_methods = length(methods)		

#Get the Halton sequence draws to be used for numerical integration
sims_draws = halton(100, 5, normal=TRUE)
	
#parameter names
#parameters including "_ac2" are irrelevant:
  #code allows for a 2-segment mixture in the initial acquisition process,
  #but we do not use it in our process because it is not empirically identified.
  #instead we fix the 2nd segment's parameters and set the size of the segment to 0 in our optimizations
parnames = c("mu_lam_ac1", "c_ac1", "pi_1",  "b_ac1",  "mu_lam_ac2",
				 "c_ac2", "pi_2", "b_ac2", "m_star", "mu_lam_rac",
				 "c_rac", "pi_NA", "b_rac", "mu_lam_cc", "c_cc",
				 "b_cc", "mu_lam_rcc", "c_rcc", "b_rcc", "sigsq_lambda_ac1",
				 "sig_ac1_ac2", "sigsq_lambda_ac2", "sig_ac1_rac", "sig_ac2_rac", "sigsq_lambda_rac",
				 "sig_ac1_cc", "sig_ac2_cc", "sig_rac_cc", "sigsq_lambda_cc", "sig_ac1_rcc",
				 "sig_ac2_rcc", "sig_rac_rcc", "sig_cc_rcc", "sigsq_lambda_rcc", "sel_b_0",
				 "sel_b_ac1", "sel_b_ac2", "sel_b_rac", "sel_b_cc", "sel_b_rcc")
npar = length(parnames)
parnames_wo_ac1ac2 = parnames[-which(parnames %in% c("sig_ac1_ac2")) ]



# create folder to store all outputs
dir.create("simulation outputs")
dir.create("simulation outputs/monthly_agg_data")
dir.create("simulation outputs/param_actuals")
dir.create("simulation outputs/param_ests")

# set what scenarios to run: either N=20K scenarios, or all scenarios

# this takes all N=20K data settings, and subsets to the first two parameter settings. Remove '[1:2]' to estimate all methods for all N=20K scenarios
if(do_abridged_code==TRUE){ runs_to_do = which(all_settings_vals$N == Ns[1])[1:2] }		

# this runs through all data settings and all parameter settings
if(do_abridged_code==FALSE){ runs_to_do = 1:nsims }


for(i in runs_to_do){

	# pick out our scenario
	settings_i = all_settings_mat[i,]

	# setting seed for reproducibility
	set.seed(1234)

	# convert each row of this matrix back to the variables they represent
	M_i = M_companys[as.numeric(settings_i["M"])]											# unpack data settings
	Q_i = M_i/3
	N_i = Ns[as.numeric(settings_i["N"])]
	M_tot = M_i + M_holdout
	Q_tot = M_tot/3
	sel_bias_i = sel_bias[as.numeric(settings_i["sel_bias"]),]
	panel_pct_i = panel_pcts[as.numeric(settings_i["panel_pct"])]

	IA_lam_c_i = IA_params[as.numeric(settings_i["IA_lam_c"]),]							# unpack parameter settings
	RA_lam_c_i = RA_params[as.numeric(settings_i["RA_lam_c"]),]
	IC_lam_c_i = IC_params[as.numeric(settings_i["IC_lam_c"]),]
	RC_lam_c_i = RC_params[as.numeric(settings_i["RC_lam_c"]),]
	sd_IA_IC_i = sd_IA_ICs[as.numeric(settings_i["sd_mean_adj_IA_IC"])]
	sd_RA_RC_i = sd_RA_RCs[as.numeric(settings_i["sd_mean_adj_RA_RC"])]
	covmat_corr_w_in_i = covmat_correl_within_process[as.numeric(settings_i["covmat_corr_w_in"])]
	covmat_corr_across_i = covmat_correl_across_process[as.numeric(settings_i["covmat_corr_across"])]


	# convert variables to underlying parameters if they aren't already
	mu_lam_ac1 = IA_lam_c_i["lam"]				
	c_ac1 = IA_lam_c_i["c"]				
	mu_lam_rac = RA_lam_c_i["lam"]				
	c_rac = RA_lam_c_i["c"]										
	mu_lam_cc = IC_lam_c_i["lam"]				
	c_cc = IC_lam_c_i["c"]				
	mu_lam_rcc = RC_lam_c_i["lam"]				
	c_rcc = RC_lam_c_i["c"]				

	sigsq_lambda_ac1 = sd_IA_IC_i^2 			
	sigsq_lambda_rac = sd_RA_RC_i^2 
	sigsq_lambda_cc = sd_IA_IC_i^2 
	sigsq_lambda_rcc = sd_RA_RC_i^2

	sig_ac1_rac = covmat_corr_w_in_i * sqrt(sigsq_lambda_ac1) * sqrt(sigsq_lambda_rac)
	sig_cc_rcc = covmat_corr_w_in_i * sqrt(sigsq_lambda_cc) * sqrt(sigsq_lambda_rcc)
	sig_ac1_cc = covmat_corr_across_i * sqrt(sigsq_lambda_ac1) * sqrt(sigsq_lambda_cc)
	sig_rac_cc = covmat_corr_across_i * sqrt(sigsq_lambda_rac) * sqrt(sigsq_lambda_cc)
	sig_ac1_rcc = covmat_corr_across_i * sqrt(sigsq_lambda_ac1) * sqrt(sigsq_lambda_rcc)
	sig_rac_rcc = covmat_corr_across_i * sqrt(sigsq_lambda_rac) * sqrt(sigsq_lambda_rcc)

	sel_b_ac1 = sel_bias_i["sel_b_ac1"]			# extract selection parameters (except for sel_b_0 which we get below)
	sel_b_rac = sel_bias_i["sel_b_rac"]
	sel_b_cc = sel_bias_i["sel_b_cc"]
	sel_b_rcc = sel_bias_i["sel_b_rcc"]

	# there is one prospect pool of size N
	prospect_pools = cbind(
	  birth_month = seq(0, M_i-1),
	  pool_size = c(N_i,rep(0,M_i-1)))
	prospect_pools = prospect_pools[prospect_pools[,2] > 0,,drop=FALSE]

	# (hardcoded) covariate matrix: quarterly seasonality
	X_ctime_w_holdout = cbind(Q1 = rep(0,M_tot))
	p = ncol(X_ctime_w_holdout)
	X_ctime = X_ctime_w_holdout[1:M_i,,drop=FALSE]

	# splice in all parameters into a true parameter vector (except for sel_b_0)
	true_par = rep(NA,npar)
	names(true_par) = parnames
	true_par["mu_lam_ac1"] = mu_lam_ac1
	true_par["c_ac1"] = c_ac1
	true_par["pi_1"] = pi_1
	true_par["b_ac1"] = b_ac1
	true_par["mu_lam_ac2"] = mu_lam_ac2
	true_par["c_ac2"] = c_ac2
	true_par["pi_2"] = pi_2
	true_par["b_ac2"] = b_ac2
	true_par["m_star"] = m_star_ctime
	true_par["mu_lam_rac"] = mu_lam_rac
	true_par["c_rac"] = c_rac
	true_par["pi_NA"] = pi_NA
	true_par["b_rac"] = b_rac
	true_par["mu_lam_cc"] = mu_lam_cc
	true_par["c_cc"] = c_cc
	true_par["b_cc"] = b_cc
	true_par["mu_lam_rcc"] = mu_lam_rcc
	true_par["c_rcc"] = c_rcc
	true_par["b_rcc"] = b_rcc
	true_par["sigsq_lambda_ac1"] = sigsq_lambda_ac1
	true_par["sig_ac1_ac2"] = sig_ac1_ac2
	true_par["sigsq_lambda_ac2"] = sigsq_lambda_ac2
	true_par["sig_ac1_rac"] = sig_ac1_rac
	true_par["sig_ac2_rac"] = sig_ac2_rac
	true_par["sigsq_lambda_rac"] = sigsq_lambda_rac
	true_par["sig_ac1_cc"] = sig_ac1_cc
	true_par["sig_ac2_cc"] = sig_ac2_cc
	true_par["sig_rac_cc"] = sig_rac_cc
	true_par["sigsq_lambda_cc"] = sigsq_lambda_cc
	true_par["sig_ac1_rcc"] = sig_ac1_rcc
	true_par["sig_ac2_rcc"] = sig_ac2_rcc
	true_par["sig_rac_rcc"] = sig_rac_rcc
	true_par["sig_cc_rcc"] = sig_cc_rcc
	true_par["sigsq_lambda_rcc"] = sigsq_lambda_rcc
	true_par["sel_b_ac1"] = sel_b_ac1
	true_par["sel_b_ac2"] = sel_b_ac2
	true_par["sel_b_rac"] = sel_b_rac
	true_par["sel_b_cc"] = sel_b_cc
	true_par["sel_b_rcc"] = sel_b_rcc


	# generate true parameter names by process
	acq_parnames = c("mu_lam_ac1","c_ac1", "pi_1", "b_ac1", "mu_lam_ac2", "c_ac2", "pi_2", "b_ac2", 	
					"m_star", "mu_lam_rac", "c_rac", "pi_NA", "b_rac") 
	ret_parnames = c("mu_lam_cc", "c_cc", "b_cc", "mu_lam_rcc", "c_rcc", "b_rcc")		
	sel_parnames = c('sel_b_0', 'sel_b_ac1', 'sel_b_ac2', 'sel_b_rac', 'sel_b_cc', 'sel_b_rcc') 

	# get parameters in forms convenient to simulate data from
	model_par = true_par[which(parnames %in% c(acq_parnames,ret_parnames))]
	cov_mat_vec = true_par[grep("sig",parnames)]		
	n_ind_par = length(grep("sigsq",parnames))
	cov_mat = matrix(NA, nrow=n_ind_par, ncol=n_ind_par)			# initialize conversion covariance parameter vector into a matrix Sigma_0
	cov_mat[upper.tri(cov_mat, TRUE)] = cov_mat_vec					# add upper triangular elements to covariance matrix
	for(j in 2:5){													# port upper triangular elements to lower triangular elements
		for(k in 1:(j-1))
		  cov_mat[j,k] = cov_mat[k,j]
		}

	# generate simulated parameters for true population
	presimulated = FALSE
	ind_par_mat = get_sim_params(model_par, cov_mat, presimulated = presimulated, nsim = N_i)		# simulate individual-level parameters
	ind_par_mat[is.infinite(ind_par_mat)] = max(ind_par_mat[!is.infinite(ind_par_mat)])
	ind_par_mat[,5] = .1					# just putting in arbitrary non-zero values for lambda ac2 b/c it gets zeroed out anyways

	# extract the lambdas, which are an input in the selection process
	all_lambdas = ind_par_mat[,c(1,5,10,14,17)]			
	sel_par = true_par[grep("sel_",parnames)]
	sel_par_no_b_0 = sel_par[!(names(sel_par) %in% "sel_b_0")]
	offsets = rowSums(many_rows(sel_par_no_b_0,N_i)*log(all_lambdas))
	
	# we obtain the value for sel_b_0 (the intercept term in the selection process)
	# which gets us as close as possible (by mean squared error) to the target size 
	# of the panel as a % of the population given the parameters of the model
	MSE_panel_pct = function(sel_b0){
		x = sel_b0 + offsets
		probs = exp(x)/(1+exp(x))
		(panel_pct_i - mean(probs))^2
		}
	lower_upper_sel_b_0_search = c(-20,5)
	sel_b_0_optim_results = optimize(f=MSE_panel_pct, 
									lower = lower_upper_sel_b_0_search[1],
									upper = lower_upper_sel_b_0_search[2])
	sel_b_0 = sel_b_0_optim_results$`minimum`
	true_par["sel_b_0"] = sel_b_0
	sel_par["sel_b_0"] = sel_b_0
	
	# save true parameters
	saveRDS(true_par, file=paste0("simulation outputs/param_actuals/true_par_scenario_num_",i,".rds"))		

	# run simulator to get one simulated dataset
	# split population into panel and non-panel
	selection_sim = panel_selection_sim(ind_par_mat, X_ctime_w_holdout, sel_par)

	# get number of people in the panel versus outside the panel
	non_panel_pop = nrow(selection_sim$agg_par_mat)
	panel_pop = N_i - non_panel_pop
	
	# get the prospect birth times for panel and non-panel members individually
	prospect_month_ctimes = rep(prospect_pools[,"birth_month"],times=prospect_pools[,"pool_size"])
	panel_cohs = rep(0,panel_pop)			# panel members are always born at ctime 0
	non_panel_cohs = prospect_month_ctimes[-which(prospect_month_ctimes==0)[1:panel_pop]]

	# simulate data for non-panel members
	agg_addloss_sim_list_w_holdout = agg_acqchurn_sim(ind_par_mat = selection_sim$agg_par_mat, X_ctime = X_ctime_w_holdout, 
										cohs = non_panel_cohs,init_only = FALSE)
	agg_addloss_sim_list_cal = lapply(agg_addloss_sim_list_w_holdout, function(x) x[1:M_i])							

	# simulate data for panel members
	panel_sim_list = panel_acqchurn_sim(selection_sim$panel_par_mat, X_ctime_w_holdout, cohs = panel_cohs, left_trunc_time)

	# Get all granular active panel member data (all panel members who aren't active are accounted for below)
	# Each list element is a panel member. The entries of this element are the acquisition (1st column) and churn (2nd column) times
	# for that panel member
	acqchurn_ptime_list_w_holdout = panel_sim_list$acqchurn_ptime_list

	# create function to take granular panel member data over calibration period and holdout period
	# and remove all the holdout period data, so that we only see the granular panel data during 
	# the calibration period
	trim_acqchurn_list_to_cal = function(list_element,M_cal){
		# input: 
		# list_element: a matrix list element of acqchurn_ptime_list_w_holdout
		# M_cal: length of calibration period in months (company time)
		# output: list_element trimmed down to just the calibration period
		
		if(list_element[1,1]>M_cal){list_element=NULL}
		if(!is.null(list_element)){
			if(list_element[1,1]<=M_cal){
				reacq_in_holdout_inds = which(list_element[,1]>M_cal)
				if(length(reacq_in_holdout_inds)>0){list_element = list_element[-reacq_in_holdout_inds,,drop=FALSE]}
				
				survived_to_cal_inds = which(list_element[,2]>M_cal)
				list_element[survived_to_cal_inds,2]=M_cal+1
				}
			}
			
		list_element
		}

	# Use function above to get granular panel member data for active customers for just the calibration period
	acqchurn_ptime_list_cal = lapply(acqchurn_ptime_list_w_holdout,trim_acqchurn_list_to_cal,M_cal=M_i)
	acqchurn_ptime_list_cal = acqchurn_ptime_list_cal[unlist(lapply(acqchurn_ptime_list_cal,function(x) is.null(x)==FALSE))]		# remove any panel members that are totally NULL
	n_NULLs = length(acqchurn_ptime_list_w_holdout) - length(acqchurn_ptime_list_cal)		# number of people who were first acquired after end of calibration period

	# get the NULL count and add it back to the number of panel inactive people to get the total number of inactive population members during the cal period
	panel_inactive_pop_cal = panel_sim_list[[6]]+n_NULLs

	# add back initial and repeat acquisitions and churns to the ex-panel aggregated holdout data to get the aggregated holdout data across all customers (in panel or not in panel)
	panel_IAs_w_holdout = data.table(unlist(lapply(acqchurn_ptime_list_w_holdout, function(x) x[1,1])))[,.N,by=V1][order(V1)]
	init_acq_counts_panel_w_holdout = data.table(V1 = 1:M_tot)										# to deal with possibility of 0's, create DT w/ all months, merge in
	init_acq_counts_panel_w_holdout = merge(x=init_acq_counts_panel_w_holdout,y=panel_IAs_w_holdout,by="V1",all.x=TRUE)
	init_acq_counts_panel_w_holdout[is.na(N), N := 0]
	init_acq_counts_panel_w_holdout = init_acq_counts_panel_w_holdout$N

	panel_ICs_w_holdout = data.table(unlist(lapply(acqchurn_ptime_list_w_holdout, function(x) x[1,2])))[,.N,by=V1][order(V1)]
	init_churn_counts_panel_w_holdout = data.table(V1 = 1:M_tot)										# to deal with possibility of 0's, create DT w/ all months (ex-survival), merge in
	init_churn_counts_panel_w_holdout = merge(x=init_churn_counts_panel_w_holdout,y=panel_ICs_w_holdout,by="V1",all.x=TRUE)
	init_churn_counts_panel_w_holdout[is.na(N), N := 0]
	init_churn_counts_panel_w_holdout = init_churn_counts_panel_w_holdout$N

	panel_rept_activity_w_holdout = lapply(acqchurn_ptime_list_w_holdout, function(x) x[-1,,drop=FALSE])			# remove all initial activity
	panel_rept_activity_w_holdout = panel_rept_activity_w_holdout[unlist(lapply(panel_rept_activity_w_holdout, function(x) length(x)>0))]		# remove those w/ only initial activity

	panel_RAs_w_holdout = data.table(unlist(lapply(panel_rept_activity_w_holdout, function(x) x[,1])))[,.N,by=V1][order(V1)]
	rept_acq_counts_panel_w_holdout = data.table(V1 = 1:M_tot)										# to deal with possibility of 0's, create DT w/ all months (ex-survival), merge in
	rept_acq_counts_panel_w_holdout = merge(x=rept_acq_counts_panel_w_holdout,y=panel_RAs_w_holdout,by="V1",all.x=TRUE)
	rept_acq_counts_panel_w_holdout[is.na(N), N := 0]
	rept_acq_counts_panel_w_holdout = rept_acq_counts_panel_w_holdout$N

	panel_RCs_w_holdout = data.table(unlist(lapply(panel_rept_activity_w_holdout, function(x) x[,2])))[,.N,by=V1][order(V1)]
	rept_churn_counts_panel_w_holdout = data.table(V1 = 1:M_tot)										# to deal with possibility of 0's, create DT w/ all months (ex-survival), merge in
	rept_churn_counts_panel_w_holdout = merge(x=rept_churn_counts_panel_w_holdout,y=panel_RCs_w_holdout,by="V1",all.x=TRUE)
	rept_churn_counts_panel_w_holdout[is.na(N), N := 0]
	rept_churn_counts_panel_w_holdout = rept_churn_counts_panel_w_holdout$N

	# get monthly aggregated data across all population members (in or out of the panel)
	agg_addloss_sim_list_w_holdout_panel_only = list(`init_acq_counts`=init_acq_counts_panel_w_holdout,
													  init_churn_counts = init_churn_counts_panel_w_holdout,
													  rept_acq_counts = rept_acq_counts_panel_w_holdout,
													  rept_churn_counts = rept_churn_counts_panel_w_holdout)
	obs_IAC_ILC_RAC_RLC_w_holdout_ex_panel = c(unname(unlist(agg_addloss_sim_list_w_holdout))) 
	obs_IAC_ILC_RAC_RLC_w_holdout_incl_panel = obs_IAC_ILC_RAC_RLC_w_holdout_ex_panel + c(unname(unlist(agg_addloss_sim_list_w_holdout_panel_only)))		# this is the agg data w/ holdout, adding back the aggregated panel data
	obs_IAC_ILC_RAC_RLC_cal = c(unname(unlist(agg_addloss_sim_list_cal))) 													# this is the add data w/o holdout, w/o adding back aggregated panel data, for estimation
	agg_addloss_from_panel_sim_list_cal = lapply(agg_addloss_sim_list_w_holdout_panel_only, function(x) x[1:M_i])	
	obs_IAC_ILC_RAC_RLC_cal_incl_panel = obs_IAC_ILC_RAC_RLC_cal + 	c(unname(unlist(agg_addloss_from_panel_sim_list_cal)))

	# save monthly IA IC RA RC aggregated data across all population members
	saveRDS(obs_IAC_ILC_RAC_RLC_w_holdout_incl_panel, file=paste0("simulation outputs/monthly_agg_data/IA_IC_RA_RC_incl_panel_scenario_num_",i,".rds"))		# save true parameters
	
	# convert monthly aggregated data into ADD and END data - with and without holdout period
	end_time_ctime_w_holdout = seq(3,M_i+M_holdout,by=3)
	start_time_ctime_w_holdout = end_time_ctime_w_holdout-2
	ntimes_all = length(end_time_ctime_w_holdout)

	#Create table which records (over both calibration and holdout period)
	  #which disclosures are observed (i.e. ADD/END/LOSS)
	  #which months the disclosure covers
	  #e.g.: gross acquisitions in first quarter of commercial operations would be
		#start_mth_ctime = 1 (first month of commercial operations)
		#end_mth_ctime = 3 (third month of commercial operations)
		#disc = ADD (gross acquisitions)
	disc_mat_all_w_holdout = data.frame( start_mth_ctime=rep(start_time_ctime_w_holdout,3), end_mth_ctime=rep(end_time_ctime_w_holdout,3), disc=rep(c("ADD","LOSS","END"),each=ntimes_all) )
	
	#Convert this into a matrix K 
	#K encodes the linear transformation from monhtly [IA IC RA RC] vector to the observed disclosures
	K_mat_all_w_holdout = disc_to_K(disc_mat_all_w_holdout, M_i+M_holdout)
	
	# Get the corresponding objects for just the calibration period, ignoring the holdout period
	disc_mat_all_cal = disc_mat_all_w_holdout %>% filter(end_mth_ctime <= M_i)			# all disclosures (including ADD ones) for just cal period
	K_mat_all_cal = disc_to_K(disc_mat_all_cal, M_i)

	# remove redundant disclosures - ADD is deterministic function of END and LOSS, which are already in the disclosure matrix
	disc_mat_cal = disc_mat_all_cal %>% filter(disc != "ADD")							
	n_disc_cal = nrow(disc_mat_cal)
	K_mat_cal = disc_to_K(disc_mat_cal, M_i)					# this has full row rank

	# Multiply monthly IA IC RA RC vector by K_matrix to get observed disclosures
	obs_vec_cal = as.vector(K_mat_cal %*% as.matrix(obs_IAC_ILC_RAC_RLC_cal,ncol=1))			# convert monthly aggregate data to quarterly disclosures - just LOSS and END - ex-panel
	obs_vec_cal_incl_panel = as.vector(K_mat_cal %*% as.matrix(obs_IAC_ILC_RAC_RLC_cal_incl_panel,ncol=1))


	#################################################
	# ESTIMATION
	#################################################

	# setting seed for reproducibility
	set.seed(1234)

	p = ncol(X_ctime)
	
    #transform the parameters into unconstrained space (log-transform non-negative parameters, logit-transform bounded parameters)
	true_par_transformed = transform_params(true_par, p, M_i, cov_terms=TRUE)
	
	#choose a starting place for the optimization (same starting point for all estimation methods) by perturbing the true reparameterized parameters
	true_par_transformed_w_noise = true_par_transformed + rnorm(length(true_par), s = 0.2)		# perturb the true parameters
	
	# maximum number of iterations to run optimizer for
	itermax=2000


	##################
	# PROPOSED METHOD (MPL)

	# Initialize the covariance matrix of the disclosures
	MPL_covmat = diag(n_disc_cal)*.25*N_i

    #subset to only free parameters (remove parameters related to second acquisition phase, which are fixed, as well as the covariate, which is set to 0)
	init_par_wo_ac1ac2_transformed = true_par_transformed_w_noise[-which(parnames=="sig_ac1_ac2")]
	parnames_wo_ac1ac2 = parnames[-which(parnames %in% c("sig_ac1_ac2")) ]
	init_free_params_MPL = init_par_wo_ac1ac2_transformed[-c(grep("pi_2",parnames_wo_ac1ac2),	
																			 grep("m_star",parnames_wo_ac1ac2),
																			 grep("ac2",parnames_wo_ac1ac2),
																			 grep("b",substr(parnames_wo_ac1ac2,1,4)))]

    #Perform first stage of MPL estimation using non-linear minimizer (nlm)
	MPL_stage_1_results = nlm(f=nll_full_reparam_allpars_multicoh_no_ac2_no_covariate,  #function to minimize (negative proxy likelihood)
					p=init_free_params_MPL, 								#initial parameters
					sims = sims_draws,										#mixing distribution simulations for numerical integration
					acqchurn_ptime_list = acqchurn_ptime_list_cal,			#panel data for active panel members
					obs_vec = obs_vec_cal,									#aggregate data
					panel_inactive_pop = panel_inactive_pop_cal,			#number of panel members not acquired during panel period
					prospect_pools = prospect_pools,						#birth time and population size of prospect pools
					X_ctime = X_ctime,										#covariates (irrelevant, as this the coefficient associated with covariate is set to 0)
					left_trunc_time = left_trunc_time,						#start time ("month 0") of panel
					M_company = M_i,										#time horizon of analysis
					parnames=parnames,										#names of full parameter vector
					K_mat=K_mat_cal,										#linear transformation matrix (for calibration period aggregatedd ata)
					diag_covmat=TRUE, #tells objective function not to continuously update covariance matrix
					covmat_precomputed=MPL_covmat,					#feeds in the initialized (inverse) weight matrix
					iterlim=itermax,										#maximum iterations for optimization
					print.level = 2)										#tells optimizer to print progress at every iteration

	saveRDS(MPL_stage_1_results, file=paste0("simulation outputs/param_ests/results_nlm1_proposed_scenario_num_",i,".rds"))

									 

	# Do second stage estimator for MPL method
	
	# pull out the parameter estimate from first-stage nlm estimation
	stage_1_free_params_MPL = MPL_stage_1_results$estimate				

	# precompute the covariance matrix with the exact likelihood
	init_par_wo_ac1ac2_transformed[-unique(c(grep("pi_2",parnames_wo_ac1ac2),
											  grep("m_star",parnames_wo_ac1ac2),
											  grep("ac2",parnames_wo_ac1ac2),
											  grep("b_",substr(parnames_wo_ac1ac2,1,4))))] = stage_1_free_params_MPL		# insert estimated free parameters into parameter vector
	init_par_wo_ac1ac2_transformed[grep("pi_2",parnames_wo_ac1ac2)]= -10000				# as noted earlier, we "shut off" the second acquisition segment by setting its size to zero
	par_transformed = c(init_par_wo_ac1ac2_transformed[1:(5*p + 15)], 0, init_par_wo_ac1ac2_transformed[-(1:(5*p + 15))])			# Insert back in a 0 for covariance(ac1,ac2) term - the ac2 process is excluded
	names(par_transformed) = parnames												# add back names to the parameter vector
	est_par = inverse_transform_params(par_transformed, p, M_i, cov_terms=TRUE)		# transform the reparameterized parameters back to the original parameterization of the parameters


    # Compute covariance matrix of disclosures at estimated parameter values
    # This is to be used as the (inverse) weight matrix 
	MPL_covmat = get_expected_disclosures(est_par, 					
							   sims = sims_draws,
							   acqchurn_ptime_list = acqchurn_ptime_list_cal,
							   panel_inactive_pop = panel_inactive_pop_cal,
							   prospect_pools = prospect_pools,
							   X_ctime = X_ctime,
							   left_trunc_time = left_trunc_time,
							   M_company = M_i,
							   K_mat = K_mat_cal,
							   diag_covmat = FALSE, 
							   get_covmat = TRUE)

    #Perform second stage of MPL estimation using non-linear minimizer (nlm)
	MPL_stage_2_results = nlm(f=nll_full_reparam_allpars_multicoh_no_ac2_no_covariate,  #function to minimize (negative proxy likelihood)
					p=stage_1_free_params_MPL, 								#initial parameters
					sims = sims_draws,										#mixing distribution simulations for numerical integration
					acqchurn_ptime_list = acqchurn_ptime_list_cal,			#panel data for active panel members
					obs_vec = obs_vec_cal,									#aggregate data
					panel_inactive_pop = panel_inactive_pop_cal,			#number of panel members not acquired during panel period
					prospect_pools = prospect_pools,						#birth time and population size of prospect pools
					X_ctime = X_ctime,										#covariates (irrelevant, as this the coefficient associated with covariate is set to 0)
					left_trunc_time = left_trunc_time,						#start time ("month 0") of panel
					M_company = M_i,										#time horizon of analysis
					parnames=parnames,										#names of full parameter vector
					K_mat=K_mat_cal,										#linear transformation matrix (for calibration period aggregatedd ata)
					diag_covmat=TRUE, #tells objective function not to continuously update covariance matrix
					covmat_precomputed=MPL_covmat,					#feeds in the initialized (inverse) weight matrix
					iterlim=itermax,										#maximum iterations for optimization
					print.level = 2)										#tells optimizer to print progress at every iteration
	
	# print notification that this estimation for this scenario is completed
	print(paste0("scenario number ",i,", MPL method estimation done at ",Sys.time()))
	
	# save output from 2nd stage MPL estimation
	saveRDS(MPL_stage_2_results, file=paste0("simulation outputs/param_ests/results_nlm2_proposed_scenario_num_",i,".rds"))





	##################
	# AGG ONLY METHOD

	# Initialize the covariance matrix of the disclosures
	AGG_covmat = diag(n_disc_cal)*.25*N_i		

	# for AGG method, remove ac2 inds,  one covariate, and selection parameters, which are all fixed
	init_par_wo_ac1ac2_transformed = true_par_transformed_w_noise[-which(names(true_par_transformed_w_noise)=="sig_ac1_ac2")]
	init_free_params_AGG = init_par_wo_ac1ac2_transformed[-c(grep("pi_2",parnames_wo_ac1ac2),			
															 grep("m_star",parnames_wo_ac1ac2),
															 grep("ac2",parnames_wo_ac1ac2),
															 grep("b",substr(parnames_wo_ac1ac2,1,4)),
															 grep("sel_",parnames_wo_ac1ac2))]

    #Perform first stage of AGG estimation using non-linear minimizer (nlm)
    #same as MPL but without panel data (zeroes out panel selection parameters since there is no panel data to have selection bias on)
	# We only provide comments for rows that are different from the corresponding rows in the MPL estimation procedure
	AGG_stage_1_results = nlm(f=nll_full_reparam_allpars_multicoh_no_ac2_no_covariate_no_panel_sel, 
					p=init_free_params_AGG, 									# initial parameters for AGG
					sims = sims_draws,
					acqchurn_ptime_list = NULL,									# there is no panel data for AGG
					obs_vec = obs_vec_cal_incl_panel,							
					panel_inactive_pop = 0,										# there is no panel data for AGG
					prospect_pools = prospect_pools,
					X_ctime = X_ctime,
					left_trunc_time = left_trunc_time,
					M_company = M_i,
					parnames=parnames,
					K_mat=K_mat_cal,
					diag_covmat=TRUE,  
					covmat_precomputed=AGG_covmat,	
					iterlim=itermax,
					print.level = 2)

	# save output from 1st stage AGG estimation
	saveRDS(AGG_stage_1_results, file=paste0("simulation outputs/param_ests/results_nlm1_aggonly_scenario_num_",i,".rds"))

	
	# do second stage estimator for agg-only method
	
	# pull out the parameter estimate from first-stage nlm estimation
	stage_1_free_params_AGG = AGG_stage_1_results$estimate				

	
	# When estimating parameters using AGG, there is no selection process assumed. Everyone is in the aggregate data, and only the aggregate data is used for estimation.
	# As before, we perform estimation upon the free parameters
	init_par_wo_ac1ac2_transformed[-unique(c(grep("pi_2",parnames_wo_ac1ac2),
												grep("m_star",parnames_wo_ac1ac2),
												grep("ac2",parnames_wo_ac1ac2),
												grep("b_",substr(parnames_wo_ac1ac2,1,4)),
												grep("sel_",parnames_wo_ac1ac2)))] = stage_1_free_params_AGG		# Insert estimated free parameters into parameter vector
	init_par_wo_ac1ac2_transformed[grep("pi_2",parnames_wo_ac1ac2)]= -10000											# As noted earlier, we "shut off" the second acquisition segment by setting its size to zero
	init_par_wo_ac1ac2_transformed[grep("sel_b_0",parnames_wo_ac1ac2)]= -12				# By setting sel_b_0 = -12, all population members have effectively a 0% probability of being selected into the panel.									
	
	par_transformed = c(init_par_wo_ac1ac2_transformed[1:(5*p + 15)], 0, init_par_wo_ac1ac2_transformed[-(1:(5*p + 15))])	# Insert back in a 0 for covariance(ac1,ac2) term - the ac2 process is excluded
	names(par_transformed) = parnames												# add back names to the parameter vector
	est_par = inverse_transform_params(par_transformed, p, M_i, cov_terms=TRUE)		# transform the reparameterized parameters back to the original parameterization of the parameters
	
    # Compute covariance matrix of disclosures at estimated parameter values
    # This is to be used as the (inverse) weight matrix 
	AGG_covmat = get_expected_disclosures(est_par, 				
							   sims = sims_draws,
							   acqchurn_ptime_list = NULL,
							   panel_inactive_pop = 0,
							   prospect_pools = prospect_pools,
							   X_ctime = X_ctime,
							   left_trunc_time = left_trunc_time,
							   M_company = M_i,
							   K_mat = K_mat_cal,
							   diag_covmat = FALSE, 
							   get_covmat = TRUE)


    #Perform second stage of AGG estimation using non-linear minimizer (nlm)
	# Because of weak identification of some of the subprocesses estimated using AGG, the estimated covariance matrix
	# can sometimes be degenerate, causing the 2nd stage of AGG estimation to error out. If the covariance matrix is 
	# degenerate, the 2nd stage AGG estimate is the same as the 1st stage AGG estimate. 
	if(sum(is.nan(AGG_covmat))==0){
		AGG_stage_2_results = nlm(f=nll_full_reparam_allpars_multicoh_no_ac2_no_covariate_no_panel_sel, 
						p=stage_1_free_params_AGG, 
						sims = sims_draws,
						acqchurn_ptime_list = NULL,									
						obs_vec = obs_vec_cal_incl_panel,
						panel_inactive_pop = 0,										
						prospect_pools = prospect_pools,
						X_ctime = X_ctime,
						left_trunc_time = left_trunc_time,
						M_company = M_i,
						parnames=parnames,
						K_mat=K_mat_cal,
						diag_covmat=TRUE,  
						covmat_precomputed=AGG_covmat,	
						iterlim=itermax,
						print.level = 2)
		}
		
	# print notification that this estimation for this scenario is completed
	print(paste0("scenario number ",i,", agg-only estimation done at ",Sys.time()))

	# save output from 2nd stage AGG estimation
	saveRDS(AGG_stage_2_results, file=paste0("simulation outputs/param_ests/results_nlm2_aggonly_scenario_num_",i,".rds"))





	##################
	# PANEL ONLY METHOD

	# For PAN, our posited data generating process is that the population is the panel, and thus all population members are selected into the panel
	# Analogous to AGG, this is coded into the model by setting sel_b_0 equal to a very large number, so that the effective probability of being selected into the panel is 100%.
	sel_b_0_val_panel_only = 30									
	prospect_pools_panel_only = prospect_pools
	prospect_pools_panel_only[,"pool_size"] = length(acqchurn_ptime_list_cal)+panel_inactive_pop_cal		# now the size of the population is equal to the size of the panel

	# Initialize the covariance matrix of the disclosures
	PAN_covmat = diag(n_disc_cal)*.25*N_i		# initialize - but doesn't matter for panel-only fit (won't update either b/c of this)

	# As with AGG, we subset parameters down to just the free parameters, excluding the second acquisition process, the covariate, and the selection process parameters which are all fixed
	init_par_wo_ac1ac2_transformed = true_par_transformed_w_noise[-which(names(true_par_transformed_w_noise)=="sig_ac1_ac2")]
	init_free_params_PAN = init_par_wo_ac1ac2_transformed[-c(grep("pi_2",parnames_wo_ac1ac2),			# for proposed method, remove ac2 inds and one covariate
																			 grep("m_star",parnames_wo_ac1ac2),
																			 grep("ac2",parnames_wo_ac1ac2),
																			 grep("b",substr(parnames_wo_ac1ac2,1,4)),
																			 grep("sel_",parnames_wo_ac1ac2))]

    # Perform first and only stage of PAN estimation using non-linear minimizer (nlm)
	# No second stage of estimation is required for PAN
	# We only provide comments for rows that are different from the corresponding rows in the MPL and AGG estimation procedures
	PAN_results = nlm(f=nll_full_reparam_allpars_multicoh_no_ac2_no_covariate_no_panel_sel, 
					p=init_free_params_PAN, 
					sims = sims_draws,
					acqchurn_ptime_list = acqchurn_ptime_list_cal,
					obs_vec = NULL,											# PAN estimation means there is no aggregate data that is observed
					panel_inactive_pop = panel_inactive_pop_cal,
					prospect_pools = prospect_pools_panel_only,
					X_ctime = X_ctime,
					left_trunc_time = left_trunc_time,
					M_company = M_i,
					parnames=parnames,
					K_mat=K_mat_cal,
					diag_covmat=TRUE,  
					covmat_precomputed=PAN_covmat,
					sel_b_0_val=sel_b_0_val_panel_only,						# same as previous estimation, except that the sel_b_0 parameter is fixed upon a value of 30, instead of -12 with AGG
					iterlim=itermax,
					print.level = 2)
					
	# print notification that this estimation for this scenario is completed
	print(paste0("scenario number ",i,", panel-only estimation done at ",Sys.time()))

	# save output from PAN estimation
	saveRDS(PAN_results, file=paste0("simulation outputs/param_ests/results_nlm2_panelonly_scenario_num_",i,".rds"))

	
	}




# get all scenario numbers that have all the estimates fully completed (PAN is the final estimation in the loop above)
param_est_files = list.files(paste0("simulation outputs/param_ests/"))
all_ests = param_est_files[grep("results_nlm1_panelonly_scenario",param_est_files)]
scenarios_all_results = sort(as.numeric(unlist(lapply(strsplit(gsub(".rds","",all_ests),"num_"), function(x){x[2]}))))
n_scenarios_all_results = length(scenarios_all_results)

disclosures = c("QIA","QIC","QRA","QRC","QADD","QLOSS","END")
disclosures_vec = rep(c(disclosures),each=Q_holdout)		# this is what we obtain actual and expected values for, for each scenario
n_disclosures_vec = length(disclosures_vec)

# initialize where we will store the outputs
all_output = NULL
						 
# Make MAPE table to compare performance for this set of parameters across data settings
for(i in scenarios_all_results){

	# Get data settings
	M_i = all_settings_vals[i,]$M
	Q_i = M_i/3
	M_tot = M_i + M_holdout
	Q_tot = M_tot/3
	X_ctime_w_holdout = cbind(Q1 = rep(0,M_tot))	# initialize a covariate matrix because function expects one, but zero it out
	p = ncol(X_ctime_w_holdout)
	X_ctime = X_ctime_w_holdout[1:M_i,,drop=FALSE]
	N_i = all_settings_vals[i,]$N
	
	# Get actual "granular" data
	obs_IAC_ILC_RAC_RLC_w_holdout_incl_panel = readRDS(file=paste0("simulation outputs/monthly_agg_data/IA_IC_RA_RC_incl_panel_scenario_num_",i,".rds"))
	
	# get observed monthly figures
	obs_IA = obs_IAC_ILC_RAC_RLC_w_holdout_incl_panel[1:M_tot]
	obs_IC = obs_IAC_ILC_RAC_RLC_w_holdout_incl_panel[M_tot+1:M_tot]
	obs_RA = obs_IAC_ILC_RAC_RLC_w_holdout_incl_panel[2*M_tot+1:M_tot]
	obs_RC = obs_IAC_ILC_RAC_RLC_w_holdout_incl_panel[3*M_tot+1:M_tot]
	obs_END = cumsum(obs_IA)+cumsum(obs_RA)-cumsum(obs_IC)-cumsum(obs_RC)
	
	# get observed quarterly figures
	obs_QIA = rollapplyr(obs_IA,3,sum)[seq(1,M_tot,by=3)]
	obs_QIC = rollapplyr(obs_IC,3,sum)[seq(1,M_tot,by=3)]
	obs_QRA = rollapplyr(obs_RA,3,sum)[seq(1,M_tot,by=3)]
	obs_QRC = rollapplyr(obs_RC,3,sum)[seq(1,M_tot,by=3)]
	obs_QEND = obs_END[seq(3,M_tot,by=3)]
	obs_QADD = obs_QIA + obs_QRA
	obs_QLOSS = obs_QIC + obs_QRC
	
	# get only quarterly holdout figures
	obs_QIA_holdout = obs_QIA[Q_i+1:Q_holdout]
	obs_QIC_holdout = obs_QIC[Q_i+1:Q_holdout]
	obs_QRA_holdout = obs_QRA[Q_i+1:Q_holdout]
	obs_QRC_holdout = obs_QRC[Q_i+1:Q_holdout]
	obs_QADD_holdout = obs_QADD[Q_i+1:Q_holdout]
	obs_QLOSS_holdout = obs_QLOSS[Q_i+1:Q_holdout]
	obs_QEND_holdout = obs_QEND[Q_i+1:Q_holdout]
	
	# Collect all observed data into an object
	Actual = c(obs_QIA_holdout,
				obs_QIC_holdout,
				obs_QRA_holdout,
				obs_QRC_holdout,
				obs_QADD_holdout,
				obs_QLOSS_holdout,
				obs_QEND_holdout)
	
	#create matrix giving the list of prospect pools - one prospect pool for simulations
	prospect_pools = cbind(
	  birth_month = seq(0, M_i-1),
	  pool_size = c(N_i,rep(0,M_i-1)))
	prospect_pools = prospect_pools[prospect_pools[,2] > 0,,drop=FALSE]

	# convert monthly aggregated data into ADD and END data - with and without holdout
	end_time_ctime_w_holdout = seq(3,M_i+M_holdout,by=3)
	start_time_ctime_w_holdout = end_time_ctime_w_holdout-2
	ntimes_all = length(end_time_ctime_w_holdout)

	# get full K matrix for granular holdout figures
	K_mat_all_monthly = diag(4*M_tot)
	
	# Collect parameter estimates for the estimation methods: proposed, agg-only, panel-only
	for(j in 1:n_methods){
		init_par_wo_ac1ac2_transformed = rep(0,length(parnames_wo_ac1ac2))			
		if(j == 1){			# proposed
			init_par_wo_ac2_elmts_transformed = tryCatch(readRDS(file=paste0("simulation outputs/param_ests/results_nlm2_proposed_scenario_num_",i,".rds"))$estimate, 
									error = function(e) {NA})
			ac2_and_cov_inds = unique(c(grep("pi_2",parnames_wo_ac1ac2),grep("m_star",parnames_wo_ac1ac2),grep("ac2",parnames_wo_ac1ac2),grep("b_",substr(parnames_wo_ac1ac2,1,4))))		# identify ac2-related and recession-related inds
			}
		if(j==2){				# panel-only
			init_par_wo_ac2_elmts_transformed = tryCatch(readRDS(file=paste0("simulation outputs/param_ests/results_nlm2_panelonly_scenario_num_",i,".rds"))$estimate, 
									error = function(e) {NA})
			ac2_and_cov_inds = unique(c(grep("pi_2",parnames_wo_ac1ac2),grep("m_star",parnames_wo_ac1ac2),grep("ac2",parnames_wo_ac1ac2),grep("b_",substr(parnames_wo_ac1ac2,1,4)),grep("sel_",parnames_wo_ac1ac2)))		# identify ac2-related and recession-related inds
			}
		if(j == 3){			# agg only
			init_par_wo_ac2_elmts_transformed = tryCatch(readRDS(file=paste0("simulation outputs/param_ests/results_nlm2_aggonly_scenario_num_",i,".rds"))$estimate, 
									error = function(e) {NA})
			ac2_and_cov_inds = unique(c(grep("pi_2",parnames_wo_ac1ac2),grep("m_star",parnames_wo_ac1ac2),grep("ac2",parnames_wo_ac1ac2),grep("b_",substr(parnames_wo_ac1ac2,1,4)),grep("sel_",parnames_wo_ac1ac2)))		# identify ac2-related and recession-related inds
			}
		if(!is.na(init_par_wo_ac2_elmts_transformed[1])){
			init_par_wo_ac1ac2_transformed[-ac2_and_cov_inds] = init_par_wo_ac2_elmts_transformed
			init_par_wo_ac1ac2_transformed[grep("pi_2",parnames_wo_ac1ac2)]= -10000
			par_transformed = c(init_par_wo_ac1ac2_transformed[1:(5*p + 15)], 0, init_par_wo_ac1ac2_transformed[-(1:(5*p + 15))])			# Splice back in a 0 for covariance(ac1,ac2) term
			names(par_transformed) = parnames			# add back names to the parameter vector
			est_par = inverse_transform_params(par_transformed, p, M_i, cov_terms=TRUE)		# reparams => params
			est_par[is.nan(est_par)]=0
			est_par[grep("sel_",parnames)]=0; est_par[grep("sel_b_0",parnames)]=-12		# estimations are done so that we should be forecasting the entire population at this step
			
			# Get monthly [IA IC RA RC] for holdout
			Eobs_vec_monthly_holdout = as.numeric(get_expected_disclosures(est_par, 					# these figures are consistent with Eobs_vec
																		   sims = sims_draws,
																		   acqchurn_ptime_list = NULL,
																		   panel_inactive_pop = 0,
																		   prospect_pools = prospect_pools,
																		   X_ctime = X_ctime_w_holdout,
																		   left_trunc_time = left_trunc_time,
																		   M_company = M_tot,
																		   K_mat = K_mat_all_monthly))

			# get expected monthly figures
			E_IA = Eobs_vec_monthly_holdout[1:M_tot]
			E_IC = Eobs_vec_monthly_holdout[M_tot+1:M_tot]
			E_RA = Eobs_vec_monthly_holdout[2*M_tot+1:M_tot]
			E_RC = Eobs_vec_monthly_holdout[3*M_tot+1:M_tot]
			E_END = cumsum(E_IA)+cumsum(E_RA)-cumsum(E_IC)-cumsum(E_RC)
			
			# get expected quarterly figures
			E_QIA = rollapplyr(E_IA,3,sum)[seq(1,M_tot,by=3)]
			E_QIC = rollapplyr(E_IC,3,sum)[seq(1,M_tot,by=3)]
			E_QRA = rollapplyr(E_RA,3,sum)[seq(1,M_tot,by=3)]
			E_QRC = rollapplyr(E_RC,3,sum)[seq(1,M_tot,by=3)]
			E_QEND = E_END[seq(3,M_tot,by=3)]
			E_QADD = E_QIA + E_QRA
			E_QLOSS = E_QIC + E_QRC
			
			# get expected quarterly holdout figures
			E_QIA_holdout = E_QIA[Q_i+1:Q_holdout]
			E_QIC_holdout = E_QIC[Q_i+1:Q_holdout]
			E_QRA_holdout = E_QRA[Q_i+1:Q_holdout]
			E_QRC_holdout = E_QRC[Q_i+1:Q_holdout]
			E_QADD_holdout = E_QADD[Q_i+1:Q_holdout]
			E_QLOSS_holdout = E_QLOSS[Q_i+1:Q_holdout]
			E_QEND_holdout = E_QEND[Q_i+1:Q_holdout]
			
			# compile expected results
			Expected = c(E_QIA_holdout,
				E_QIC_holdout,
				E_QRA_holdout,
				E_QRC_holdout,
				E_QADD_holdout,
				E_QLOSS_holdout,
				E_QEND_holdout)		
	
			# summarize all outputs
			new_results = cbind.data.frame(all_settings_vals[rep(i,n_disclosures_vec),,drop=FALSE],
											Method=methods[j],
											Disclosure=disclosures_vec,
											Horizon=rep(1:Q_holdout,7),
											Actual,
											Expected)
			
			# add new output to existing output
			all_output = rbind.data.frame(all_output,new_results)
			}

		}

	print(i)
	}


# Add column for absolute percentage error 
all_output = all_output %>%
		mutate(APE = abs((Actual-Expected)/Actual))
		
saveRDS(all_output, file=paste0("simulation outputs/all_output.rds"))
# all_output = readRDS(file=paste0("simulation outputs/all_output.rds"))		# uncomment to reload it in if starting from a fresh R session
	
	
		
	


# Get MAPE by disclosure - average up to 6 quarters ahead
MAPE_overall_results_6Qahead = all_output %>%
	dplyr::filter(Horizon<=6) %>%
	dplyr::group_by(Method,Disclosure) %>%
	dplyr::summarize(MAPE = round(mean(APE,na.rm=TRUE),3)) %>%
	ungroup %>%
	tidyr::spread(key = Method, value = MAPE) %>%
	dplyr::rename(Proposed = proposed_2_stage, Panel_Only = panel, Aggregate_Only = agg_2_stage) %>%
	dplyr::select(Disclosure, Panel_Only, Aggregate_Only, Proposed) %>%
	dplyr::mutate(Disclosure=factor(Disclosure, levels=disclosures)) %>%
	dplyr::arrange(Disclosure)
	
MAPE_overall_results_6Qahead_by_M = all_output %>%
	dplyr::filter(Horizon<=6) %>%
	dplyr::group_by(Method,Disclosure,M) %>%
	dplyr::summarize(MAPE = round(mean(APE,na.rm=TRUE),3)) %>%
	ungroup %>%
	tidyr::spread(key = Method, value = MAPE) %>%
	dplyr::rename(Proposed = proposed_2_stage, Panel_Only = panel, Aggregate_Only = agg_2_stage) %>%
	dplyr::select(Disclosure, M, Panel_Only, Aggregate_Only, Proposed) %>%
	dplyr::mutate(Disclosure=factor(Disclosure, levels=disclosures)) %>%
	dplyr::arrange(Disclosure,M) %>%
	dplyr::mutate(Data_Setting='M') %>%
	dplyr::select(Data_Setting, Disclosure, everything()) %>%
	tidyr::gather(Method,MAPE,'Panel_Only':'Proposed')
											

MAPE_overall_results_6Qahead_by_N = all_output %>%
	dplyr::filter(Horizon<=6) %>%
	dplyr::group_by(Method,Disclosure,N) %>%
	dplyr::summarize(MAPE = round(mean(APE,na.rm=TRUE),3)) %>%
	ungroup %>%
	tidyr::spread(key = Method, value = MAPE) %>%
	dplyr::rename(Proposed = proposed_2_stage, Panel_Only = panel, Aggregate_Only = agg_2_stage) %>%
	dplyr::select(Disclosure, N, Panel_Only, Aggregate_Only, Proposed) %>%
	dplyr::mutate(Disclosure=factor(Disclosure, levels=disclosures)) %>%
	dplyr::arrange(Disclosure,N) %>%
	dplyr::mutate(Data_Setting='N') %>%
	dplyr::select(Disclosure, everything()) %>%
	tidyr::gather(Method,MAPE,'Panel_Only':'Proposed')


MAPE_overall_results_6Qahead_by_panel_pct = all_output %>%
	dplyr::filter(Horizon<=6) %>%
	dplyr::group_by(Method,Disclosure,panel_pct) %>%
	dplyr::summarize(MAPE = round(mean(APE,na.rm=TRUE),3)) %>%
	ungroup %>%
	tidyr::spread(key = Method, value = MAPE) %>%
	dplyr::rename(Proposed = proposed_2_stage, Panel_Only = panel, Aggregate_Only = agg_2_stage) %>%
	dplyr::select(Disclosure, panel_pct, Panel_Only, Aggregate_Only, Proposed) %>%
	dplyr::mutate(Disclosure=factor(Disclosure, levels=disclosures)) %>%
	dplyr::arrange(Disclosure,panel_pct) %>%
	dplyr::mutate(Data_Setting='Panel Size') %>%
	dplyr::select(Disclosure, everything()) %>%
	tidyr::gather(Method,MAPE,'Panel_Only':'Proposed')


sel_levels=c("None","Medium","High")
MAPE_overall_results_6Qahead_by_sel_bias = all_output %>%
	dplyr::filter(Horizon<=6) %>%
	dplyr::group_by(Method,Disclosure,sel_bias) %>%
	dplyr::summarize(MAPE = round(mean(APE,na.rm=TRUE),3)) %>%
	ungroup %>%
	tidyr::spread(key = Method, value = MAPE) %>%
	dplyr::rename(Proposed = proposed_2_stage, Panel_Only = panel, Aggregate_Only = agg_2_stage) %>%
	dplyr::select(Disclosure, sel_bias, Panel_Only, Aggregate_Only, Proposed) %>%
	dplyr::mutate(Disclosure=factor(Disclosure, levels=disclosures)) %>%
	dplyr::arrange(Disclosure,sel_bias)%>%
	dplyr::mutate(sel_bias=sel_levels[match(sel_bias,sel_bias_levels)]) %>%
	dplyr::mutate(Data_Setting='Panel Bias') %>%
	dplyr::select(Disclosure, everything()) %>%
	tidyr::gather(Method,MAPE,'Panel_Only':'Proposed')

		

# Tables for overall results, with additional formatting to be appropriate for the manuscript
MAPE_overall_results_pretty = MAPE_overall_results_6Qahead %>%
	mutate_at(vars(Panel_Only:Proposed), percent_format(.1)) 
	
MAPE_overall_results_6Qahead_by_M_pretty = MAPE_overall_results_6Qahead_by_M %>%
											spread(key = Method, value = MAPE) %>%
											select(Disclosure, M, Panel_Only, Aggregate_Only, Proposed) %>%
											mutate_at(vars(Panel_Only:Proposed), percent_format(.1)) 
		
MAPE_overall_results_6Qahead_by_N_pretty = MAPE_overall_results_6Qahead_by_N %>%
											spread(key = Method, value = MAPE) %>%
											select(Disclosure, N, Panel_Only, Aggregate_Only, Proposed) %>%
											mutate_at(vars(Panel_Only:Proposed), percent_format(.1)) %>%
											mutate_at(vars(N), number_format(scale = 1e-3, suffix = "K", big.mark=",")) 

MAPE_overall_results_6Qahead_by_sel_bias_pretty = MAPE_overall_results_6Qahead_by_sel_bias %>%
											spread(key = Method, value = MAPE) %>%
											select(Disclosure, sel_bias, Panel_Only, Aggregate_Only, Proposed) %>%
											mutate_at(vars(Panel_Only:Proposed), percent_format(.1)) 

MAPE_overall_results_6Qahead_by_panel_pct_pretty = MAPE_overall_results_6Qahead_by_panel_pct %>%
											spread(key = Method, value = MAPE) %>%
											select(Disclosure, panel_pct, Panel_Only, Aggregate_Only, Proposed) %>%
											mutate_at(vars(panel_pct:Proposed), percent_format(.1)) 

# Generate latex outputs - Table 1 in paper											
table_MAPE_output_1 = kable(MAPE_overall_results_pretty, 
		format = 'latex', 
		align = c('l','c','c','c'),
		booktabs = TRUE)

# Generate latex outputs - Tables 1, 2, 3, and 4 in online supplement, Section C
table_MAPE_output_2 = kable(MAPE_overall_results_6Qahead_by_M_pretty, 
		format = 'latex', 
		align = c('l','c','c','c','c'),
		booktabs = TRUE)

table_MAPE_output_3 = kable(MAPE_overall_results_6Qahead_by_N_pretty, 
		format = 'latex', 
		align = c('l','c','c','c','c'),
		booktabs = TRUE)

table_MAPE_output_4 = kable(MAPE_overall_results_6Qahead_by_sel_bias_pretty, 
		format = 'latex', 
		align = c('l','c','c','c','c'),
		booktabs = TRUE)

table_MAPE_output_5 = kable(MAPE_overall_results_6Qahead_by_panel_pct_pretty, 
		format = 'latex', 
		align = c('l','c','c','c','c'),
		booktabs = TRUE)

# Save table outputs
saveRDS(list(table_MAPE_output_1=table_MAPE_output_1,
			 table_MAPE_output_2=table_MAPE_output_2,
			 table_MAPE_output_3=table_MAPE_output_3,
			 table_MAPE_output_4=table_MAPE_output_4,
			 table_MAPE_output_5=table_MAPE_output_5),file="simulation outputs/outputs_for_paper/table_outputs_latex_MAPE_results.rds")

			 
# if we obtained all scenario outputs, make other figures from paper			 
if(do_abridged_code == FALSE){


	data_setting_names = c(`M` = "Data Length (Months)",
							`N` = "Total Population Size",
							`Panel Size` = "Panel Size, % of Population",
							`Panel Bias` = "Panel Selection Bias")										
	
	# initialize list objects that we will store the ggplot outputs in, by data setting
	plot_M = list()						
	plot_N = list()						
	plot_sel_bias = list()						
	plot_panel_pct = list()						
	
	# set upper y limit on graphs for MAPE figures 
	upper_ylim = c('QIA'=.45,'QRC'=.3,'QIC'=.32,'QRA'=.28)
		
	# make facet plot showing average MAPE figures by data setting, disclosure, and method
	discs = c('QIA','QRC','QIC','QRA')
	ndiscs = length(discs)
	labs = c("AGG" = "Aggregate Only", "PAN" = "Panel Only", "MPL" = "Proposed")
	labs = c("Aggregate_Only" = "AGG", "Panel_Only" = "PAN", "Proposed" = "MPL")
	for(i in 1:ndiscs){
		disc_i = discs[i]
		
		# generate MAPE plot for number of monthly observations M by method and disclosure
		Data_Setting_i = 'M'
		plot_M[[i]] = MAPE_overall_results_6Qahead_by_M %>%
			filter((Disclosure==disc_i)&(!is.na(MAPE))) %>% 
			filter(Data_Setting==Data_Setting_i) %>%
			ggplot(aes(x=M,y=MAPE)) + 
			aes(group=Method, color=Method) + 
			geom_line() + 
			geom_point(aes(shape=Method), size=2) + 
			facet_wrap(~Data_Setting, scales="free", labeller = as_labeller(data_setting_names)) + 
			theme(
					axis.title.x=element_blank(),
					axis.title.y=element_blank(),
					panel.grid.minor.y = element_blank(),
					plot.title = element_text(hjust=.5)) + 
			scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,upper_ylim[disc_i])) + 
			scale_x_continuous(breaks=M_companys) + 
			ggthemes::scale_color_colorblind(labels = labs) +
			scale_shape_discrete(labels = labs)

		# generate MAPE plot for extent of panel bias by method and disclosure
		Data_Setting_i = 'Panel Bias'
		plot_sel_bias[[i]] = MAPE_overall_results_6Qahead_by_sel_bias %>%
			filter((Disclosure==disc_i)&(!is.na(MAPE))) %>% 
			filter(Data_Setting==Data_Setting_i) %>%
			ggplot(aes(x=sel_bias,y=MAPE)) + 
			aes(group=Method, color=Method) + 
			geom_line() + 
			geom_point(aes(shape=Method), size=2) + 
			facet_wrap(~Data_Setting, scales="free", labeller = as_labeller(data_setting_names)) + 
			theme(
					axis.title.x=element_blank(),
					axis.title.y=element_blank(),
					axis.text.y=element_blank(),
					axis.ticks.y=element_blank(),
					panel.grid.minor.y = element_blank(),
					plot.title = element_text(hjust=.5)) + 
			scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,upper_ylim[disc_i])) + 
			scale_x_discrete(expand=c(.03,.03)) + 
			ggthemes::scale_color_colorblind(labels = labs) +
			scale_shape_discrete(labels = labs)

		# generate MAPE plot for population size, N, by method and disclosure
		Data_Setting_i = 'N'
		plot_N[[i]] = MAPE_overall_results_6Qahead_by_N %>%
			filter((Disclosure==disc_i)&(!is.na(MAPE))) %>% 
			filter(Data_Setting==Data_Setting_i) %>%
			ggplot(aes(x=N,y=MAPE)) + 
			aes(group=Method, color=Method) + 
			geom_line() + 
			geom_point(aes(shape=Method), size=2) + 
			facet_wrap(~Data_Setting, scales="free", labeller = as_labeller(data_setting_names)) + 
			theme(
					axis.title.x=element_blank(),
					axis.title.y=element_blank(),
					panel.grid.minor.y = element_blank(),
					plot.title = element_text(hjust=.5)) + 
			scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,upper_ylim[disc_i])) + 
			scale_x_continuous(trans='log10', labels=number_format(scale = 1e-3, suffix = "K", big.mark=","), breaks=Ns) +
			ggthemes::scale_color_colorblind(labels = labs) +
			scale_shape_discrete(labels = labs)

		# generate MAPE plot for panel size as a % of the population by method and disclosure
		Data_Setting_i = 'Panel Size'
		plot_panel_pct[[i]] = MAPE_overall_results_6Qahead_by_panel_pct %>%
			filter((Disclosure==disc_i)&(!is.na(MAPE))) %>% 
			filter(Data_Setting==Data_Setting_i) %>%
			ggplot(aes(x=panel_pct,y=MAPE)) + 
			aes(group=Method, color=Method) + 
			geom_line() + 
			geom_point(aes(shape=Method), size=2) + 
			facet_wrap(~Data_Setting, scales="free", labeller = as_labeller(data_setting_names)) + 
			theme(legend.position='bottom',
					axis.title.x=element_blank(),
					axis.title.y=element_blank(),
					axis.text.y=element_blank(),
					axis.ticks.y=element_blank(),
					panel.grid.minor.y = element_blank(),
					plot.title = element_text(hjust=.5)) + 
			scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,upper_ylim[disc_i])) + 
			scale_x_continuous(labels=percent_format(accuracy=1), breaks=panel_pcts) +
			ggthemes::scale_color_colorblind(labels = labs) +
			scale_shape_discrete(labels = labs)
		}
		
	# make a common legend shared by all the facet plots	
	g_legend = function(a.gplot){
	  tmp = ggplot_gtable(ggplot_build(a.gplot))
	  leg = which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
	  legend = tmp$grobs[[leg]]
	  return(legend)}

	mylegend = g_legend(plot_panel_pct[[1]])

	# Generate first 2x2 plot showing all the MAPE results for disclosures IA and RC
	# This is Figure 1 in the paper
	pdf(file="simulation outputs/outputs_for_paper/avg_mape_fig_by_data_setting_IA_RC.pdf")
	p1 = grid.arrange(textGrob("Initial Acquisition", gp=gpar(fontsize=15)),
					  arrangeGrob(plot_M[[1]] + theme(legend.position="none"),
							 plot_sel_bias[[1]] + theme(legend.position="none"),
							 plot_N[[1]] + theme(legend.position="none"),
							 plot_panel_pct[[1]] + theme(legend.position="none"),
							 nrow=2),
					  textGrob("Repeat Churn", gp=gpar(fontsize=15)),
					  arrangeGrob(plot_M[[2]] + theme(legend.position="none"),
							 plot_sel_bias[[2]] + theme(legend.position="none"),
							 plot_N[[2]] + theme(legend.position="none"),
							 plot_panel_pct[[2]] + theme(legend.position="none"),
							 nrow=2),			  
					  mylegend, 
					  nrow=5,
					  heights=c(1, 10, 1, 10, 1),
					  left=textGrob("Average Mean Absolute Percentage Error", gp=gpar(fontsize=12), rot=90)
					  )	
	dev.off()	
				  
	# Generate second 2x2 plot showing all the MAPE results for disclosures IC and RA
	pdf(file="simulation outputs/outputs_for_paper/avg_mape_fig_by_data_setting_IC_RA.pdf")
	p2 = grid.arrange(textGrob("Initial Churn", gp=gpar(fontsize=15)),
					  arrangeGrob(plot_M[[3]] + theme(legend.position="none"),
							 plot_sel_bias[[3]] + theme(legend.position="none"),
							 plot_N[[3]] + theme(legend.position="none"),
							 plot_panel_pct[[3]] + theme(legend.position="none"),
							 nrow=2),
					  textGrob("Repeat Acquisition", gp=gpar(fontsize=15)),
					  arrangeGrob(plot_M[[4]] + theme(legend.position="none"),
							 plot_sel_bias[[4]] + theme(legend.position="none"),
							 plot_N[[4]] + theme(legend.position="none"),
							 plot_panel_pct[[4]] + theme(legend.position="none"),
							 nrow=2),			  
					  mylegend, 
					  nrow=5,
					  heights=c(1, 10, 1, 10, 1),
					  left=textGrob("Average Mean Absolute Percentage Error", gp=gpar(fontsize=12), rot=90)
					  )	
	dev.off()	

	}
			 







