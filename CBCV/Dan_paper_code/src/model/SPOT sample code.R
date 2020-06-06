library(tidyverse)
library(readxl)
library(lubridate)
library(knitr)
library(scales)
library(ggthemes)
library(grid)
library(gridExtra)

#propietary package contains all the code for computing objective function
#requires a working C++ compiler recent enough to support Armadillo
#may need to install Rcpp and RcppArmadillo
install.packages('MPL_0.1.0.tar.gz', repos = NULL, type = 'source')
library(MPL)

#OPTIONAL: register parallel backend to speed up computation (works on Mac/Linux)
#alternately can use doParallel package for Windows
library(doMC)
#Best to set number of parallel threads to total available threads minus 1
registerDoMC(detectCores() - 1)

set.seed(1)

###########################################################################################
############################## Loading and manipulating data ##############################
###########################################################################################

# SPOT hardcoded high-level data
SPOT_years_in_operation = 2008:2018
M_company = 120  # October 2008 to September 2018
M_holdout = 18 #Predict 18 months ahead of end of data
M_total = M_company + M_holdout
left_trunc_time = 75 # October 2008 to December 2014

# aggregated data: collected from SEC disclosures, investor reports, etc.
aggregate_data_all = read_csv('Data/SPOT_END_LOSS_data.csv')

# population data: collected from World Bank (size of global population by year)
population_data = read_excel('Data/world_pop_data.xls')
years_data = as.numeric(colnames(population_data)[-(1:7)])
population_data = cummax(as.numeric(population_data[population_data$`Country Name`=="World",][,-(1:7)]))	# make sure it's non-negative each year (already is though)

#fit a regression of annual population growth rate on year
y_y_growths = diff(population_data)/tail(population_data,-1)
n_years = length(y_y_growths)
yr_nums = 1:n_years
n_years_holdout = 3 #project through 2020 (3 years past end of data, 2017)
yr_nums_plus_holdout = 1:(n_years+1+n_years_holdout)		# add an extra 1 because the world bank data runs through 2017

#fit an ARIMA(0,1,0) model on year-on-year population growth rates
y_y_growths_tsmod1 = arima(y_y_growths, order=c(0,1,0),xreg=yr_nums)
resids = as.numeric(y_y_growths_tsmod1$residuals)
summary(lm(resids[-1]~head(resids,-1)))					# test for AR(1) of residuals - none
E_y_y_growths_tsmod1_insamp = y_y_growths-resids

# get forecasts
Xstar = n_years+1:(n_years_holdout+1)
E_y_y_growths_future = as.numeric(predict(y_y_growths_tsmod1,n.ahead=n_years_holdout+1,newxreg=Xstar)$pred)
E_y_y_growths_future = c(E_y_y_growths_tsmod1_insamp,E_y_y_growths_future)

#predict population growth rate (and population count) in future years
future_growths = E_y_y_growths_future[-(1:n_years)]
future_multipliers_off_final_year = cumprod(1+pmax(future_growths,0))				# make isotonic (although here, never turns negative), turn into a multiplier applied to final year of data

years_data_plus_holdout = c(years_data,tail(years_data,1)+1:(n_years_holdout+1))
population_data_plus_holdout = c(population_data, tail(population_data,1)*future_multipliers_off_final_year)

# get the household counts in the relevant years
household_counts_w_holdout = population_data_plus_holdout[years_data_plus_holdout %in% 
                                                            c(SPOT_years_in_operation, last(SPOT_years_in_operation)+1:2)]

#create matrix giving the list of prospect pools (annualized net increase in population)
birth_months = seq(0, M_total, by = 12)
prospect_pools_w_holdout = cbind(
  birth_month = birth_months,
  pool_size = c(household_counts_w_holdout[1], diff(household_counts_w_holdout))[1:length(birth_months)] )

#subset to only those "prospect pools" which are during the time horizon of analysis and are non-zero
prospect_pools_w_holdout = prospect_pools_w_holdout[prospect_pools[,2] > 0,]
prospect_pools = prospect_pools_w_holdout[prospect_pools_w_holdout[,"birth_month"]<M_company,]

aggregate_data = aggregate_data_all		# initialize
aggregate_data_dt = data.table(aggregate_data)

#Collect vector of observations of aggregate data
#multiply by 1E6 because figures are in units of millions
obs_IAC_ILC_RAC_RLC = 1000000 * aggregate_data$value

#Create table which records
  #which disclosures are observed (i.e. ADD/END/LOSS)
  #which months the disclosure covers
  #e.g.: gross acquisitions in first quarter of commercial operations would be
    #start_mth_ctime = 1 (first month of commercial operations)
    #end_mth_ctime = 3 (third month of commercial operations)
    #disc = ADD (gross acquisitions)
disc_mat = aggregate_data %>%
  dplyr::select(-value) %>%
  as.data.frame

#Convert this into a matrix K 
#K encodes the linear transformation from monhtly [IA IC RA RC] vector to the observed disclosures
K_mat = disc_to_K(disc_mat, M_company)

# (hardcoded) covariate matrix: quarterly seasonality
X_ctime = cbind(													
  Q1 = rep(c(rep(0,3),rep(1,3),rep(0,6)), length = M_company),
  Q2 = rep(c(rep(0,6),rep(1,3),rep(0,3)), length = M_company),
  Q3 = rep(c(rep(0,9),rep(1,3)), length = M_company)
)

#parameter names
#parameters including "_ac2" are irrelevant:
  #code allows for a 2-segment mixture in the initial acquisition process,
  #but we do not use it in our process because it is not empirically identified.
  #instead we fix the 2nd segment's parameters and set the size of the segment to 0 in our optimizations
parnames = c("mu_lam_ac1", "c_ac1", "pi_1",  rep("b_ac1", 3),  "mu_lam_ac2",
             "c_ac2", "pi_2", rep("b_ac2", 3), "m_star", "mu_lam_rac",
             "c_rac", "pi_NA", rep("b_rac", 3), "mu_lam_cc", "c_cc",
             rep("b_cc", 3), "mu_lam_rcc", "c_rcc", rep("b_rcc", 3), "sigsq_lambda_ac1",
             "sig_ac1_ac2", "sigsq_lambda_ac2", "sig_ac1_rac", "sig_ac2_rac", "sigsq_lambda_rac",
             "sig_ac1_cc", "sig_ac2_cc", "sig_rac_cc", "sigsq_lambda_cc", "sig_ac1_rcc",
             "sig_ac2_rcc", "sig_rac_rcc", "sig_cc_rcc", "sigsq_lambda_rcc", "sel_b_0",
             "sel_b_ac1", "sel_b_ac2", "sel_b_rac", "sel_b_cc", "sel_b_rcc")
npar = length(parnames)

#SPOT panel data

#Read in synthetic panel data
full_data = read_csv('Data/Spotify_memberlevel_synthetic.csv')
data_binary_mat = as.matrix(dplyr::select(full_data, `2015_01`:`2018_09`))

#Panel data is simulated for 350,000 people
total_panel_pop = 3.5E5

#27,918 members of the panel were acquired sometime during panel observation period
#the remaining "panel_inactive_pop" were either acquired before left_trunc_time or not at all
panel_inactive_pop = total_panel_pop - nrow(data_binary_mat)

#list of tables for each customer, giving all acquisition and churn times
# note that these times are in panel time (time shifted by left_trunc_time), so if acq_time = 1, that means they were acquired in January 2015.
acqchurn_ptime_list = apply(data_binary_mat, 1, function(x){ #loop through all active panel members
  changes = diff(c(0,x)) #first-differences: +1 means going from inactive to active (i.e. acquired), -1 means churned
  acq_times = which(changes == 1) #which months in which an acquisition occurred
  churn_times = which(changes == -1) #which months in which a churn occurred
  if(length(acq_times) > length(churn_times)){
    churn_times[length(churn_times) + 1] = length(x) + 1 #churn time 1 period after end of panel data indicates customers who "survived" to end of panel
  }
  return(cbind(acq_times, churn_times))}) #combine into a matrix

#Get the Halton sequence draws to be used for numerical integration
sims_draws = halton(100, 5, normal = TRUE)

#Logical toggle:
  #TRUE: estimate the parameters via AGG and MPL, starting from 2nd stage (could take a full day or more)
  #FALSE: skip estimation and load in final parameters
do_estimation = FALSE

###########################################################################################
######################## Estimation with aggregate data only (AGG) ########################
###########################################################################################

if(do_estimation){
  #this section estimates the model using only aggregate data (reduces to 2-stage GMM)
  #so that the computation doesn't take too long, we load in the parameter estimates from the first stage and just do the second stage of estimation
  #load in initial parameter values from first stage of estimation
  init_par_AGG = readRDS("SPOT parameters/AGG_init.RDS")
  
  #transform the parameters into unconstrained space (log-transform non-negative parameters, logit-transform bounded parameters)
  #this is useful so that we can use an unconstrained optimizer
  init_par_AGG_transformed = transform_params(init_par_AGG, ncol(X_ctime), M_company, cov_terms = TRUE)
  
  #subset to only free parameters 
  #remove parameters related to second acquisition phase, which are fixed
  #remove parameters related to selection bias, since with AGG there is no panel data on which to have selection bias
  init_free_params_AGG = init_par_AGG_transformed[-c(grep("pi_2",parnames), grep("m_star",parnames), grep("ac2",parnames), grep("sel_",parnames))]
  
  #Compute covariance matrix of disclosures at initial parameter values
  #This is to be used as the (inverse) weight matrix in the first stage of optimization
  AGG_covmat = get_expected_disclosures(init_par_AGG, #initial parameters
                                        sims = sims_draws, #mixing distribution simulations for numerical integration
                                        acqchurn_ptime_list = NULL, #there is no panel data for AGG
                                        panel_inactive_pop = 0, #there is no panel data for AGG
                                        prospect_pools = prospect_pools, #birth time and population size of prospect pools
                                        X_ctime = X_ctime, #covariates (quarterly seasonality dummies)
                                        left_trunc_time = left_trunc_time, #start time ("month 0") of panel
                                        M_company = M_company, #time horizon of analysis
                                        K_mat = K_mat, #linear transformation matrix
                                        diag_covmat = FALSE, #compute full covariance matrix (not just diagonal elements)
                                        get_covmat = TRUE) #return covariance matrix (instead of mean vector)
  
  t1 = Sys.time()
  #Perform second stage of AGG estimation using non-linear minimizer (nlm)
  #same as MPL but without panel data (zeroes out panel selection parameters since there is no panel data to have selection bias on)
  #took about 650 iterations/2.5 hours to run locally (parallelizing on 3 threads)
  AGG_stage2 = nlm(f = nll_full_reparam_allpars_multicoh_no_ac2_no_panel_sel, #function to minimize (negative proxy likelihood)
                   p = init_free_params_AGG, #initial parameters
                   sims = sims_draws, #mixing distribution simulations for numerical integration
                   acqchurn_ptime_list = NULL, #there is no panel data for AGG
                   obs_vec = obs_IAC_ILC_RAC_RLC, #aggregate data
                   panel_inactive_pop = 0, #there is no panel data for AGG
                   prospect_pools = prospect_pools, #birth time and population size of prospect pools
                   X_ctime = X_ctime, #covariates (quarterly seasonality dummies)
                   left_trunc_time = left_trunc_time, #start time ("month 0") of panel
                   M_company = M_company, #time horizon of analysis
                   parnames = parnames, #names of full parameter vector
                   K_mat = K_mat, #linear transformation matrix
                   diag_covmat = TRUE, #tells objective function not to continuously update covariance matrix
                   covmat_precomputed = AGG_covmat, #feeds in the initialized (inverse) weight matrix
                   iterlim = 1000, #maximum iterations for optimization
                   print.level = 2) #tells optimizer to print progress at every iteration
  t2 = Sys.time()
  difftime(t2,t1)
  
  #Save output
  saveRDS(AGG_stage2, "SPOT parameters/AGG_out.RDS")
}else{
  #Read in output if skipping estimation phase
  AGG_stage2 = readRDS("SPOT parameters/AGG_out.RDS")
}

###########################################################################################
################## Estimation with aggregate and disaggregate data (MPL) ##################
###########################################################################################

if(do_estimation){
  #this section estimates the model using only both the aggregate and disaggregate data (using the proposed MPL method)
  #so that the computation doesn't take too long, we load in the parameter estimates from the first stage and just do the second stage of estimation
  #load in initial parameter values from first stage of estimation
  init_par_MPL = readRDS("SPOT parameters/MPL_init.RDS")
  
  #Compute covariance matrix of disclosures at initial parameter values
  #This is to be used as the (inverse) weight matrix in the first stage of optimization
  MPL_covmat = get_expected_disclosures(init_par_MPL, #initial parameters
                                        sims = sims_draws, #mixing distribution simulations for numerical integration
                                        acqchurn_ptime_list = acqchurn_ptime_list, #panel data for active panel members
                                        panel_inactive_pop = panel_inactive_pop, #number of panel members not acquired during panel
                                        prospect_pools = prospect_pools, #birth time and population size of prospect pools
                                        X_ctime = X_ctime, #covariates (quarterly seasonality dummies)
                                        left_trunc_time = left_trunc_time, #start time ("month 0") of panel
                                        M_company = M_company, #time horizon of analysis
                                        K_mat = K_mat, #linear transformation matrix
                                        diag_covmat = FALSE, #compute full covariance matrix (not just diagonal elements)
                                        get_covmat = TRUE) #return covariance matrix (instead of mean vector)
  
  #transform the parameters into unconstrained space (log-transform non-negative parameters, logit-transform bounded parameters)
  init_par_MPL_transformed = transform_params(init_par_MPL, ncol(X_ctime), M_company, cov_terms = TRUE)
  
  #subset to only free parameters (remove parameters related to second acquisition phase, which are fixed)
  init_free_params_MPL = init_par_MPL_transformed[-c(grep("pi_2",parnames), grep("m_star",parnames), grep("ac2",parnames))]
  
  t1 = Sys.time()
  #Perform second stage of MPL estimation using non-linear minimizer (nlm)
  #took about 500 iterations/18 hours to run locally (parallelizing on 3 threads)
  MPL_stage2 = nlm(f = nll_full_reparam_allpars_multicoh_no_ac2, #function to minimize (negative proxy likelihood)
                   p = init_free_params_MPL, #initial parameters
                   sims = sims_draws, #mixing distribution simulations for numerical integration
                   acqchurn_ptime_list = acqchurn_ptime_list, #granular data for panel members acquired during panel period
                   obs_vec = obs_IAC_ILC_RAC_RLC, #aggregate data
                   panel_inactive_pop = panel_inactive_pop, #number of panel members not acquired during panel period
                   prospect_pools = prospect_pools, #birth time and population size of prospect pools
                   X_ctime = X_ctime, #covariates (quarterly seasonality dummies)
                   left_trunc_time = left_trunc_time, #start time ("month 0") of panel
                   M_company = M_company, #time horizon of analysis
                   parnames=parnames, #names of full parameter vector
                   K_mat = K_mat, #linear transformation matrix
                   diag_covmat = TRUE, #tells objective function not to continuously update covariance matrix
                   covmat_precomputed = MPL_covmat, #feeds in the initialized (inverse) weight matrix
                   iterlim=1000, #maximum iterations for optimization
                   print.level = 2) #tells optimizer to print progress at every iteration
  t2 = Sys.time()
  difftime(t2,t1)
  
  #Save output
  saveRDS(MPL_stage2, "SPOT parameters/MPL_out.RDS")
}else{
  #Read in output if skipping estimation phase
  MPL_stage2 = readRDS("SPOT parameters/MPL_out.RDS")
}

###########################################################################################
########################### Summarizing and visualizing results ###########################
###########################################################################################

############## First, extract parameter estimates and return to original parameterization

#Convert estimated parameters for AGG back to regular parameterization
AGG_par_transformed = rep(0,length(parnames))				# initialize 
omit_inds = unique(c(grep("pi_2",parnames),grep("m_star",parnames),grep("ac2",parnames),grep("sel",parnames)))		# identify ac2-related and selection-related inds
AGG_par_transformed[-omit_inds] = AGG_stage2$estimate #merge in relevant parameter estimates
AGG_par_transformed[grep("pi_2",parnames)]= -10000 #set size of second acquisition segment to close to 0
AGG_par_transformed[grep("sel_b_0",parnames)]= -12 #set probability of selection into panel close to 0

names(AGG_par_transformed) = parnames			# add back names to the parameter vector
AGG_par = inverse_transform_params(AGG_par_transformed, p, M_company, cov_terms=TRUE)		# reparams => params


#Convert estimated parameters for MPL back to regular parameterization
MPL_par_transformed = rep(0,length(parnames))				# initialize 
omit_inds = unique(c(grep("pi_2",parnames),grep("m_star",parnames),grep("ac2",parnames)))		# identify ac2-related inds
MPL_par_transformed[-omit_inds] = MPL_stage2$estimate #merge in relevant parameter estimates
MPL_par_transformed[grep("pi_2",parnames)]= -10000 #set size of second acquisition segment to close to 0

names(MPL_par_transformed) = parnames			# add back names to the parameter vector
MPL_par = inverse_transform_params(MPL_par_transformed, p, M_company, cov_terms=TRUE)		# reparams => params

#Extract acquisition and churn-related parameters as in first part of Table 3 in paper
acqchurn_par = data.frame(Parameter = c("lambda_0", "c", "beta_Q1", "beta_Q2", "beta_Q3", "sigma_lambda", "pi_A"),
                          Initial.Acquisition = MPL_par[c(1:2, 4:6, 30, 3)],
                          Initial.Churn = MPL_par[c(20:24, 39, NA)],
                          Repeat.Acquisition = MPL_par[c(14:15, 17:19, 35, 16)],
                          Repeat.Churn = MPL_par[c(25:29, 44, NA)])
#pi_RA was parameterized in terms of probability of *not* being reacquired; take probability complement
acqchurn_par[acqchurn_par$Parameter == "pi_A", "Repeat.Acquisition"] = 1-acqchurn_par[acqchurn_par$Parameter == "pi_A", "Repeat.Acquisition"]
#change variances to standard deviations
acqchurn_par[acqchurn_par$Parameter == "sigma_lambda",-1] = sqrt(acqchurn_par[acqchurn_par$Parameter == "sigma_lambda",-1])

#round off to 4 sig figs
acqchurn_par = mutate_at(acqchurn_par, vars(Initial.Acquisition:Repeat.Churn), function(x){as.character(signif(x,4))}) 

#extract panel selection bias parameters
sel_par = data.frame(Parameter = c("Intercept", "IA", "IC", "RA", "RC"), 
                     Estimate = MPL_par[c("sel_b_0", "sel_b_ac1", "sel_b_cc", "sel_b_rac", "sel_b_rcc")]) %>%
  mutate_at(vars(Estimate), function(x){as.character(signif(x,4))}) #round off to 4 sig figs

#extract correlations between heterogeneous parameters
cov_mat = matrix(0, nrow = 5, ncol = 5)
cov_mat[upper.tri(cov_mat, TRUE)] = MPL_par[grep("sig", parnames)]
cov_mat = cov_mat + t(cov_mat)
diag(cov_mat) = diag(cov_mat)/2
cor_mat = cov2cor(cov_mat)[-2,-2]

cor_par = data.frame(Parameter = c("rho_IA_IC", "rho_IA_RA", "rho_IA_RC", "rho_IC_RA", "rho_IC_RC", "rho_RA_RC"),
                     Estimate = c(cor_mat[1,3], cor_mat[1,2], cor_mat[1,4], cor_mat[3,2], cor_mat[3,4], cor_mat[2,4])) %>%
  mutate_at(vars(Estimate), function(x){as.character(signif(x,4))}) #round off to 4 sig figs

############## Table 3: MPL parameter estimates
kable(acqchurn_par)
kable(sel_par)
kable(cor_par)
# estimates are numerically a bit different from main paper
# since panel data in this example is synthetic (and 10x smaller than the real panel)

############## Now, obtain predictions from estimated models
# get full matrix of all times for getting what we expect
end_time_ctime_all = seq(3,M_total,by=3)				
start_time_ctime_all = end_time_ctime_all-2
ntimes_all = length(end_time_ctime_all)
disc_mat_ADD_END_LOSS = data.frame(start_mth_ctime=rep(start_time_ctime_all,3), end_mth_ctime=rep(end_time_ctime_all,3), disc=rep(c("ADD","LOSS","END"),each=ntimes_all))
K_mat_ADD_END_LOSS = disc_to_K(disc_mat_ADD_END_LOSS, M_total)
#use this augmented K matrix to get predictions of ADD/END/LOSS for all quarters

# break up K_mat_ADD into ones for QIA and QRA separately (initial vs. repeat acquisitions)
K_mat_QIA = K_mat_ADD_END_LOSS[disc_mat_ADD_END_LOSS$disc == "ADD", ]
K_mat_QIA[,which(substr(colnames(K_mat_QIA),1,3)=="RAC") ]=0 #zero out the weight on repeat behavior
K_mat_QRA = K_mat_ADD_END_LOSS[disc_mat_ADD_END_LOSS$disc == "ADD", ]
K_mat_QRA[,which(substr(colnames(K_mat_QRA),1,3)=="IAC") ]=0 #zero out the weight on initial behavior

disc_mat_all_QIA = filter(disc_mat_ADD_END_LOSS, disc == "ADD") %>%  mutate(disc='QIA')
disc_mat_all_QRA = filter(disc_mat_ADD_END_LOSS, disc == "ADD") %>%  mutate(disc='QRA')


# break up K_mat_LOSS into ones for QIC and QRC separately (initial vs. repeat churns)
K_mat_QIC = K_mat_ADD_END_LOSS[disc_mat_ADD_END_LOSS$disc == "LOSS", ]
K_mat_QIC[,which(substr(colnames(K_mat_QIC),1,3)=="RLC") ]=0 #zero out the weight on repeat behavior
K_mat_QRC = K_mat_ADD_END_LOSS[disc_mat_ADD_END_LOSS$disc == "LOSS", ]
K_mat_QRC[,which(substr(colnames(K_mat_QRC),1,3)=="ILC") ]=0 #zero out the weight on initial behavior

disc_mat_all_QIC = filter(disc_mat_ADD_END_LOSS, disc == "LOSS") %>%  mutate(disc='QIC')
disc_mat_all_QRC = filter(disc_mat_ADD_END_LOSS, disc == "LOSS") %>%  mutate(disc='QRC')


# recombine to form full K_mat for QIA, QIC, QRA, QRC, END
K_mat_all = as.matrix(rbind.data.frame(K_mat_QIA, K_mat_QIC, K_mat_QRA, K_mat_QRC, 
                                       K_mat_ADD_END_LOSS[disc_mat_ADD_END_LOSS$disc == "END", ]))
disc_mat_all = rbind.data.frame(disc_mat_all_QIA, disc_mat_all_QIC, disc_mat_all_QRA, disc_mat_all_QRC, 
                                filter(disc_mat_ADD_END_LOSS, disc == "END"))

# make extended seasonality matrix to cover holdout period
X_ctime_holdout = cbind(													
  Q1 = rep(c(rep(0,3),rep(1,3),rep(0,6)), length = M_total),
  Q2 = rep(c(rep(0,6),rep(1,3),rep(0,3)), length = M_total),
  Q3 = rep(c(rep(0,9),rep(1,3)), length = M_total)
) 

#Get predictions for AGG
AGG_pred_vec = get_expected_disclosures(AGG_par,
                                        sims = sims_draws,
                                        acqchurn_ptime_list = NULL,
                                        panel_inactive_pop = 0,
                                        prospect_pools = prospect_pools,
                                        X_ctime = X_ctime_holdout,
                                        left_trunc_time = left_trunc_time,
                                        M_company = M_total,
                                        K_mat = K_mat_all)

#Get predictions for MPL
MPL_pred_vec = get_expected_disclosures(MPL_par,
                                        sims = sims_draws,
                                        acqchurn_ptime_list = acqchurn_ptime_list,
                                        panel_inactive_pop = panel_inactive_pop,
                                        prospect_pools = prospect_pools,
                                        X_ctime = X_ctime_holdout,
                                        left_trunc_time = left_trunc_time,
                                        M_company = M_total,
                                        K_mat = K_mat_all)

prediction_mat = data.frame(disc_mat_all, AGG_pred_vec, MPL_pred_vec)

#Impute ADD data from END and LOSS data
obs_ADD_END_LOSS = aggregate_data_all %>%
  mutate(value = 1E6 * value) %>%
  spread(disc, value) %>%
  mutate(ADD = END - lag(END) + LOSS) %>%
  gather(key = "disc", value = "value", END:ADD, na.rm = TRUE) %>%
  mutate(Year = ymd("2008-10-01") + duration(end_mth_ctime, "months")) %>%
  select(-(start_mth_ctime:end_mth_ctime)) %>%
  mutate(disc = factor(disc, levels = c("ADD", "LOSS", "END")))

#Compile predictions into in-sample ADD/END/LOSS
fig2_pred_mat = prediction_mat %>%
  filter(end_mth_ctime <= M_company) %>%
  select(-AGG_pred_vec) %>%
  spread(disc, MPL_pred_vec) %>%
  mutate(ADD = QIA + QRA, LOSS = QIC + QRC) %>%
  select(-(QIA:QRC)) %>%
  mutate(Year = ymd("2008-10-01") + duration(end_mth_ctime, "months")) %>%
  select(-(start_mth_ctime:end_mth_ctime)) %>%
  gather("disc", "value", END:LOSS) %>%
  mutate(disc = factor(disc, levels = c("ADD", "LOSS", "END")))

############## Figure 2: Spotify in-sample fit (quarterly ADD, LOSS, and END)
ggplot(fig2_pred_mat, aes(Year, value)) +
  geom_line(lty = 2, lwd = 1, col = "gray30") +
  geom_line(data = obs_ADD_END_LOSS, lwd = 1) +
  facet_wrap(~disc, nrow = 3, scales = "free_y") +
  scale_y_continuous(name = "Customers", labels = number_format(scale = 1E-6, suffix = "M"))


# Get predictions of END in the 18 months before/after time end of time horizon
END_predictions = prediction_mat %>%
  filter(disc == "END", end_mth_ctime >= M_company) %>%
  gather("method", "value", AGG_pred_vec:MPL_pred_vec) %>%
  mutate(method = substr(method, 1, 3))

# Get actual END in the 18 months before end of time horizon
END_actual = aggregate_data_all %>% 
  filter(disc == "END", end_mth_ctime >= M_company - 18) %>%
  mutate(method = "Actual", value = 1E6*value) %>%
  select(start_mth_ctime:disc, method, value)

# Combine into a plot
END_prediction_plot = rbind(END_predictions, END_actual) %>%
  mutate(Year = ymd("2008-10-01") + duration(end_mth_ctime, unit = "months")) %>%
  ggplot(aes(Year, value, group = method, color = method, linetype = method)) +
  geom_vline(aes(xintercept = as_datetime(ymd("2018-09-30"))), lwd = 1, lty = 2, col = "gray70") +
  geom_line(lwd = 1) +
  scale_color_colorblind(name = NULL) +
  scale_linetype_discrete(name = NULL) +
  scale_y_continuous(name = "Total Subscribers", labels = number_format(scale = 1E-6, suffix = "M")) +
  ggtitle("Projections of Total Subscriber Base by Method") +
  theme(plot.title = element_text(hjust = .5))

# get plot comparing prediction of initial vs. repeat acquisition in holdout
acq_prediction_plot = prediction_mat %>%
  filter(disc %in% c("QIA", "QRA"), end_mth_ctime > M_company) %>%
  mutate(quarter = factor(rep(paste0("Q", c(4,1:4,1), "-", c(18, rep(19,4), 20)),2),
                          levels = paste0("Q", c(4,1:4,1), "-", c(18, rep(19,4), 20))),
         disc = factor(ifelse(disc == "QIA", "Initial", "Repeat"), levels = c("Repeat", "Initial"))) %>%
  gather("Method", "value", AGG_pred_vec:MPL_pred_vec) %>%
  mutate(Method = substr(Method, 1, 3)) %>%
  ggplot(aes(Method, value, group = disc, fill = disc)) +
  geom_bar(stat = 'identity', position = 'stack') +
  facet_wrap(~quarter, nrow = 1) +
  scale_fill_economist(name = "Acquisition\nForm") +
  scale_y_continuous(name = "Subscriber Acquisitions", labels = number_format(scale = 1E-6, suffix = "M")) +
  ggtitle("Forecasted Acquisitions by Method, Initial Versus Repeat, Q4 2018 - Q1 2020") +
  theme(plot.title = element_text(hjust = .5))

# get plot comparing prediction of initial vs. repeat churn in holdout
loss_prediction_plot = prediction_mat %>%
  filter(disc %in% c("QIC", "QRC"), end_mth_ctime > M_company) %>%
  mutate(quarter = factor(rep(paste0("Q", c(4,1:4,1), "-", c(18, rep(19,4), 20)),2),
                          levels = paste0("Q", c(4,1:4,1), "-", c(18, rep(19,4), 20))),
         disc = factor(ifelse(disc == "QIC", "Initial", "Repeat"), levels = c("Repeat", "Initial"))) %>%
  gather("Method", "value", AGG_pred_vec:MPL_pred_vec) %>%
  mutate(Method = substr(Method, 1, 3)) %>%
  ggplot(aes(Method, value, group = disc, fill = disc)) +
  geom_bar(stat = 'identity', position = 'stack') +
  facet_wrap(~quarter, nrow = 1) +
  scale_fill_economist(name = "Churn Form") +
  scale_y_continuous(name = "Subscriber Churn", labels = number_format(scale = 1E-6, suffix = "M")) + 
  ggtitle("Forecasted Churn by Method, Initial Versus Repeat, Q4 2018 - Q1 2020") +
  theme(plot.title = element_text(hjust = .5))

############## Figure 4: Spotify holdout prediction comparison
grid.arrange(END_prediction_plot,			  
             acq_prediction_plot,
             loss_prediction_plot,
             nrow=3,
             heights=rep(7,3))
