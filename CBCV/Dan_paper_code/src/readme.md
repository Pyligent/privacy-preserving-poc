For R code


-----
NOTES
-----

This file has all notable implementation details associated with the running of the reproducibility code included with this submission, associated with the paper "Using Aggregate-Disaggregate Data Fusion to Forecast the Inflow and Outflow of Customers". All code was written in the R programming language.

The reproducibility code consists of two R scripts, 'simulation code.R' and 'SPOT sample code.R'. The former is used to demonstrate how the key results from the simulation study (Section 4 of the paper, and supplementary results in Tables 1, 2, 3, and 4 in the online supplement, Section C) can be obtained. The latter is used to demonstrate how the key results from the empirical application of the methodology to data from Spotify (Section 5 of the paper) can be obtained. 

These scripts can be run on a standalone basis, and do not assume that the user has run the other script. Both scripts use functions contained in an R package created for the paper, MPL. There is code at the beginning of both R scripts to install this package from source for self-sufficiency, but the package need only be installed once.

Both R scripts assume that the working directory points to the folder that contains this README file.

Below we provide comments about each of the aforementioned R scripts in turn.




-----------------
simulation code.R
-----------------

There are a total of 2,816 scenarios that we run in the manuscript, consisting of all possible combinations of data settings and parameter settings considered in the paper. Running the estimation procedure for all 2,816 scenarios takes a considerable amount of time (on AWS, we spent over $18,000 to complete the scenarios). For expedience, we default to running two of the scenarios associated with one particular data setting: N = 20K. In this way, users can replicate two of the scenarios associated with N=20K data point from Figure 1 (i.e., the left-most points in the bottom-left plot of the 2 x 2 plots for Initial Acquisition and Repeat Churn). However, we provide a toggle (R object 'do_abridged_code') that allows users to run all of the simulations if they would like to replicate all of the results, including Figure 1 itself. We also make it very easy to run all of the parameter settings associated with the N = 20K data setting.

Running the code creates a folder in the working directory called 'simulation outputs'. This folder contains outputs corresponding to tables and figures in the paper and the online supplement. 

When 'do_abridged_code = FALSE', the PDF shown in Figure 1 in the paper is generated in the 'simulation outputs' folder, as well as the corresponding PDF for the two disclosures that are not shown in Figure 1. 

An RDS file is saved which contains the results shown in Table 1 in the paper and Tables 1, 2, 3, and 4 in the online supplement. The results are outputted in latex format. Minimal processing is applied to these latex figures to obtain the tables shown in the paper and the online supplement. 




------------------
SPOT sample code.R
------------------

This sample code file demonstrates how we implement our model on the empirical example of Spotify; in particular, it demonstrates estimation of the model using only aggregate data ("AGG") and using the proposed combination of aggregate and panel data ("MPL"). In the "Data" subdirectory we include the full aggregate data for Spotify, but for confidentiality reasons we cannot share the panel data; instead, we provide synthetic panel data simulated from the estimated model, which is about 10% of the size of the original panel.

To allow the computation to complete in a reasonable amount of time, we include the first-stage estimates for AGG and MPL in the "SPOT parameters" subdirectory, such that the script simply loads these parameter values in and uses them to initialize the second phase of estimation (both AGG and MPL are two-stage estimators). This can still take some time (1-2 days running locally on a single processor), so we also include a toggle (R object 'do_estimation') which allows for the script to skip the estimation phase and just produce outputs.

Given estimates, the script demonstrates how to tabulate the parameter estimates (similar to Table 3 in the paper) and how to create Figures 2 and 4 in the paper. Because the panel data is simulated and only 10% of the size of the actual panel, these results will be different from those presented in the tables/figures in the paper; differences across machines in numerical precision, R defaults, etc. may also cause further discrepancies. However, the results should be directionally consistent and qualitatively similar to those presented in the paper.
