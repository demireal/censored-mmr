#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from tqdm import tqdm
from time import sleep, time

import warnings
warnings.filterwarnings("ignore")

from SyntheticDataModule import *
from estimators import *
from utils import *
from pathlib import Path
import os

from joblib import Parallel, delayed
from multiprocessing import cpu_count


def single_mmr_run(test_signals, save_df, d, rct_size, obs_size,
                   px_dist_r, px_args_r, prop_fn_r, prop_args_r, tte_params_r,
                   px_dist_o, px_args_o, prop_fn_o, prop_args_o, tte_params_o):
    
    RCTData = SyntheticDataModule(save_df, d, rct_size, 0, px_dist_r, px_args_r, prop_fn_r, prop_args_r, tte_params_r)
    OBSData = SyntheticDataModule(save_df, d, obs_size, 1, px_dist_o, px_args_o, prop_fn_o, prop_args_o, tte_params_o)

    df_rct_oracle, df_rct = RCTData.get_df()
    df_obs_oracle, df_obs = OBSData.get_df()

    df_combined = pd.concat([df_rct, df_obs], axis=0, ignore_index=True)  # merge the dataframes into one
    cov_list = RCTData.get_covs()

    # Estimate the nuisance parameters

    df_combined['P(S=1|X)'] = prop_score_est(df_combined.copy(), 'S', cov_list, 'logistic')

    df_combined.loc[df_combined.S==0, 'P(A=1|X,S)'] = prop_score_est(df_combined.query('S==0').copy(), 'A', cov_list, 'logistic')
    df_combined.loc[df_combined.S==1, 'P(A=1|X,S)'] = prop_score_est(df_combined.query('S==1').copy(), 'A', cov_list, 'logistic')

    gc_est(df_combined, cov_list, tte_model='coxph')

    ipcw_est(df_combined, S=0)
    ipcw_est(df_combined, S=1)
    ipw_impute_est(df_combined, S=0)
    ipw_impute_est(df_combined, S=1)
    
    mmr_stats = np.zeros((len(test_signals), 2))  # store results and p-val for each mmr test

    for kind, key in enumerate(test_signals):
        signal0, signal1 = test_signals[key][0], test_signals[key][1]
        mmr_stats[kind, 0], mmr_stats[kind, 1] = mmr_test(df_combined, cov_list, B, laplacian_kernel, signal0, signal1)
        
    return mmr_stats


sce = 1  # scenario index number. only used for saving the results at the very end
save_df = False  # do not save the dataframes generated during experiments

d = 1  # covariate dimensions
rct_size = 1000 
m_arr = [1, 2, 3, 4, 5]  # obs_size = rct_size * m for m in m_arr

B = 100  # num. samples to model the null distribution for a single hypothesis test
num_exp = 40  # number of repetitions for each experimental setup


# RCT data generating model parameters

px_dist_r, px_args_r = 'Gaussian', {'mean': [0], 'cov': [[1]]}
prop_fn_r, prop_args_r = 'sigmoid', {'beta': [0, 1e-4]}
tte_params_r = {'model': 'coxph',
                'hazard': 'weibull',
                'cox_args': {'Y0': {'beta': [0,0.75], 'lambda': 0.5, 'p': 5},
                            'Y1': {'beta': [0,0.25], 'lambda': 0.3, 'p': 5},
                            'C0': {'beta': [0,0], 'lambda': 0.2, 'p': 4},
                            'C1': {'beta': [0,0], 'lambda': 0.15, 'p': 4},},
                }


# OBS data generating model parameters

px_dist_o, px_args_o = 'Gaussian', {'mean': [-0.5], 'cov': [[1.5]]}
prop_fn_o, prop_args_o = 'sigmoid', {'beta': [0.8, 0.25]}
tte_params_o = {'model': 'coxph',
                'hazard': 'weibull',
                'cox_args': {'Y0': {'beta': [0,0.75], 'lambda': 0.5, 'p': 5},
                            'Y1': {'beta': [0,0.25], 'lambda': 0.3, 'p': 5},
                            'C0': {'beta': [0,0], 'lambda': 0.2, 'p': 1.5},
                            'C1': {'beta': [0,0], 'lambda': 0.2, 'p': 1.5},},
                }


test_signals = {'IPCW-Contrast': ['S0_ipcw_est_CATE', 'S1_ipcw_est_CATE'],
                'IPCW-Y1': ['S0_ipcw_est_Y1', 'S1_ipcw_est_Y1'],
                'IPCW-Y0': ['S0_ipcw_est_Y0', 'S1_ipcw_est_Y0'],
                'Impute-IPW-Contrast': ['S0_impute_ipw_est_Y0', 'S1_impute_ipw_est_Y0'],
                'Impute-IPW-Y1': ['S0_impute_ipw_est_Y0', 'S1_impute_ipw_est_Y0'],
                'Impute-IPW-Y0': ['S0_impute_ipw_est_Y0', 'S1_impute_ipw_est_Y0'],}


mmr_results = np.zeros((len(m_arr), len(test_signals), num_exp))
mmr_pvals = np.zeros((len(m_arr), len(test_signals), num_exp))

m_cols = ['m = ' + str(m) for m in m_arr]
mmr_results_df = pd.DataFrame(columns=['Test'] + m_cols, index=range(len(test_signals)))


for mind, m in enumerate(m_arr):
    start_time = time()
    obs_size = rct_size * m 
    
    local_mmr_results = Parallel(n_jobs=int(cpu_count()))(
                            delayed(single_mmr_run)(test_signals, save_df, d, rct_size, obs_size,
                                               px_dist_r, px_args_r, prop_fn_r, prop_args_r, tte_params_r,
                                               px_dist_o, px_args_o, prop_fn_o, prop_args_o, tte_params_o)
                            for nind in range(num_exp)
                        )
    
    for nind in range(num_exp):
        for kind in range(len(test_signals)):
            mmr_results[mind, kind, nind] = local_mmr_results[nind][kind][0]
            mmr_pvals[mind, kind, nind] = local_mmr_results[nind][kind][1]
            
    exec_time =  time() - start_time
    print(f'Scenario: {sce}, m = {m}, time passed = {exec_time:.1f}')



for kind, key in enumerate(test_signals):
    for mind, m_col in enumerate(m_cols):
        mmr_results_df.loc[kind, 'Test'] = key
        mmr_results_df.loc[kind, m_col] = mmr_results[mind, kind, :].mean()
        
res_subdir = Path('./results')
res_newdir = f'sce{sce}'

res_findir = res_subdir / res_newdir
res_findir.mkdir(parents=True, exist_ok=True)
        
mmr_results_df.to_csv(os.path.join(res_subdir, res_newdir, 'res_summary.csv'), index=False)    
np.save(os.path.join(res_subdir, res_newdir, 'pvals.npy'), mmr_pvals)        







