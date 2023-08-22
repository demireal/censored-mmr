#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel

import os
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from pathlib import Path
import argparse
from tqdm import tqdm
from time import sleep, time

from SyntheticDataModule import *
from estimators import *
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--save_df', default=False, help='Whether to save the dataframes generated during experiments')
parser.add_argument('--res_subdir', default='', help='e.g. cic_same-po_nuc')

parser.add_argument('--dim', type=int, default=1, help='Covariate dimensionality')
parser.add_argument('--rct_size', type=int, default=1000, help='Trial cohort size')
parser.add_argument('--maxm', type=int, default=5, help='obs_size = rct_size * m for m in [1,2, ..., maxm]')
parser.add_argument('--B', type=int, default=100, help='Num. samples to model the null H0')
parser.add_argument('--num_exp', type=int, default=40, help='Number of repetitions for each experimental setup')

args = parser.parse_args()

print(f'Cov. dim.: {args.dim}')
print(f'RCT size: {args.rct_size}')
print(f'Max. m: {args.maxm}')
print(f'B: {args.B}')
print(f'Num exp: {args.num_exp}')


# RCT data generating model parameters

px_dist_r, px_args_r = 'Gaussian', {'mean': [0], 'cov': [[1]]}
prop_fn_r, prop_args_r = 'sigmoid', {'beta': [0, 1e-4]}
tte_params_r = {'model': 'coxph',
                'hazard': 'weibull',
                'cox_args': {'Y0': {'beta': [0,0.75], 'lambda': 0.5, 'p': 5},
                            'Y1': {'beta': [0,0.25], 'lambda': 0.15, 'p': 5},
                            'C0': {'beta': [0,0], 'lambda': 0.2, 'p': 4},
                            'C1': {'beta': [0,0], 'lambda': 0.1, 'p': 4},},
                }

# OBS data generating model parameters

px_dist_o, px_args_o = 'Gaussian', {'mean': [-0.5], 'cov': [[1.5]]}
prop_fn_o, prop_args_o = 'sigmoid', {'beta': [0.8, 0.25]}
tte_params_o = {'model': 'coxph',
                'hazard': 'weibull',
                'cox_args': {'Y0': {'beta': [0,0.75], 'lambda': 0.5, 'p': 5},
                            'Y1': {'beta': [0,0.25], 'lambda': 0.15, 'p': 5},
                            'C0': {'beta': [0,0], 'lambda': 0.2, 'p': 4},
                            'C1': {'beta': [0,0], 'lambda': 0.2, 'p': 1.5},},
                }

assert args.dim == len(px_args_o['mean']), "Check covariate dimensions."

# Signals to test for equivalence in the MMR test

test_signals = {'IPCW-Contrast': ['S0_ipcw_est_CATE', 'S1_ipcw_est_CATE'],
                'IPCW-Y1': ['S0_ipcw_est_Y1', 'S1_ipcw_est_Y1'],
                'IPCW-Y0': ['S0_ipcw_est_Y0', 'S1_ipcw_est_Y0'],
                'Impute-IPW-Contrast': ['S0_impute_ipw_est_CATE', 'S1_impute_ipw_est_CATE'],
                'Impute-IPW-Y1': ['S0_impute_ipw_est_Y1', 'S1_impute_ipw_est_Y1'],
                'Impute-IPW-Y0': ['S0_impute_ipw_est_Y0', 'S1_impute_ipw_est_Y0'],
                'Drop-IPW-Contrast': ['S0_drop_ipw_est_CATE', 'S1_drop_ipw_est_CATE'],
                'Drop-IPW-Y1': ['S0_drop_ipw_est_Y1', 'S1_drop_ipw_est_Y1'],
                'Drop-IPW-Y0': ['S0_drop_ipw_est_Y0', 'S1_drop_ipw_est_Y0'],
               }


mmr_results = np.zeros((args.maxm, len(test_signals), args.num_exp))
mmr_pvals = np.zeros((args.maxm, len(test_signals), args.num_exp))

m_cols = ['m = ' + str(m+1) for m in range(args.maxm)]
mmr_results_df = pd.DataFrame(columns=['Test'] + m_cols, index=range(len(test_signals)))


for mind, m in enumerate(list(np.arange(1, args.maxm+1))):
    start_time = time()
    obs_size = args.rct_size * m 
    
    local_mmr_results = Parallel(n_jobs=int(cpu_count()))(
                    delayed(single_mmr_run)(test_signals, args.save_df,
                                       args.dim, args.rct_size, obs_size, args.B, laplacian_kernel,
                                       px_dist_r, px_args_r, prop_fn_r, prop_args_r, tte_params_r,
                                       px_dist_o, px_args_o, prop_fn_o, prop_args_o, tte_params_o)
                    for nind in range(args.num_exp)
                )
    
    for nind in range(args.num_exp):
        for kind in range(len(test_signals)):
            mmr_results[mind, kind, nind] = local_mmr_results[nind][kind][0]
            mmr_pvals[mind, kind, nind] = local_mmr_results[nind][kind][1]
            
    exec_time =  time() - start_time
    print(f'm = {m}, time passed = {exec_time:.1f}')


for kind, key in enumerate(test_signals):
    for mind, m_col in enumerate(m_cols):
        mmr_results_df.loc[kind, 'Test'] = key
        mmr_results_df.loc[kind, m_col] = mmr_results[mind, kind, :].mean()
  

# Saving the results

script_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f'/results/{args.res_subdir}')
save_findir = script_dir / f'cov_dim_{args.dim}'
save_findir.mkdir(parents=True, exist_ok=True)
        
mmr_results_df.to_csv(os.path.join(script_dir, f'cov_dim_{args.dim}', 'res_summary.csv'), index=False)    
np.save(os.path.join(script_dir, f'cov_dim_{args.dim}', 'pvals.npy'), mmr_pvals)     
readme_summary(os.path.join(script_dir, f'cov_dim_{args.dim}', 'README.txt'), args, 
              px_dist_r, px_args_r, prop_fn_r, prop_args_r, tte_params_r,
              px_dist_o, px_args_o, prop_fn_o, prop_args_o, tte_params_o,)



        
        











