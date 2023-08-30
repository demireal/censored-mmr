#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel

import json
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
parser.add_argument('--json_path', required=True, help='Path to the JSON file containing experiment configuration')
parser.add_argument('--CD', nargs='+', type=int, help='List of covariate dimension counts')
parser.add_argument('--UC', nargs='+', type=int, help='List of unmeasured confounder counts')
parser.add_argument('--M', nargs='+', type=int, help='List of multiplying factors for observational sample size')
args = parser.parse_args()

assert min(args.UC) >=0, 'Number of unmeasured confounders cannot be negative'
assert min(args.CD) >=0, 'Number of covariates cannot be negative'
assert max(args.UC) <= min(args.CD), 'Number of unmeasured confounders cannot exceed the number of covariates'
 
for cov_dim in args.CD:
    for unmeas_conf in args.UC:
        jD = read_json(args.json_path, cov_dim, unmeas_conf)

        mmr_results = np.zeros((len(args.M), len(jD['test_signals']), jD['num_exp']))
        mmr_pvals = np.zeros((len(args.M), len(jD['test_signals']), jD['num_exp']))

        m_cols = ['m = ' + str(m) for m in args.M]
        mmr_results_df = pd.DataFrame(columns=['Test'] + m_cols, index=range(len(jD['test_signals'])))

        for mind, m in enumerate(args.M):
            start_time = time()
            os_size = jD['rct_size'] * m 

            local_mmr_results = Parallel(n_jobs=int(cpu_count()))(
                                    delayed(single_mmr_run)(
           jD['test_signals'], jD['save_df'], cov_dim, jD['rct_size'], os_size,
           jD['B'], laplacian_kernel, jD['cov_list'], jD['crop_prop'],
           jD['RCT']['px_dist'], jD['RCT']['px_args'], jD['RCT']['prop_fn'], jD['RCT']['prop_args'], jD['RCT']['tte_params'],
           jD['OS']['px_dist'], jD['OS']['px_args'], jD['OS']['prop_fn'], jD['OS']['prop_args'], jD['OS']['tte_params'],
               )
                for nind in range(jD['num_exp'])
            )

            for nind in range(jD['num_exp']):
                for kind in range(len(jD['test_signals'])):
                    mmr_results[mind, kind, nind] = local_mmr_results[nind][kind][0]
                    mmr_pvals[mind, kind, nind] = local_mmr_results[nind][kind][1]

            exec_time =  time() - start_time
            print(f'CD: {cov_dim}, UC: {unmeas_conf}, m: {m}, time elapsed: {exec_time:.1f}')


        for kind, key in enumerate(jD['test_signals']):
            for mind, m_col in enumerate(m_cols):
                mmr_results_df.loc[kind, 'Test'] = key
                mmr_results_df.loc[kind, m_col] = mmr_results[mind, kind, :].mean()


        # Saving the results

        save_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f'/results/{args.json_path[:-5]}/CD-{cov_dim}/UC-{unmeas_conf}')
        save_dir.mkdir(parents=True, exist_ok=True)

        mmr_results_df.to_csv(os.path.join(save_dir, 'res_summary.csv'), index=False)    
        np.save(os.path.join(save_dir, 'pvals.npy'), mmr_pvals) 