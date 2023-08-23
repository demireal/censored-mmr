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
parser.add_argument('--config_json', required=True, help='Path to the JSON file containing experiment configuration')
args = parser.parse_args()

with open(args.config_json, 'r') as file:
    try:
        jD = json.load(file)
    except json.JSONDecodeError:
        print("Invalid JSON format in the input file.")

assert jD['cov_dim'] == len(jD['RCT']['px_args']['mean']), "Check covariate dimensions."

mmr_results = np.zeros((jD['maxm'], len(jD['test_signals']), jD['num_exp']))
mmr_pvals = np.zeros((jD['maxm'], len(jD['test_signals']), jD['num_exp']))

m_cols = ['m = ' + str(m + 1) for m in range(jD['maxm'])]
mmr_results_df = pd.DataFrame(columns=['Test'] + m_cols, index=range(len(jD['test_signals'])))

for mind, m in enumerate(list(np.arange(1, jD['maxm'] + 1))):
    start_time = time()
    obs_size = jD['rct_size'] * m 
    
    local_mmr_results = Parallel(n_jobs=int(cpu_count()))(
                            delayed(single_mmr_run)(
       jD['test_signals'], jD['save_df'], jD['cov_dim'], jD['rct_size'], obs_size, jD['B'], laplacian_kernel,
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
    print(f'm = {m}, time passed = {exec_time:.1f}')


for kind, key in enumerate(jD['test_signals']):
    for mind, m_col in enumerate(m_cols):
        mmr_results_df.loc[kind, 'Test'] = key
        mmr_results_df.loc[kind, m_col] = mmr_results[mind, kind, :].mean()
  

# Saving the results

save_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f'/results/{jD["res_subdir"]}/cov_dim_{jD["cov_dim"]}')
save_dir.mkdir(parents=True, exist_ok=True)
        
mmr_results_df.to_csv(os.path.join(save_dir, 'res_summary.csv'), index=False)    
np.save(os.path.join(save_dir, 'pvals.npy'), mmr_pvals) 

with open(os.path.join(save_dir, 'config.json'), 'w') as output_file:
    json.dump(jD, output_file, indent=4)  # The `indent` parameter adds pretty-printing with indentation