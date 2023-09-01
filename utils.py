import numpy as np
import pandas as pd
import json


def sigmoid_fn(X, beta):
    return 1/(1 + np.exp(- X @ beta))


def sample_weibull_tte(n, args, x):
    '''
    Sample Weibull time-to-event outcomes.
    see: https://web.stanford.edu/~lutian/coursepdf/unit1.pdf
        h(t | x) = p * (lambda_ ** p) * (t ** (p-1)) * exp(beta*x)
        S(t | x) = exp(-((lambda_ * t) ** p) * exp(beta*x))

    @params:
        n: number of samples (integer)
        x: covariates (list)
        args: Cox PH model parameters (dict)
    '''

    C = np.exp(x @ args['beta'])
    U = np.random.uniform(0, 1, n)
    return (-np.log(U) / C) ** (1 / args['p']) * (1 / args['lambda'])


def weibull_oracle_adj_surv(T, x, args):
    '''
    see: https://web.stanford.edu/~lutian/coursepdf/unit1.pdf
        S(t | x) = exp(-((lambda_ * t) ** p) * exp(beta*x))
    '''

    return np.exp(-((args['lambda'] * T) ** args['p']) * np.exp(x @ args['beta']))


def read_json(json_path, d, uc, sigs_to_keep):
    with open('exp_configs/' + json_path, 'r') as file:
        try:
            jD = json.load(file)
        except json.JSONDecodeError:
            print("Invalid JSON format in the input file.")

    assert jD['num_cov'] == len(jD['RCT']['px_args']['mean']) \
                         == len(jD['RCT']['px_args']['cov']) \
                         == len(jD['RCT']['prop_args']['beta']) - 1\
                         == len(jD['RCT']['tte_params']['cox_args']['Y0']['beta']) - 1 \
                         == len(jD['RCT']['tte_params']['cox_args']['Y1']['beta']) - 1 \
                         == len(jD['RCT']['tte_params']['cox_args']['C0']['beta']) - 1 \
                         == len(jD['RCT']['tte_params']['cox_args']['C1']['beta']) - 1 \
                         == len(jD['OS']['px_args']['mean']) \
                         == len(jD['OS']['px_args']['cov']) \
                         == len(jD['OS']['prop_args']['beta']) - 1 \
                         == len(jD['OS']['tte_params']['cox_args']['Y0']['beta']) - 1 \
                         == len(jD['OS']['tte_params']['cox_args']['Y1']['beta']) - 1 \
                         == len(jD['OS']['tte_params']['cox_args']['C0']['beta']) - 1 \
                         == len(jD['OS']['tte_params']['cox_args']['C1']['beta']) - 1 \
                    , "Check covariate dimensions."
    
    jD_new = jD.copy()
    
    jD_new['RCT']['px_args']['mean'] = jD['RCT']['px_args']['mean'][:d]
    jD_new['RCT']['px_args']['cov'] = np.array(jD['RCT']['px_args']['cov'])[:d, :d]
    jD_new['RCT']['prop_args']['beta'] = jD['RCT']['prop_args']['beta'][:d+1]
    jD_new['RCT']['tte_params']['cox_args']['Y0']['beta'] = jD['RCT']['tte_params']['cox_args']['Y0']['beta'][:d+1]
    jD_new['RCT']['tte_params']['cox_args']['Y1']['beta'] = jD['RCT']['tte_params']['cox_args']['Y1']['beta'][:d+1]
    jD_new['RCT']['tte_params']['cox_args']['C0']['beta'] = jD['RCT']['tte_params']['cox_args']['C0']['beta'][:d+1]
    jD_new['RCT']['tte_params']['cox_args']['C1']['beta'] = jD['RCT']['tte_params']['cox_args']['C1']['beta'][:d+1]
    
    jD_new['OS']['px_args']['mean'] = jD['OS']['px_args']['mean'][:d]
    jD_new['OS']['px_args']['cov'] = np.array(jD['OS']['px_args']['cov'])[:d, :d]
    jD_new['OS']['prop_args']['beta'] = jD['OS']['prop_args']['beta'][:d+1]
    jD_new['OS']['tte_params']['cox_args']['Y0']['beta'] = jD['OS']['tte_params']['cox_args']['Y0']['beta'][:d+1]
    jD_new['OS']['tte_params']['cox_args']['Y1']['beta'] = jD['OS']['tte_params']['cox_args']['Y1']['beta'][:d+1]
    jD_new['OS']['tte_params']['cox_args']['C0']['beta'] = jD['OS']['tte_params']['cox_args']['C0']['beta'][:d+1]
    jD_new['OS']['tte_params']['cox_args']['C1']['beta'] = jD['OS']['tte_params']['cox_args']['C1']['beta'][:d+1]
    
    jD_new['cov_list'] = [f'X{i}' for i in range(d - uc + 1)]
    jD_new['test_signals'] = {key: value for key, value in jD['test_signals'].items() if key in sigs_to_keep}
    
    
    
    return jD_new
    