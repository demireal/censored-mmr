import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


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
    return np.exp(-((args['lambda'] * T) ** args['p']) * np.exp(x @ args['beta']))


def coxph_base_surv(df, flip=False):
    '''
    Fit a CoxPH model.

    @params:
        df: Data (pd.DataFrame)
        flip: whether to flip the censoring indicator (bool). E.g., "True" if fitting the model on censoring variable

    @returns:
        t_event: event/censoring times
        est_base_surv: corresponding (to t_event) baseline survival probabilities
        cph.params_: estimated hazards (beta)
    '''
    if flip:  # to fit the model for the censoring variable
        df['Delta'] = 1 - df['Delta']
    cph = CoxPHFitter()
    cph.fit(df, duration_col='T', event_col='Delta')
    
    est_base_surv = cph.baseline_survival_
    t_event = np.array(est_base_surv.index)

    return t_event, est_base_surv, np.array(list(cph.params_).insert(0,0))  


def get_ate(df, psi, sig_name):
    '''
    estimate ATE from an instance-wise signal
    '''
    for i in range(len(df)):
        row = df.loc[i]
        df.loc[i, sig_name] = psi(row)

    return np.mean(df[sig_name]), df


# def ipw_psi_est(row, px_model, sc_time, sc_val, cov_list):
#     if row['Delta'] == 1:
#         px = px_model.predict_proba([np.array(row[cov_list])])[0][1]
#         sc = np.array(sc_val)[np.abs(sc_time - row['T']).argmin()]

#         part1 = int(row['A'] == 1) / px
#         part2 = int(row['A'] == 0) / (1 - px)

#         ipw_psi = row['T'] * (part1 - part2) / sc
#     else:
#         ipw_psi = 0

#     return ipw_psi