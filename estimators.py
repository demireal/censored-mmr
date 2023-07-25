import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter

from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel


def prop_score_est(df, target, feature, model_name):
    '''
    Train a propensity score model, e.g., P(S=1 | X) or P(A=1 | X, S=0), and return its predictions.

    @params:
        df: Data to learn the model from (pd.DataFrame)
        target: target variable, e.g. A, S, (string)
        feature: regressor features (list of strings)
        model_name: model to use, e.g., logistic regression (string)

    @return:
        estimated propensity scores
    '''

    X = df[feature]
    y = df[target]

    if model_name == 'logistic':
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(disp=0)
        probs = result.predict(X)

    elif model_name == 'mean':
        probs = y.mean() * np.ones(len(y))

    else:
        raise NotImplementedError(f'{model_name} is not implemented for propensity score estimation')
    
    return probs


def coxph_base_surv(df, cov_list, flip=False):
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

    crop_df = df[cov_list + ['T', 'Delta']].copy()
    if flip:  # to fit the model for the censoring variable
        crop_df['Delta'] = 1 - crop_df['Delta']
    cph = CoxPHFitter()
    cph.fit(crop_df, duration_col='T', event_col='Delta')
    
    est_base_surv = cph.baseline_survival_
    t_event = np.array(est_base_surv.index)

    return t_event, np.array(est_base_surv).reshape(-1), np.array(list(cph.params_).insert(0,0))  
    

def gc_est(df, cov_list, tte_model):
    '''
    Calculate the adjusted survival curve for the censoring variable ("G_C(T | X,S,A)") separately for S=0,1 and A=0,1 (4 models total)
    Record the results in the dataframe.

    @params:
        df: Data (pd.DataFrame)
        cov_list: list of covariates except from S and A (string)
        tte_model: generative model for the outcome (censoring variable) to be fit
    '''

    cbse = {}  # censoring-baseline-survival-estimate (cbse) 

    if tte_model == 'coxph':
        cbse['t_S0_C0'], cbse['St_S0_C0'], _ = coxph_base_surv(df.query('S==0 & A==0').copy(), cov_list[1:], flip=True)
        cbse['t_S0_C1'], cbse['St_S0_C1'], _ = coxph_base_surv(df.query('S==0 & A==1').copy(), cov_list[1:], flip=True)
        cbse['t_S1_C0'], cbse['St_S1_C0'], _ = coxph_base_surv(df.query('S==1 & A==0').copy(), cov_list[1:], flip=True)
        cbse['t_S1_C1'], cbse['St_S1_C1'], _ = coxph_base_surv(df.query('S==1 & A==1').copy(), cov_list[1:], flip=True)

        for i in range(len(df)):
            t_str = 't_S{}_C{}'.format(df.loc[i, 'S'], df.loc[i, 'A'])
            st_str = 'St_S{}_C{}'.format(df.loc[i, 'S'], df.loc[i, 'A'])
            obs_val = df.loc[i, 'T']
            df.loc[i, 'G_C(T|X,S,A)'] = cbse[st_str][np.argmin(np.abs(obs_val-cbse[t_str]))]

    else:
        raise NotImplementedError(f'Time-to-event model <{tte_model}> is not implemented.')


def ipcw_est(df, S):
    '''
    Calculate the instance-wise inverse propensity weighted signal for CATE, using the *combined* dataframe.
    Record the IPCW-signals in the dataframe.

    @params:
        df: Data (pd.DataFrame)
        S: study index (integer)
    '''

    for i in range(len(df)):
        row = df.loc[i]

        if row['Delta'] == 1 and row['S'] == S:
            part1 = row['A'] / (row['P(A=1|X,S)'])
            part0 = (1 - row['A']) / (1 - row['P(A=1|X,S)'])

            psx = row['S'] * row['P(S=1|X)'] + (1 - row['S']) * (1 - row['P(S=1|X)'])
            denom = psx * row['G_C(T|X,S,A)']

            ipcw = row['T'] * (part1 - part0) / denom

        else:
            ipcw = 0

        df.loc[i, f'S{S}_ipcw_est_CATE'] = ipcw
        df.loc[i, f'S{S}_ipcw_est_Y1'] = row['A'] * ipcw
        df.loc[i, f'S{S}_ipcw_est_Y0'] = - (1 - row['A']) * ipcw


def ipw_impute_est(df, S):
    '''
    Calculate the instance-wise inverse propensity weighted signal for CATE, using the *combined* dataframe.
    Record the IPW-signals in the dataframe.
    Note that we impute the censored values.

    @params:
        df: Data (pd.DataFrame)
        S: study index (integer)
    '''

    for i in range(len(df)):
        row = df.loc[i]

        if row['S'] == S:
            part1 = row['A'] / (row['P(A=1|X,S)'])
            part0 = (1 - row['A']) / (1 - row['P(A=1|X,S)'])

            psx = row['S'] * row['P(S=1|X)'] + (1 - row['S']) * (1 - row['P(S=1|X)'])
            ipw = row['T'] * (part1 - part0) / psx

        else:
            ipw = 0

        df.loc[i, f'S{S}_impute_ipw_est_CATE'] = ipw
        df.loc[i, f'S{S}_impute_ipw_est_Y1'] = row['A'] * ipw
        df.loc[i, f'S{S}_impute_ipw_est_Y0'] = - (1 - row['A']) * ipw


def mmr_test(df, cov_list, B=100, kernel=rbf_kernel, signal0='S0_ipcw_est_CATE', signal1='S1_ipcw_est_CATE'):
    n = len(df)
    Kxx = kernel(df[cov_list])
    np.fill_diagonal(Kxx, 0)
    psi = np.array(df[signal1] - df[signal0])

    # calculate the MMR test statistic n * M^2_n
    mmr_stat = psi @ Kxx @ psi / (n - 1)

    # obtain a sample from the null distribution n * M^2_{n(k)}
    h0_sample = np.zeros(B)
    for k in range(B):
        wpsi = psi * (np.random.multinomial(n, [1 / n] * n) - 1) 
        wprod = wpsi @ Kxx @ wpsi / n
        h0_sample[k] = wprod

    # return 0 for accepting and 1 for rejecting the null H0.
    pval = (np.sum(mmr_stat < h0_sample) + 1) / (len(h0_sample) + 1)
    return int(pval < 0.05), pval