import numpy as np
import pandas as pd
import statsmodels.api as sm
from time import sleep, time
from lifelines import CoxPHFitter
from SyntheticDataModule import *

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
    cph = CoxPHFitter(penalizer=0.1)
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
        
    for sind in range(2):
        for aind in range(2):
            
            if len(df.query(f'S=={sind} & A=={aind} & Delta==0')) == 0:
                cbse[f't_S{sind}_C{aind}'], cbse[f'St_S{sind}_C{aind}'] = [-1], [1]
                
            else:
                if tte_model == 'coxph':
                    cbse[f't_S{sind}_C{aind}'], cbse[f'St_S{sind}_C{aind}'], _ = \
                    coxph_base_surv(df.query(f'S=={sind} & A=={aind}').copy(), cov_list[1:], flip=True)  
                    
                else:
                    raise NotImplementedError(f'Time-to-event model <{tte_model}> is not implemented.')
            
    for i in range(len(df)):
        t_str = 't_S{}_C{}'.format(df.loc[i, 'S'], df.loc[i, 'A'])
        st_str = 'St_S{}_C{}'.format(df.loc[i, 'S'], df.loc[i, 'A'])
        obs_val = df.loc[i, 'T']

        if list(cbse[t_str]) == [-1]:
            df.loc[i, 'G_C(T|X,S,A)'] = 1
        else:
            df.loc[i, 'G_C(T|X,S,A)'] = cbse[st_str][np.argmin(np.abs(obs_val-cbse[t_str]))]
            
    return cbse



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


def ipw_est(df, S, baseline):
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

        df.loc[i, f'S{S}_{baseline}_ipw_est_CATE'] = ipw
        df.loc[i, f'S{S}_{baseline}_ipw_est_Y1'] = row['A'] * ipw
        df.loc[i, f'S{S}_{baseline}_ipw_est_Y0'] = - (1 - row['A']) * ipw
        

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


def single_mmr_run(test_signals, save_df, d, rct_size, os_size, B, kernel, cov_list, crop_prop,
                   px_dist_r, px_args_r, prop_fn_r, prop_args_r, tte_params_r,
                   px_dist_o, px_args_o, prop_fn_o, prop_args_o, tte_params_o):
    
    RCTData = SyntheticDataModule(save_df, d, rct_size, 0, px_dist_r, px_args_r, prop_fn_r, prop_args_r, tte_params_r)
    OSData = SyntheticDataModule(save_df, d, os_size, 1, px_dist_o, px_args_o, prop_fn_o, prop_args_o, tte_params_o)

    df_rct_oracle, df_rct = RCTData.get_df()
    df_os_oracle, df_os = OSData.get_df()

    df_combined = pd.concat([df_rct, df_os], axis=0, ignore_index=True)  # merge the dataframes into one
    df_comb_drop = df_combined.query('Delta == 1').reset_index(drop=True).copy()  # drop the censored observations

    # Estimate the nuisance parameters for the combined dataframe

    df_combined['P(S=1|X)'] = prop_score_est(df_combined.copy(), 'S', cov_list, 'logistic')

    df_combined.loc[df_combined.S==0, 'P(A=1|X,S)'] = prop_score_est(df_combined.query('S==0').copy(), 'A', cov_list, 'logistic')
    df_combined.loc[df_combined.S==1, 'P(A=1|X,S)'] = prop_score_est(df_combined.query('S==1').copy(), 'A', cov_list, 'logistic')

    _ = gc_est(df_combined, cov_list, tte_model='coxph')


    ipcw_est(df_combined, S=0)
    ipcw_est(df_combined, S=1)
    ipw_est(df_combined, S=0, baseline='impute')  # censored observations are IMPUTED
    ipw_est(df_combined, S=1, baseline='impute')  # censored observations are IMPUTED
    
    
    # Estimate the nuisance parameters for the combined dataframe with censored observations dropped
    
    df_comb_drop['P(S=1|X)'] = prop_score_est(df_comb_drop.copy(), 'S', cov_list, 'logistic')

    df_comb_drop.loc[df_comb_drop.S==0, 'P(A=1|X,S)'] = prop_score_est(df_comb_drop.query('S==0').copy(), 'A', cov_list, 'logistic')
    df_comb_drop.loc[df_comb_drop.S==1, 'P(A=1|X,S)'] = prop_score_est(df_comb_drop.query('S==1').copy(), 'A', cov_list, 'logistic')

    ipw_est(df_comb_drop, S=0, baseline='drop')  # censored observations are DROPPED
    ipw_est(df_comb_drop, S=1, baseline='drop')  # censored observations are DROPPED
    
    mmr_stats = np.zeros((len(test_signals), 2))  # store results and p-val for each mmr test

    for kind, key in enumerate(test_signals):
        
        if 'Drop' in key:
            
            if crop_prop:
                df_mmr = df_comb_drop[(0.05 < df_comb_drop['P(S=1|X)']) & (df_comb_drop['P(S=1|X)'] < 0.95) &\
                    (0.05 < df_comb_drop['P(A=1|X,S)']) & (df_comb_drop['P(A=1|X,S)'] < 0.95)].copy().reset_index(drop=True)
                
            else:
                df_mmr = df_comb_drop.copy()
        else:
            
            if crop_prop:
                df_mmr = df_combined[(0.05 < df_combined['P(S=1|X)']) & (df_combined['P(S=1|X)'] < 0.95) &\
                        (0.05 < df_combined['P(A=1|X,S)']) & (df_combined['P(A=1|X,S)'] < 0.95)].copy().reset_index(drop=True)
               
            else:
                df_mmr = df_combined.copy()
            
        signal0, signal1 = test_signals[key][0], test_signals[key][1]
        mmr_stats[kind, 0], mmr_stats[kind, 1] = mmr_test(df_mmr, cov_list, B, kernel, signal0, signal1)
        
    return mmr_stats