import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from time import sleep, time
from lifelines import CoxPHFitter
from SyntheticDataModule import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import tqdm

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
    
    cph_params = list(cph.params_)
    cph_params.insert(0,0)

    return t_event, np.array(est_base_surv).reshape(-1), np.array(cph_params)  

    
def est_surv(df, cov_list, tte_model):
    '''
    Estimate the survival function for the censoring time C and the time-to-event outcome Y
    
    @params:
        df: Data (pd.DataFrame)
        cov_list: list of covariates except from S and A (string)
        tte_model: generative model for the outcome (censoring variable) to be fit
    '''

    cbse = {}  # censoring-baseline-survival-estimate (cbse) 
    ybse = {}  # TimeToEvent-baseline-survival-estimate (ybse, TTE is y)
        
    for s in range(2):
        for a in range(2):
            
            # Estimate the survival function for the censoring variable C
            if len(df.query(f'S=={s} & A=={a} & Delta==0')) == 0:  # deal with "lifelines" lib errors 
                cbse[f't_S{s}_A{a}'], cbse[f'St_S{s}_A{a}'] = [-1], [1]
                
            else:
                if tte_model == 'coxph':
                    cbse[f't_S{s}_A{a}'], cbse[f'St_S{s}_A{a}'], cbse[f'beta_S{s}_A{a}'] = \
                    coxph_base_surv(df.query(f'S=={s} & A=={a}').copy(), cov_list[1:], flip=True) # fit for C 
                    
                else:
                    raise NotImplementedError(f'Time-to-event model <{tte_model}> is not implemented.')
                    
            # Estimate the survival function for the time-to-event variable Y
            if len(df.query(f'S=={s} & A=={a} & Delta==1')) == 0:
                ybse[f't_S{s}_A{a}'], ybse[f'St_S{s}_A{a}'] = [-1], [1]
                
            else:
                if tte_model == 'coxph':
                    ybse[f't_S{s}_A{a}'], ybse[f'St_S{s}_A{a}'], ybse[f'beta_S{s}_A{a}'] = \
                    coxph_base_surv(df.query(f'S=={s} & A=={a}').copy(), cov_list[1:], flip=False) # fit for Y
                    
                else:
                    raise NotImplementedError(f'Time-to-event model <{tte_model}> is not implemented.')
                    
    return cbse, ybse 


def eval_surv(s, a, x, T, surv_dict):
    '''
    Evaluate the value of a survival function specificied by "surv_dict, s, a" at "T" with covariates "x"
    '''
    
    t_arr = surv_dict[f't_S{s}_A{a}']
    st_arr = surv_dict[f'St_S{s}_A{a}']
    
    if list(t_arr) == [-1]:
        res = 1
    else:
        beta_arr = surv_dict[f'beta_S{s}_A{a}']
        base_val = st_arr[np.argmin(np.abs(T - t_arr))]
        res = base_val ** (np.exp(beta_arr @ x))
    
    return res   

    
#Conditional survival function (P(Y>yy|Y>cc))
def cond_surv(yy,cc,ybse, a,s, x, S0_t):
    """
    Returns the probability that Y>yy | Y>cc
    """
    if yy < cc: # survival probability is one if time is smaller than the condition
        return 1
    else:
        beta_arr = ybse[f'beta_S{s}_A{a}'] 

        return (np.interp(yy, ybse[f't_S{s}_A{a}'], ybse[f'St_S{s}_A{a}']) / S0_t)** (np.exp(beta_arr @ x))


def eval_Qfunc_ratio(s,a,x,T,ybse, thresh = 1e-6):
    """
    Q function with the ratio method + additional checks for stability

    """
    S0_t = np.interp(T, ybse[f't_S{s}_A{a}'], ybse[f'St_S{s}_A{a}'])

    beta_arr = ybse[f'beta_S{s}_A{a}'] 
    if (S0_t** (np.exp(beta_arr @ x))) < thresh: # If survival probability is too small, we just return the conditionned time.
        return T

    FY_C_S_A = lambda yy, cc : cond_surv(yy,cc, ybse,a =a ,s = s, x = x, S0_t = S0_t )
    t_max = np.max(np.max(ybse[f't_S{s}_A{a}']))
    if T>=t_max:
        return T # When we are outside the support, we assume instantaneous failure.
    else:
        return quad(lambda y : FY_C_S_A(y,T) ,a = 0, b=t_max, points = 100 )[0] 

def eval_Qfunc(s, a, x, T, ybse):  
    
    t_arr = ybse[f't_S{s}_A{a}']
    st_arr = ybse[f'St_S{s}_A{a}']
    beta_arr = ybse[f'beta_S{s}_A{a}'] 
    
    denum = eval_surv(s, a, x, T, ybse)
    
    stx_arr = st_arr ** (np.exp(beta_arr @ x))
    func = interp1d(t_arr, stx_arr, kind='linear', fill_value='extrapolate')
    num, _ = quad(func, T, t_arr.max(), limit=10)
    
    res = num / denum 
    
    return res, num, denum 


def eval_Qfunc_onarray(s, a, x, gt_arr, ybse):  
    
    denum = np.array([eval_surv(s, a, x, T, ybse) for T in gt_arr])
    
    t_arr = ybse[f't_S{s}_A{a}']
    st_arr = ybse[f'St_S{s}_A{a}']
    beta_arr = ybse[f'beta_S{s}_A{a}'] 
    stx_arr = st_arr ** (np.exp(beta_arr @ x))
    func = interp1d(t_arr, stx_arr, kind='linear', fill_value='extrapolate')
    
    func_vals = np.array(list(map(func, gt_arr)))
    gt_arr = np.append(gt_arr, t_arr.max())
    gt_diff_arr = [b - a for a, b in zip(gt_arr, gt_arr[1:])]
    
    areas = func_vals * gt_diff_arr
    shift_cumsum = np.roll(np.cumsum(areas), 1)
    shift_cumsum[0] = 0  
    num = np.sum(areas) - shift_cumsum

    res = num / denum 
    
    return res


def eval_int_term(s, a, x, T, cbse, ybse):
    
    t_arr = cbse[f't_S{s}_A{a}']
    
    if list(t_arr) == [-1]:
        res = 0    
    else:
        if len(np.where(t_arr < T)[0]) == 0:
            res = 0
        else:
            st_arr = cbse[f'St_S{s}_A{a}']
            beta_arr = cbse[f'beta_S{s}_A{a}']
            new_st_arr, idx = np.unique(st_arr, return_index=True)
            new_st_arr, idx = new_st_arr[::-1], idx[::-1]
            t_arr = t_arr[idx]
            
            stx_arr = new_st_arr ** (np.exp(beta_arr @ x))
            Gc = 1 - stx_arr
            Gc = np.append(Gc, 0)
            dGc = [b - a for a, b in zip(Gc, Gc[1:])]
            start_time = time()
            num = eval_Qfunc_onarray(s, a, x, t_arr, ybse)
            #print('Time elapsed for Q-function integration: {:.2f}'.format(time() - start_time))
            denum = np.array([eval_surv(s, a, x, c, cbse) ** 2 for c in t_arr])
            res = np.sum(dGc * num / denum)            
            
#             ub_ind = np.where(t_arr < T)[0][-1] + 1 # get upper bound index for T

#             t_arr = cbse[f't_S{s}_A{a}'][:ub_ind]
#             st_arr = cbse[f'St_S{s}_A{a}'][:ub_ind]
#             beta_arr = cbse[f'beta_S{s}_A{a}'] 

#             stx_arr = st_arr ** (np.exp(beta_arr @ x))
                
#             if len(stx_arr) == 1:
#                 res = (1 - stx_arr[0]) * eval_Qfunc(s, a, x, t_arr[0], ybse) / (eval_surv(s, a, x, t_arr[0], cbse) ** 2)
#             else:            
#                 deriv = np.gradient(1 - stx_arr)
#                 start_time = time()
#                 num = eval_Qfunc_onarray(s, a, x, t_arr, ybse)
#                 #print('Time elapsed for Q-function integration: {:.2f}'.format(time() - start_time))
#                 #num = np.array([eval_Qfunc(s, a, x, c, ybse) for c in t_arr])
#                 denum = np.array([eval_surv(s, a, x, c, cbse) ** 2 for c in t_arr])
#                 res = np.sum(deriv * num / denum)     
    
    return res


def eval_Ystar(s, a, x, delta, T, cbse, ybse):
    
    if delta == 1:
        ft_num = T
    else:
        ft_num = eval_Qfunc(s, a, x, T, ybse)

    first_term = ft_num / eval_surv(s, a, x, T, cbse) 
    res = first_term - eval_int_term(s, a, x, T, cbse, ybse)
    
    return res


def eval_mu(s, a, x, ybse):
    t_arr = ybse[f't_S{s}_A{a}']
    st_arr = ybse[f'St_S{s}_A{a}']
    beta_arr = ybse[f'beta_S{s}_A{a}'] 
    
    stx_arr = st_arr ** (np.exp(beta_arr @ x))
    func = interp1d(t_arr, stx_arr, kind='nearest-up', fill_value='extrapolate')
    res, _ = quad(func, 0, t_arr.max(), limit=4)
    
    return res     


def fill_barG(df, cov_list, cbse):
    '''
    Fill the values for G_bar(T|X,S,A) that are used in both the IPCW and the DR signal.
    '''
    df['G_bar(T|X,S,A)'] = df.apply(lambda r: eval_surv(int(r['S']), int(r['A']), r[cov_list], r['T'], cbse), axis=1)
    

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
            denom = psx * row['G_bar(T|X,S,A)']

            ipcw = row['T'] * (part1 - part0) / denom

        else:
            ipcw = 0

        df.loc[i, f'S{S}_ipcw_est_CATE'] = ipcw
        df.loc[i, f'S{S}_ipcw_est_Y1'] = row['A'] * ipcw
        df.loc[i, f'S{S}_ipcw_est_Y0'] = - (1 - row['A']) * ipcw
        

def cdr_est(df, cov_list, cbse, ybse, S):

    for i in range(len(df)):  
        row = df.loc[i]
        
        if row['S'] == S:
            aind = int(row['A'])
            psx = S * row['P(S=1|X)'] + (1 - S) * (1 - row['P(S=1|X)'])
            
#             t1 = time()
            
            mu_xsa1 = eval_mu(S, 1, row[cov_list], ybse)
            mu_xsa0 = eval_mu(S, 0, row[cov_list], ybse)
            
#             t2 = time()
#             print(f"mu time: {t2 - t1:.4f}")
            
            mu_xs = mu_xsa1 - mu_xsa0
            Ystar_xsa = eval_Ystar(S, aind, row[cov_list], row['Delta'], row['T'], cbse, ybse)
            
#             t3 = time()
#             print(f"YSTAR time: {t3 - t2:.4f}")
            
            mu_xsa = aind * mu_xsa1 + (1 - aind) * mu_xsa0
            pa_xs = aind * row['P(A=1|X,S)'] + (1 - aind) * (1 - row['P(A=1|X,S)'])            
            
            cdr = ((Ystar_xsa - mu_xsa) / pa_xs + mu_xs) / psx

        else:
            cdr = 0

        df.loc[i, f'S{S}_cdr_est_CATE'] = cdr


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

    cbse, ybse = est_surv(df_combined, cov_list, tte_model='coxph')
    fill_barG(df_combined, cov_list, cbse)

    if any("IPCW" in key for key in test_signals.keys()):
        ipcw_est(df_combined, S=0)
        ipcw_est(df_combined, S=1)
        
    if any("IPW-Impute" in key for key in test_signals.keys()):
        ipw_est(df_combined, S=0, baseline='impute')  # censored observations are IMPUTED
        ipw_est(df_combined, S=1, baseline='impute')  # censored observations are IMPUTED
        
    if any("CDR" in key for key in test_signals.keys()):
        cdr_est(df_combined, cov_list, cbse, ybse, S=0)  
        cdr_est(df_combined, cov_list, cbse, ybse, S=1)  
    
    
    # Estimate the nuisance parameters for the combined dataframe with censored observations dropped
    
    if any("IPW-Drop" in key for key in test_signals.keys()):
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