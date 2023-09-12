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
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel

from estimators import *
from utils import *


def eval_surv_(t, St, T):
    '''
    Evaluate P(Y>T) = St(T)
    '''
    
    if list(t) == [-1]:
        return 1
    else:
        return np.interp(T, t, St)
    

def eval_cond_surv_(y, c, t, St):
    '''
    Evaluate P(Y>y|Y>c) 
    The implementation here is correct under "(conditional) independence" between Y and C
    '''
    
    if y < c: # survival probability is one if time is smaller than the condition
        return 1
    else:
        num = eval_surv_(t, St, y)
        denum = eval_surv_(t, St, c)
        return num / denum 


def eval_Qfunc_(s, a, x, T, Fb_Y, mis_spec, thresh=1e-10):
    '''
    Q function with the ratio method + additional checks for stability
    '''
    
    Fb_sa_t = Fb_Y[f't_S{s}_A{a}']         # t indices for Fb(t|S=s,A=a)
    
    if mis_spec == 'Fb':
        Fb_sa = Fb_Y[f'St_S{s}_A{a}_misspec']         # Fb(t|S=s,A=a) = P(Y>t|S=s,A=a) (baseline survival function for Y)
        Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}_misspec']  # CoxPH param. estimates for Fb(t|X,S=s,A=a)   
    elif mis_spec == 'Gb':
        Fb_sa = Fb_Y[f'St_S{s}_A{a}_true'] 
        Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}_true']
    else:
        Fb_sa = Fb_Y[f'St_S{s}_A{a}']
        Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}'] 
        
    Fb_sax = Fb_sa ** (np.exp(Fb_sa_beta @ x))        # Fb(t|X=x,S=s,A=a) = P(Y>t|X=x,S=s,A=a)
    
    norm = eval_surv_(Fb_sa_t, Fb_sax, T) # normalization constant F(T|X,S,A)
    t_max = Fb_sa_t.max()

    if (norm < thresh) or (T >= t_max): # return T itself when it is too big to explode the denum. or outside support
        return T 
    else:
        Fb_csax = lambda yy, cc : eval_cond_surv_(yy, cc, Fb_sa_t, Fb_sax)  # Fb(t|X=x,Y>cc,S=s,A=a) = P(Y>t|X=x,Y>cc,S=s,A=a)
        return quad(lambda y : Fb_csax(y, T), a=0, b=t_max, limit=1)[0]
    
    
def eval_Qfunc_arr_(s, a, x, Gb_sa_t_idx, Fb_Y, mis_spec, thresh=1e-10):  
    '''
    Evaluate the Q function for all the "C" values in array Gb_sa_t_idx
    '''
    
    Fb_sa_t = Fb_Y[f't_S{s}_A{a}']         # t indices for Fb(t|S=s,A=a)
    
    if mis_spec == 'Fb':
        Fb_sa = Fb_Y[f'St_S{s}_A{a}_misspec']         # Fb(t|S=s,A=a) = P(Y>t|S=s,A=a) (baseline survival function for Y)
        Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}_misspec']  # CoxPH param. estimates for Fb(t|X,S=s,A=a)      
    elif mis_spec == 'Gb':
        Fb_sa = Fb_Y[f'St_S{s}_A{a}_true'] 
        Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}_true']
    else:
        Fb_sa = Fb_Y[f'St_S{s}_A{a}']
        Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}'] 
        
    Fb_sax = Fb_sa ** (np.exp(Fb_sa_beta @ x))        # Fb(t|X=x,S=s,A=a) = P(Y>t|X=x,S=s,A=a)
    
    Fb_denum = np.array(list(map(lambda c: eval_surv_(Fb_sa_t, Fb_sax, c), Gb_sa_t_idx)))
    thresh_indices = np.where(Fb_denum < thresh)[0]
        
    t_max = Fb_sa_t.max() # a proxy for the infinity upper bound on the integral
    t_int = np.append(Gb_sa_t_idx, t_max)
    
    func = interp1d(Fb_sa_t, Fb_sax, kind='nearest', fill_value='extrapolate')  # Fb(t|X=x,S=s,A=a)
    # Calculate individual areas of Integral(Fb(t|..)) between every value in Gb_sa_t_idx and finally to infinity (tmax)
    interval_integrals = np.array(list(map(lambda c1, c2: (c2 - c1) * (func(c2) + func(c1)) / 2, t_int[:-1], t_int[1:])))
    interval_integrals[-1] = quad(func, a=t_int[-2], b=t_int[-1], limit=1)[0]

    # Three lines below compute "integrals" which is equal to Fb(c|..) for every c in Gb_sa_t_idx
    shift_cumsum = np.roll(np.cumsum(interval_integrals), 1)
    shift_cumsum[0] = 0  
    Fb_num = np.sum(interval_integrals) - shift_cumsum
    
    fin_val = (Fb_num / Fb_denum) + Gb_sa_t_idx
    
    if len(thresh_indices) > 0:
        fin_val[thresh_indices[-1] - 1:] = Gb_sa_t_idx[thresh_indices[-1] - 1:]
    
    return fin_val


def eval_int_term_(s, a, x, T, Gb_C, Fb_Y, mis_spec):
    
    Gb_sa_t = Gb_C[f't_S{s}_A{a}']        # t indices for Gb(t|S=s,A=a)
    eval_idx = np.where(Gb_sa_t < T)[0]   # indices to evaluate the integral for
    
    if (list(Gb_sa_t) == [-1]) or (len(eval_idx) == 0): 
        return 0    

    else:
        if mis_spec == 'Gb':
            Gb_sa = Gb_C[f'St_S{s}_A{a}_misspec']  
            Gb_sa_beta = Gb_C[f'beta_S{s}_A{a}_misspec'] 
        elif mis_spec == 'Fb':
            Gb_sa = Gb_C[f'St_S{s}_A{a}_true'] 
            Gb_sa_beta = Gb_C[f'beta_S{s}_A{a}_true']
        else:
            Gb_sa = Gb_C[f'St_S{s}_A{a}']               # Gb(t|S=s,A=a) = P(C>t|S=s,A=a) (baseline survival function for C)
            Gb_sa_beta = Gb_C[f'beta_S{s}_A{a}']        # CoxPH param. estimates for Gb(t|X,S=s,A=a)
            
        Gb_sax = Gb_sa ** (np.exp(Gb_sa_beta @ x))      # Gb(t|X=x,S=s,A=a) = P(C>t|X=x,S=s,A=a)           
        
        Gb_sa_t_idx = Gb_sa_t[eval_idx].copy()
        Gb_sax_idx = Gb_sax[eval_idx].copy()
    
        G_sax_idx = 1 - Gb_sax_idx
        dG_sax_idx = [b - a for a, b in zip(np.insert(G_sax_idx, 0, 0), np.insert(G_sax_idx, 0, 0)[1:])]    
        Q_num = eval_Qfunc_arr_(s, a, x, Gb_sa_t_idx, Fb_Y, mis_spec)
        Gbsq_denum = np.array(list(map(lambda c: eval_surv_(Gb_sa_t, Gb_sax, c) ** 2, Gb_sa_t_idx)))
               
        return np.sum(dG_sax_idx * Q_num / Gbsq_denum)            
    

def eval_Ystar_(s, a, x, Delta, T, Gb_C, Fb_Y, mis_spec):
    '''
    Evaluate Y*_SA(X,Y,C)
    Note that T = min(Y,C), and it is enough for the calculation of this signal.
    '''
    
    if Delta == 1: # decide the numerator of the first term (ft) based on the value of Delta
        ft_num = T
    else:
        ft_num = eval_Qfunc_(s, a, x, T, Fb_Y, mis_spec)

    Gb_sa_t = Gb_C[f't_S{s}_A{a}']                  # t indices for Gb(t|S=s,A=a)
      
    if mis_spec == 'Gb':
        Gb_sa = Gb_C[f'St_S{s}_A{a}_misspec']  
        Gb_sa_beta = Gb_C[f'beta_S{s}_A{a}_misspec'] 
    elif mis_spec == 'Fb':
        Gb_sa = Gb_C[f'St_S{s}_A{a}_true'] 
        Gb_sa_beta = Gb_C[f'beta_S{s}_A{a}_true']
    else:
        Gb_sa = Gb_C[f'St_S{s}_A{a}']               # Gb(t|S=s,A=a) = P(C>t|S=s,A=a) (baseline survival function for C)
        Gb_sa_beta = Gb_C[f'beta_S{s}_A{a}']        # CoxPH param. estimates for Gb(t|X,S=s,A=a)

    Gb_sax = Gb_sa ** (np.exp(Gb_sa_beta @ x))      # Gb(t|X=x,S=s,A=a) = P(C>t|X=x,S=s,A=a) 

    ft_denum = eval_surv_(Gb_sa_t, Gb_sax, T)   # first term denumerator, always uses "T" regardless of the numerator
    ft = ft_num / ft_denum # calculate first term (ft)
    
    return ft - eval_int_term_(s, a, x, T, Gb_C, Fb_Y, mis_spec)


def eval_mu_(s, a, x, Fb_Y, mis_spec):
    '''
    Evaluate \mu_SA(X)
    '''
    
    Fb_sa_t = Fb_Y[f't_S{s}_A{a}']                    # t indices for Fb(t|S=s,A=a)
    
    if mis_spec == 'Fb':
        Fb_sa = Fb_Y[f'St_S{s}_A{a}_misspec']         # Fb(t|S=s,A=a) = P(Y>t|S=s,A=a) (baseline survival function for Y)
        Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}_misspec']  # CoxPH param. estimates for Fb(t|X,S=s,A=a) 
    elif mis_spec == 'Gb':
        Fb_sa = Fb_Y[f'St_S{s}_A{a}_true'] 
        Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}_true']
    else:
        Fb_sa = Fb_Y[f'St_S{s}_A{a}']
        Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}'] 
        
    Fb_sax = Fb_sa ** (np.exp(Fb_sa_beta @ x))        # Fb(t|X=x,S=s,A=a) = P(Y>t|X=x,S=s,A=a)
        
    func = interp1d(Fb_sa_t, Fb_sax, kind='linear', fill_value='extrapolate')
    return quad(func, a=0, b=Fb_sa_t.max(), limit=1)[0]    


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
            denom = psx * row['Gb(T|X,S,A)']

            ipcw = row['T'] * (part1 - part0) / denom

        else:
            ipcw = 0

        df.loc[i, f'S{S}_ipcw_est_CATE'] = ipcw
        df.loc[i, f'S{S}_ipcw_est_Y1'] = row['A'] * ipcw
        df.loc[i, f'S{S}_ipcw_est_Y0'] = - (1 - row['A']) * ipcw
        

def ipw_est(df, S, baseline):
    '''
    Calculate the instance-wise inverse propensity weighted signal for CATE.
    Record the IPW-signals in the dataframe.
    Note that we impute or drop (based on the baseline variable) the censored values.

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
        
        
def cdr_est(df, cov_list, Gb_C, Fb_Y, S, mis_spec):

    for i in range(len(df)):  
        row = df.loc[i] 
        
        if row['S'] == S:  # 1{S=s}
            aind = int(row['A'])
            sind = int(row['S'])
            
            if mis_spec == 'Gb':
                pa1_xs = 0.5
                pa0_xs = 0.5    
            else:
                pa1_xs = row['P(A=1|X,S)']  
                pa0_xs = (1 - row['P(A=1|X,S)'])                

            mu_xsa1 = eval_mu_(S, 1, row[cov_list], Fb_Y, mis_spec) 
            mu_xsa0 = eval_mu_(S, 0, row[cov_list], Fb_Y, mis_spec)

            mu_diff = mu_xsa1 - mu_xsa0
            
            if aind == 1:
                ystar_1 = eval_Ystar_(S, 1, row[cov_list], row['Delta'], row['T'], Gb_C, Fb_Y, mis_spec)
                part1 = (ystar_1 - mu_xsa1) / pa1_xs
            else:
                ystar_0 = eval_Ystar_(S, 0, row[cov_list], row['Delta'], row['T'], Gb_C, Fb_Y, mis_spec)
                part1 = -(ystar_0 - mu_xsa0) / pa0_xs

            overall = part1 + mu_diff
            
#             Ystar1 = eval_Ystar_(S, 1, row[cov_list], row['Delta'], row['T'], Gb_C, Fb_Y, mis_spec)
#             Ystar0 = eval_Ystar_(S, 0, row[cov_list], row['Delta'], row['T'], Gb_C, Fb_Y, mis_spec)
#             part1 = aind * (Ystar1 - mu_xsa1) / pa1_xs + mu_xsa1
#             part0 = (1 - aind) * (Ystar0 - mu_xsa0) / pa0_xs + mu_xsa0
#             overall = part1 - part0
            
            psx = S * row['P(S=1|X)'] + (1 - S) * (1 - row['P(S=1|X)'])
            cdr = overall / psx
                
        else:
            cdr = 0

        df.loc[i, f'S{S}_cdr_Miss_{mis_spec}_est_CATE'] = cdr
#         df.loc[i, f'S{S}_Ystar0_Miss_{mis_spec}_est_CATE'] = Ystar0
#         df.loc[i, f'S{S}_Ystar1_Miss_{mis_spec}_est_CATE'] = Ystar1
#         df.loc[i, f'S{S}_muxsa0_Miss_{mis_spec}_est_CATE'] = mu_xsa0
#         df.loc[i, f'S{S}_muxsa1_Miss_{mis_spec}_est_CATE'] = mu_xsa1
        
        
def dr_est(df, S, baseline):
    
    for i in range(len(df)):  
        row = df.loc[i] 
        
        if row['S'] == S:  # 1{S=s}
            aind = int(row['A'])
            Y = row['T']

            pa1_xs = row['P(A=1|X,S)']  
            pa0_xs = (1 - row['P(A=1|X,S)'])  

            mu_xsa1 = row['mu(Y|X,S,A=1)'] 
            mu_xsa0 = row['mu(Y|X,S,A=0)'] 

            part1 = aind * (Y - mu_xsa1) / pa1_xs + mu_xsa1
            part0 = (1 - aind) * (Y - mu_xsa0) / pa0_xs + mu_xsa0
            overall = part1 - part0
         
            psx = S * row['P(S=1|X)'] + (1 - S) * (1 - row['P(S=1|X)'])
            dr = overall / psx
                
        else:
            dr = 0

        df.loc[i, f'S{S}_{baseline}_dr_est_CATE'] = dr
        

def generate_data(d, os_size, jD):
    
    RCTData = SyntheticDataModule(jD['save_df'], d, jD['rct_size'], 0, jD['RCT']['px_dist'], jD['RCT']['px_args'], jD['RCT']['prop_fn'], jD['RCT']['prop_args'], jD['RCT']['tte_params'])
    OSData = SyntheticDataModule(jD['save_df'], d, os_size, 1, jD['OS']['px_dist'], jD['OS']['px_args'], jD['OS']['prop_fn'], jD['OS']['prop_args'], jD['OS']['tte_params'])

    _, df_rct = RCTData.get_df()
    _, df_os = OSData.get_df()

    df_combined = pd.concat([df_rct, df_os], axis=0, ignore_index=True)  # merge the dataframes into one
    df_comb_drop = df_combined.query('Delta == 1').reset_index(drop=True).copy()  # drop the censored observations
    
    return df_combined, df_comb_drop, RCTData, OSData


def est_nuisance(df_combined, df_comb_drop, jD):
    
    # Estimate the nuisance parameters for the combined dataframe

    df_combined['P(S=1|X)'] = prop_score_est(df_combined.copy(), 'S', jD['cov_list'])
    
    mu_regressor = {} 

    for sind in range(2):
        df_combined.loc[df_combined['S']==sind, 'P(A=1|X,S)'] =\
            prop_score_est(df_combined.query(f'S=={sind}').copy(), 'A', jD['cov_list'])
        
        for aind in range(2):
            mu_regressor[f'S{sind}_A{aind}'] =\
            mu_est_baseline(df_combined.query(f'S=={sind} & A=={aind}').copy(), 'T', jD['cov_list'])
            
        df_combined.loc[df_combined.S==sind, 'mu(Y|X,S,A=0)'] =\
            mu_regressor[f'S{sind}_A0'].predict(df_combined.loc[df_combined.S==sind, jD['cov_list']])
        df_combined.loc[df_combined.S==sind, 'mu(Y|X,S,A=1)'] =\
            mu_regressor[f'S{sind}_A1'].predict(df_combined.loc[df_combined.S==sind, jD['cov_list']])

    Gb_C, Fb_Y = est_surv(df_combined, 'coxph', jD)
    df_combined['Gb(T|X,S,A)'] = df_combined.apply(lambda r:\
     eval_surv_(Gb_C[f"t_S{int(r['S'])}_A{int(r['A'])}"], Gb_C[f"St_S{int(r['S'])}_A{int(r['A'])}"], r['T']), axis=1)

    if any("IPCW" in key for key in jD['test_signals'].keys()):
        ipcw_est(df_combined, S=0)
        ipcw_est(df_combined, S=1)
        
    if any("IPW-Impute" in key for key in jD['test_signals'].keys()):
        ipw_est(df_combined, S=0, baseline='impute')  # censored observations are IMPUTED
        ipw_est(df_combined, S=1, baseline='impute')  # censored observations are IMPUTED
        
    if any("DR-Impute" in key for key in jD['test_signals'].keys()):
        dr_est(df_combined, S=0, baseline='impute')  # censored observations are IMPUTED
        dr_est(df_combined, S=1, baseline='impute')  # censored observations are IMPUTED
        
    if any("CDR" in key for key in jD['test_signals'].keys()):
        cdr_est(df_combined, jD['cov_list'], Gb_C, Fb_Y, S=0, mis_spec='None')  
        cdr_est(df_combined, jD['cov_list'], Gb_C, Fb_Y, S=1, mis_spec='None')  
        
    if any("CDR-MissF" in key for key in jD['test_signals'].keys()):
        cdr_est(df_combined, jD['cov_list'], Gb_C, Fb_Y, S=0, mis_spec='Fb')  
        cdr_est(df_combined, jD['cov_list'], Gb_C, Fb_Y, S=1, mis_spec='Fb')
        
    if any("CDR-MissG" in key for key in jD['test_signals'].keys()):
        cdr_est(df_combined, jD['cov_list'], Gb_C, Fb_Y, S=0, mis_spec='Gb')  
        cdr_est(df_combined, jD['cov_list'], Gb_C, Fb_Y, S=1, mis_spec='Gb')    
    
    
    # Estimate the nuisance parameters for the combined dataframe with censored observations dropped
    
    if any(key in ['IPW-Drop', 'DR-Drop'] for key in jD['test_signals'].keys()):
        
        df_comb_drop['P(S=1|X)'] = prop_score_est(df_comb_drop.copy(), 'S', jD['cov_list'], 'logistic')
        
        mu_regressor = {} 

        for sind in range(2):
            df_comb_drop.loc[df_comb_drop['S']==sind, 'P(A=1|X,S)'] =\
                prop_score_est(df_comb_drop.query(f'S=={sind}').copy(), 'A', jD['cov_list'])

            for aind in range(2):
                mu_regressor[f'S{sind}_A{aind}'] =\
                mu_est_baseline(df_comb_drop.query(f'S=={sind} & A=={aind}').copy(), 'T', jD['cov_list'])

            df_comb_drop.loc[df_comb_drop.S==sind, 'mu(Y|X,S,A=0)'] =\
                mu_regressor[f'S{sind}_A0'].predict(df_comb_drop.loc[df_comb_drop.S==sind, jD['cov_list']])
            df_comb_drop.loc[df_comb_drop.S==sind, 'mu(Y|X,S,A=1)'] =\
                mu_regressor[f'S{sind}_A1'].predict(df_comb_drop.loc[df_comb_drop.S==sind, jD['cov_list']])

        if any("IPW-Drop" in key for key in jD['test_signals'].keys()):
            ipw_est(df_comb_drop, S=0, baseline='drop')  # censored observations are DROPPED
            ipw_est(df_comb_drop, S=1, baseline='drop')  # censored observations are DROPPED
       
        if any("DR-Drop" in key for key in jD['test_signals'].keys()):
            dr_est(df_comb_drop, S=0, baseline='drop')  # censored observations are DROPPED
            dr_est(df_comb_drop, S=1, baseline='drop')  # censored observations are DROPPED
        
    return Fb_Y, Gb_C


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


def mmr_run(d, os_size, kernel, jD):
    
    df_combined, df_comb_drop, _, _ = generate_data(d, os_size, jD)
    _, _ = est_nuisance(df_combined, df_comb_drop, jD)
    
    mmr_stats = np.zeros((len(jD['test_signals']), 2))  # store results and p-val for each mmr test

    for kind, key in enumerate(jD['test_signals']):
        
        if 'Drop' in key:
            df_mmr = df_comb_drop.copy()
        else:
            df_mmr = df_combined.copy()
            
        if jD['crop_prop'] and ('Drop' not in key):
            df_mmr = df_mmr[(0.05 < df_mmr['P(S=1|X)']) & (df_mmr['P(S=1|X)'] < 0.95) &\
                    (0.05 < df_mmr['P(A=1|X,S)']) & (df_mmr['P(A=1|X,S)'] < 0.95) &\
                    (1e-4 < df_mmr['Gb(T|X,S,A)'])].copy().reset_index(drop=True)
            
        if jD['crop_prop'] and ('Drop' in key):
            df_mmr = df_mmr[(0.05 < df_mmr['P(S=1|X)']) & (df_mmr['P(S=1|X)'] < 0.95) &\
                    (0.05 < df_mmr['P(A=1|X,S)']) & (df_mmr['P(A=1|X,S)'] < 0.95)].copy().reset_index(drop=True)
            
        signal0, signal1 = jD['test_signals'][key][0], jD['test_signals'][key][1]
        mmr_stats[kind, 0], mmr_stats[kind, 1] = mmr_test(df_mmr, jD['cov_list'], jD['B'], kernel, signal0, signal1)
        
    return mmr_stats