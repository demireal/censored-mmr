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
        return result.predict(X)

    elif model_name == 'mean':
        return y.mean() * np.ones(len(y))

    else:
        raise NotImplementedError(f'{model_name} is not implemented for propensity score estimation')


def coxph_base_surv(df, cov_list, flip=False):
    '''
    Fit a CoxPH model.

    @params:
        df: Data (pd.DataFrame)
        cov_list: regressor features (list of strings)
        flip: whether to flip the censoring indicator (bool). E.g., "True" if fitting the model on censoring variable

    @returns:
        t_event: event/censoring times (np.array)
        est_base_surv: corresponding baseline survival probabilities (np.array)
        cph.params_: beta estimates (np.array)
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
        cov_list: covariates (list of strings) 
        tte_model: generative model for the outcome (censoring variable) to be fit
        
    @return:
        separate dictionaries (for Y and C) that contain the survival model parameters for different S and A variables
    '''

    Fb_Y = {}  # dictionary for TimeToEvent (Y) baseline survival estimate  
    Gb_C = {}  # dictionary for CensoringTime (C) baseline survival estimate 
        
    for s in range(2):
        for a in range(2):
            
            # Estimate the survival function for the censoring variable C
            if len(df.query(f'S=={s} & A=={a} & Delta==0')) == 0:  # deal with "lifelines" lib errors 
                Gb_C[f't_S{s}_A{a}'], Gb_C[f'St_S{s}_A{a}'], Gb_C[f'beta_S{s}_A{a}'] = [-1], [1], np.zeros(len(cov_list))
                
            else:
                if tte_model == 'coxph':
                    Gb_C[f't_S{s}_A{a}'], Gb_C[f'St_S{s}_A{a}'], Gb_C[f'beta_S{s}_A{a}'] = \
                    coxph_base_surv(df.query(f'S=={s} & A=={a}').copy(), cov_list[1:], flip=True) # fit for C 
                    
                else:
                    raise NotImplementedError(f'Time-to-event model <{tte_model}> is not implemented.')
                    
            # Estimate the survival function for the time-to-event variable Y
            if len(df.query(f'S=={s} & A=={a} & Delta==1')) == 0:
                Fb_Y[f't_S{s}_A{a}'], Fb_Y[f'St_S{s}_A{a}'], Fb_Y[f'beta_S{s}_A{a}'] = [-1], [1], np.zeros(len(cov_list))
                
            else:
                if tte_model == 'coxph':
                    Fb_Y[f't_S{s}_A{a}'], Fb_Y[f'St_S{s}_A{a}'], Fb_Y[f'beta_S{s}_A{a}'] = \
                    coxph_base_surv(df.query(f'S=={s} & A=={a}').copy(), cov_list[1:], flip=False) # fit for Y
                    
                else:
                    raise NotImplementedError(f'Time-to-event model <{tte_model}> is not implemented.')
                    
    return Gb_C, Fb_Y 


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


def eval_Qfunc_(s, a, x, T, Fb_Y, thresh=1e-6):
    '''
    Q function with the ratio method + additional checks for stability
    '''
    
    Fb_sa = Fb_Y[f'St_S{s}_A{a}']               # Fb(t|S=s,A=a) = P(Y>t|S=s,A=a) (baseline survival function for Y)
    Fb_sa_t = Fb_Y[f't_S{s}_A{a}']              # t indices for Fb(t|S=s,A=a)
    Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}']        # CoxPH param. estimates for Fb(t|X,S=s,A=a)
    Fb_sax = Fb_sa ** (np.exp(Fb_sa_beta @ x))  # Fb(t|X=x,S=s,A=a) = P(Y>t|X=x,S=s,A=a)
    
    norm = eval_surv_(Fb_sa_t, Fb_sax, T) # normalization constant F(T|X,S,A)
    t_max = Fb_sa_t.max()

    if (norm < thresh) or (T >= t_max): # return T itself when it is too big to explode the denum. or outside support
        return T 
    else:
        Fb_csax = lambda yy, cc : eval_cond_surv_(yy, cc, Fb_sa_t, Fb_sax)  # Fb(t|X=x,Y>cc,S=s,A=a) = P(Y>t|X=x,Y>cc,S=s,A=a)
        return quad(lambda y : Fb_csax(y,T), a=0, b=t_max, limit=10)[0]
    
    
def eval_Qfunc_arr_(s, a, x, Gb_sa_t_idx, Fb_Y):  
    '''
    Evaluate the Q function for all the "C" values in array Gb_sa_t_idx
    '''
    
    Fb_sa = Fb_Y[f'St_S{s}_A{a}']               # Fb(t|S=s,A=a) = P(Y>t|S=s,A=a) (baseline survival function for Y)
    Fb_sa_t = Fb_Y[f't_S{s}_A{a}']              # t indices for Fb(t|S=s,A=a)
    Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}']        # CoxPH param. estimates for Fb(t|X,S=s,A=a)
    Fb_sax = Fb_sa ** (np.exp(Fb_sa_beta @ x))  # Fb(t|X=x,S=s,A=a) = P(Y>t|X=x,S=s,A=a)
    
    Fb_denum = np.array(list(map(lambda c: eval_surv_(Fb_sa_t, Fb_sax, c), Gb_sa_t_idx)))
    
    t_max = Fb_sa_t.max() # a proxy for the infinity upper bound on the integral
    t_int = np.append(Gb_sa_t_idx, t_max)
    
    func = interp1d(Fb_sa_t, Fb_sax, kind='linear', fill_value='extrapolate')  # Fb(t|X=x,S=s,A=a)
    # Calculate individual areas of Integral(Fb(t|..)) between every value in Gb_sa_t_idx and finally to infinity (tmax)
    interval_integrals = np.array(list(map(lambda c1, c2: quad(func, a=c1, b=c2, limit=2)[0], t_int[:-1], t_int[1:])))
    
    # Three lines below compute "integrals" which is equal to Fb(c|..) for every c in Gb_sa_t_idx
    shift_cumsum = np.roll(np.cumsum(interval_integrals), 1)
    shift_cumsum[0] = 0  
    Fb_num = np.sum(interval_integrals) - shift_cumsum
    
    return (Fb_num / Fb_denum) + Gb_sa_t_idx 


def eval_int_term_(s, a, x, T, Gb_C, Fb_Y):
    
    Gb_sa_t = Gb_C[f't_S{s}_A{a}']        # t indices for Gb(t|S=s,A=a)
    eval_idx = np.where(Gb_sa_t < T)[0]   # indices to evaluate the integral for
    Gb_sa_t_idx = Gb_sa_t[eval_idx].copy()
    
    if (list(Gb_sa_t) == [-1]) or (len(eval_idx) == 0): 
        return 0    

    else:
        Gb_sa = Gb_C[f'St_S{s}_A{a}']               # Gb(t|S=s,A=a) = P(C>t|S=s,A=a) (baseline survival function for C)
        Gb_sa_beta = Gb_C[f'beta_S{s}_A{a}']        # CoxPH param. estimates for Gb(t|X,S=s,A=a)
        Gb_sax = Gb_sa ** (np.exp(Gb_sa_beta @ x))  # Gb(t|X=x,S=s,A=a) = P(C>t|X=x,S=s,A=a)
        
        Gb_sax_idx = Gb_sax[eval_idx].copy()
     
        start_time = time()
    
        G_sax_idx = 1 - Gb_sax_idx
        dG_sax_idx = [b - a for a, b in zip(np.insert(G_sax_idx, 0, 0), np.insert(G_sax_idx, 0, 0)[1:])]    
        Q_num = eval_Qfunc_arr_(s, a, x, Gb_sa_t_idx, Fb_Y)
        Gbsq_denum = np.array(list(map(lambda c: eval_surv_(Gb_sa_t, Gb_sax, c) ** 2, Gb_sa_t_idx)))
        
        print('Time elapsed for compting the integral term: {:.2f}'.format(time() - start_time))
        
        return np.sum(dG_sax_idx * Q_num / Gbsq_denum)            
    

def eval_Ystar_(s, a, x, Delta, T, Gb_C, Fb_Y):
    '''
    Evaluate Y*_SA(X,Y,C)
    Note that T = min(Y,C), and it is enough for the calculation of this signal.
    '''
    
    if Delta == 1: # decide the numerator of the first term (ft) based on the value of Delta
        ft_num = T  
    else:
        ft_num = eval_Qfunc_(s, a, x, T, Fb_Y)

    Gb_sa = Gb_C[f'St_S{s}_A{a}']               # Gb(t|S=s,A=a) = P(C>t|S=s,A=a) (baseline survival function for C)
    Gb_sa_t = Gb_C[f't_S{s}_A{a}']              # t indices for Gb(t|S=s,A=a)
    Gb_sa_beta = Gb_C[f'beta_S{s}_A{a}']        # CoxPH param. estimates for Gb(t|X,S=s,A=a)
    Gb_sax = Gb_sa ** (np.exp(Gb_sa_beta @ x))  # Gb(t|X=x,S=s,A=a) = P(C>t|X=x,S=s,A=a)  
    
    ft_denum = eval_surv_(Gb_sa_t, Gb_sax, T)   # first term (ft) denumerator, always uses "T" regardless of the numerator
    
    ft = ft_num / ft_denum # calculate first term (ft)
    return ft - eval_int_term_(s, a, x, T, Gb_C, Fb_Y)


def eval_mu_(s, a, x, Fb_Y):
    '''
    Evaluate \mu_SA(X)
    '''
    Fb_sa = Fb_Y[f'St_S{s}_A{a}']               # Fb(t|S=s,A=a) = P(Y>t|S=s,A=a) (baseline survival function for Y)
    Fb_sa_t = Fb_Y[f't_S{s}_A{a}']              # t indices for Fb(t|S=s,A=a)
    Fb_sa_beta = Fb_Y[f'beta_S{s}_A{a}']        # CoxPH param. estimates for Fb(t|X,S=s,A=a) 
    Fb_sax = Fb_sa ** (np.exp(Fb_sa_beta @ x))  # Fb(t|X=x,S=s,A=a) = P(Y>t|X=x,S=s,A=a)
        
    func = interp1d(Fb_sa_t, Fb_sax, kind='nearest-up', fill_value='extrapolate')
    return quad(func, a=0, b=Fb_sa_t.max(), limit=10)[0]    


def fill_barG(df, cov_list, Gb_C):
    '''
    Fill the values for Gb(T|X,S,A) = P(C>T|X,S,A)   that are used in both the IPCW and the DR signal.
    '''
    df['Gb(T|X,S,A)'] = df.apply(lambda r:\
     eval_surv_(Gb_C[f"t_S{int(r['S'])}_A{int(r['A'])}"], Gb_C[f"St_S{int(r['S'])}_A{int(r['A'])}"], r['T']), axis=1)
    

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
        

def cdr_est(df, cov_list, cbse, ybse, S):

    for i in range(len(df)):  
        row = df.loc[i]
        
        if row['S'] == S:
            aind = int(row['A'])
            psx = S * row['P(S=1|X)'] + (1 - S) * (1 - row['P(S=1|X)'])
            
#             t1 = time()
            
            mu_xsa1 = eval_mu_(S, 1, row[cov_list], ybse)
            mu_xsa0 = eval_mu_(S, 0, row[cov_list], ybse)
            
#             t2 = time()
#             print(f"mu time: {t2 - t1:.4f}")
            
            mu_xs = mu_xsa1 - mu_xsa0
            Ystar_xsa = eval_Ystar_(S, aind, row[cov_list], row['Delta'], row['T'], cbse, ybse)
            
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

    Gb_C, Fb_Y = est_surv(df_combined, cov_list, tte_model='coxph')
    fill_barG(df_combined, cov_list, Gb_C)

    if any("IPCW" in key for key in test_signals.keys()):
        ipcw_est(df_combined, S=0)
        ipcw_est(df_combined, S=1)
        
    if any("IPW-Impute" in key for key in test_signals.keys()):
        ipw_est(df_combined, S=0, baseline='impute')  # censored observations are IMPUTED
        ipw_est(df_combined, S=1, baseline='impute')  # censored observations are IMPUTED
        
    if any("CDR" in key for key in test_signals.keys()):
        cdr_est(df_combined, cov_list, Gb_C, Fb_Y, S=0)  
        cdr_est(df_combined, cov_list, Gb_C, Fb_Y, S=1)  
    
    
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