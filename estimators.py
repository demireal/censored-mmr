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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def prop_score_est(df, target, feature, model_name='logistic'):
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
        
        
def mu_est_baseline(df, target, feature, model_name='XGboost'):
    '''
    Train a regression model, e.g., E[Y|X,S=0,A=1], and return its predictions.

    @params:
        df: Data to learn the model from (pd.DataFrame)
        target: target variable, e.g. Y (string)
        feature: regressor features (list of strings)
        model_name: model to use, e.g., XGboost (string)

    @return:
        predictions
    '''
    
    X = df[feature]  # Features
    y = df[target]  # Target variable
    
    if model_name == 'XGboost':      
        regressor = xgb.XGBRegressor()
    elif model_name == 'linear':
        regressor = LinearRegression()     
    else:
        raise NotImplementedError(f'{model_name} is not implemented for response surface estimation')
    
    regressor.fit(X, y)

    return regressor


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

    
def est_surv(df, tte_model, jD):
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
    cov_list = jD['cov_list']
        
    for s in range(2):
        for a in range(2):
            
            # Estimate the survival function for the censoring variable C
            if len(df.query(f'S=={s} & A=={a} & Delta==0')) == 0:  # deal with "lifelines" lib errors 
                
                Gb_C[f't_S{s}_A{a}'], Gb_C[f'St_S{s}_A{a}'], Gb_C[f'beta_S{s}_A{a}'] = [-1], [1], np.zeros(len(cov_list))
                
                Gb_C[f'St_S{s}_A{a}_misspec'] = [1]
                Gb_C[f'beta_S{s}_A{a}_misspec'] = np.zeros(len(cov_list))
                
                Gb_C[f'St_S{s}_A{a}_true'] = [1]
                Gb_C[f'beta_S{s}_A{a}_true'] = np.zeros(len(cov_list))
                
                
            else:
                if tte_model == 'coxph':
                    Gb_C[f't_S{s}_A{a}'], Gb_C[f'St_S{s}_A{a}'], Gb_C[f'beta_S{s}_A{a}'] = \
                    coxph_base_surv(df.query(f'S=={s} & A=={a}').copy(), cov_list[1:], flip=True) # fit for C 
                    
                    tmax = Gb_C[f't_S{s}_A{a}'].max()
                    Stmin = Gb_C[f'St_S{s}_A{a}'].min()
                    step_size = (1 - Stmin) / tmax

                    Gb_C[f'St_S{s}_A{a}_misspec'] =\
                    np.array(list(map(lambda c: np.maximum((1 - c * step_size), 0.01), Gb_C[f't_S{s}_A{a}'])))
                    
                    Gb_C[f'beta_S{s}_A{a}_misspec'] = np.zeros(len(cov_list))
                    
                    Gb_C[f'St_S{s}_A{a}_true'], Gb_C[f'beta_S{s}_A{a}_true'] =\
                    get_oracle_surv(Gb_C[f't_S{s}_A{a}'], jD, s, f'C{a}')
                    
                else:
                    raise NotImplementedError(f'Time-to-event model <{tte_model}> is not implemented.')
                    
            # Estimate the survival function for the time-to-event variable Y
            if len(df.query(f'S=={s} & A=={a} & Delta==1')) == 0:
                
                Fb_Y[f't_S{s}_A{a}'], Fb_Y[f'St_S{s}_A{a}'], Fb_Y[f'beta_S{s}_A{a}'] = [-1], [1], np.zeros(len(cov_list))
                
                Fb_Y[f'St_S{s}_A{a}_misspec'] = [1]
                Fb_Y[f'beta_S{s}_A{a}_misspec'] = np.zeros(len(cov_list))
                
                Fb_Y[f'St_S{s}_A{a}_true'] = [1]
                Fb_Y[f'beta_S{s}_A{a}_true'] = np.zeros(len(cov_list))
                
            else:
                if tte_model == 'coxph':
                    Fb_Y[f't_S{s}_A{a}'], Fb_Y[f'St_S{s}_A{a}'], Fb_Y[f'beta_S{s}_A{a}'] = \
                    coxph_base_surv(df.query(f'S=={s} & A=={a}').copy(), cov_list[1:], flip=False) # fit for Y
      
                    tmax = Fb_Y[f't_S{s}_A{a}'].max()
                    Stmin = Fb_Y[f'St_S{s}_A{a}'].min()
                    step_size = (1 - Stmin) / tmax

                    Fb_Y[f'St_S{s}_A{a}_misspec'] =\
                    np.array(list(map(lambda c: np.maximum((1 - c * step_size), 0.01), Fb_Y[f't_S{s}_A{a}'])))
                    
                    Fb_Y[f'beta_S{s}_A{a}_misspec'] = np.zeros(len(cov_list))
                    
                    Fb_Y[f'St_S{s}_A{a}_true'], Fb_Y[f'beta_S{s}_A{a}_true'] =\
                    get_oracle_surv(Fb_Y[f't_S{s}_A{a}'], jD, s, f'Y{a}')
                    
                else:
                    raise NotImplementedError(f'Time-to-event model <{tte_model}> is not implemented.')
                    
    return Gb_C, Fb_Y