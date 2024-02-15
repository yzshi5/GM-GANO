#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:41:35 2022

@author: glavrent
"""

#load libraries
import numpy as np
from scipy import integrate as scipy_int
from pylib_stats import lse

## Tapering
##-----------------------------------
def TaperingTH(time, sig, taper_frac, sides='both'):
    '''
    Timehistory tapering

    Parameters
    ----------
    time : np.array
        DESCRIPTION.
    sig : np.array
        Signal array.
    taper_frac : real
        tapering fraction.

    Returns
    -------
    sig_taper : np.array
        tapered signal time-history.
    taper_env : np.array
        tapering envelope.
    '''

    #offset timehistory to zero origin time
    time = time - min(time)
    #ground-motion duration
    gm_dur = max(time)
    
    #tapering length
    taper_len = taper_frac * gm_dur
    
    #tapering envelope
    taper_env1 = np.sin(2*np.pi * np.minimum(time,        taper_len)/(4*taper_len))
    taper_env2 =-np.sin(2*np.pi * np.maximum(time-gm_dur,-taper_len)/(4*taper_len))
    if sides == 'both':
        taper_env = taper_env1 * taper_env2
    elif sides == 'left':
        taper_env = taper_env1
    elif sides == 'right':
        taper_env = taper_env2
    else:
        taper_env = np.nan
    
    #tappered acc
    sig_taper = sig * taper_env
        
    return sig_taper, taper_env

## Integration
##-----------------------------------
def NewmarkIntegation(time, acc, int_type='midle point'):
    '''
    Newmark integration

    Parameters
    ----------
    time : np.array
        Time array.
    acc : np.array
        Acceleration time history.
    int_type : string, optional
        Integration type. The default is 'midle point'.

    Returns
    -------
    time : np.array
        Time array.
    acc : np.array
        Acceleration time history.
    vel : np.array
        Velocity time history.
    disp : np.array
        Displacement time history.
    '''

    assert(max(np.diff(time))-min(np.diff(time))<1e-9),'Error. For Newmark integration, time history must be on constant sampling interval (dt).'

    #time interval
    dt = np.diff(time)[0]

    #inegration parameters
    if int_type == 'explicit':
        gamma = 0.5
        beta  = 0
    elif int_type == 'midle point':
        gamma = 0.5
        beta  = 0.25
        
    #velocity
    vel = (1-gamma)*dt*acc[:-1] + gamma*dt*acc[1:]
    vel = np.insert(np.cumsum(vel), 0, 0.)
    
    #displacement
    disp = dt*vel[:-1] + dt**2/2*((1-2*beta)*acc[:-1] + 2*beta*acc[1:])
    disp = np.insert(np.cumsum(disp), 0, 0.)
    
    #convert units
    vel  *=981
    disp *=981
    
    return time, acc, vel, disp

## Differentiation
##-----------------------------------
def FDDifferentiate(time, disp):
    '''
    Finite differentiation.

    Parameters
    ----------
    time : np.array
        Time array.
    disp : np.array
        Displacement time history.

    Returns
    -------
    time : np.array
        Time array.
    acc : np.array
        Acceleration time history.
    vel : np.array
        Velocity time history.
    disp : np.array
        Displacement time history.
    '''

    assert(max(np.diff(time))-min(np.diff(time))<1e-9),'Error. For FFT differentiation, time history must be on constant sampling interval (dt).'

    #number of points
    n_pt = len(time)
    #time interval
    dt = np.diff(time)[0]

    #compute acc and vel
    vel = np.gradient(disp, dt)
    acc = np.gradient(vel, dt)

    #convert units
    acc /=981

    return time, acc, vel, disp

## Time History Processing
##-----------------------------------
def BaselineCorrection(time, vel, disp, n=6, f_taper_beg=0.05, f_taper_end=0.10):
    '''
    Polynomial baseline correction.

    Parameters
    ----------
    time : np.array
        Time array.
    vel : np.array
        Velocity time history.
    disp : np.array
        Displacement time history.
    n : int, optional
        Polynomial order. The default is 6.
    f_taper_beg : float, optional
        Cosine tapering fraction, begining. The default is 0.05.
    f_taper_end : float, optional
        Cosine tapering fraction, end. The default is 0.10.

    Returns
    -------
    time : np.array
        Time array.
    acc_bs : np.array
        Baselined acceleration time history.
    vel_bs : np.array
        Baselined velocity time history.
    disp_bs : np.array
        Baselined displacement time history.
    '''
    
    # #comptue vel and disp time histories
    # _, _, vel, disp = NewmarkIntegation(time, acc)
    
    time = time - np.min(time)
    t_const = time[[0,-1]]
    
    # Basine Line
    #projection matrices of polynomial for disp and velocity
    A_disp = np.vstack([time**(p+1) for p in range(n)]).T
    A_vel  = np.vstack([p*(time**p) for p in range(n)]).T
    #combine projection matrices and signals
    A = np.vstack([A_disp[1:,], A_vel[1:,]])
    b = np.hstack([disp[1:],    vel[1:]])
    
    # Constraints
    #projection matrices 
    A_disp_const = np.vstack([t_const**(p+1) for p in range(n)]).T
    A_vel_const  = np.vstack([p*(t_const**p) for p in range(n)]).T
    #combine projection matrices and signals
    A_const = np.vstack([A_disp_const[-1,], A_vel_const[-1,]])
    b_const = np.hstack([disp[-1], vel[-1]])
    
    #polynomial coefficients
    p = lse(A,b,A_const,b_const)
    
    #baseline corrected disp
    disp_bs = disp - A_disp @ p

    #apply tapering
    _, tpr_beg = TaperingTH(time, disp_bs, f_taper_beg, sides='left')
    _, trp_end = TaperingTH(time, disp_bs, f_taper_end, sides='right')
    disp_bs = (tpr_beg * trp_end) * disp_bs
    
    #differenticate baselined TH
    _, acc_bs, vel_bs, _ = FDDifferentiate(time, disp_bs)
    
    return time, acc_bs, vel_bs, disp_bs
