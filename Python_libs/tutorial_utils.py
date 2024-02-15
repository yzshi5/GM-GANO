import numpy as np
import torch
from scipy.signal import detrend
import matplotlib.pyplot as plt
# load the baseline correction function
import pylib_gm_proc
from pylib_gm_proc import TaperingTH, NewmarkIntegation, FDDifferentiate, BaselineCorrection

def make_maps_scale(v_min, v_max):
    """
    rescale varaibles, min_max to [-1, 1], independent of input dimension, for PGA or PGV
    Parameters:
    ----------
    v_min : int
    v_max : int
    """
    def to_scale_11(x):
        dv = v_max-v_min
        xn = (x-v_min) / dv
        # rescale to [-1,1]
        xn = 2.0*xn -1.0
        return xn
    
    def to_real(x):
        dv = v_max-v_min
        xr = (x+1.0)/2.0        
        xr = xr*dv+v_min
        return xr
    
    return (to_scale_11, to_real)

# from kik-net data, log10(PGV) max and min
log10_pgv_max = 0.09604963680854223
log10_pgv_min = -4.779544184431937
fn_to_scale_11, fn_to_real = make_maps_scale(log10_pgv_min, log10_pgv_max) 


def to_syn(config, v_value, attribute_name):
    """
    rescale to [-1, 1]
    """
    idx = config['condv_names'].index(attribute_name)
    
    v_min = config['condv_min_max'][idx][0]
    v_max = config['condv_min_max'][idx][1]
    
    #print("idx, v_min, v_max", idx, v_min, v_max)
    dv = v_max-v_min
    # rescale to [0,1] # don't forget to rescale to [-1,1]
    vn = (v_value-v_min) / dv
    vn = 2.0 * vn -1.0

    return vn

def convert_attributes(config, v_names):
    """
    convert attributes to normalized conditional variables, GmGANO takes normalized variables as input
    """
    
    mag_cur = to_syn(config, v_names['magnitude'], 'magnitude')
    dist_cur = to_syn(config, v_names['rrup'], 'rrup')
    vs30_cur = to_syn(config, v_names['vs30'], 'vs30')
    tect_cur = to_syn(config, v_names['tectonic_value'], 'tectonic_value')
    v_c = np.asanyarray([mag_cur, dist_cur, vs30_cur, tect_cur])

    #print(v_c.shape) 
    return v_c 

def generate_scen_data(G, grf, v_all, fn_to_real=fn_to_real, one_condition=False, velocity=False, n_syn=100, ndim=6000, time_step=0.01, device='cpu'):
    """
    Generate synthetic data 
    
    Parameters
    ----------
    G: GANO model
    grf : 
        gaussian random field function
    v_all : numpy.array
        [N, number_of_variables]   
    fn_to_real: function
        convert the scale of the waveforms to real scale
    one_condition : boolean 
        True: generate 100 synthetic waveforms for one given condition 
        False: generate N synthetic waveforms for N conditions (1 realization for each) 
    velocity : boolen
        True: trained using velocity
        False: acceleration 
        
    Returns:
    --------
    detrended generated waveforms
    
    """
    v_all = torch.Tensor(v_all)
    
    if one_condition == True:
        # generate n synthetic waveforms
        fake_wfs_all = torch.zeros(n_syn, 3, ndim)
        
        with torch.no_grad():
            label = v_all.unsqueeze(0).unsqueeze(2)
            label = label.repeat(n_syn, 1, 1).permute(0, 2, 1).float()
            
            # x_syn shape is [N, 6, ndim+npad]
            x_syn = G(grf.sample(label.shape[0]).to(device), label.to(device))
            
            # last 3 components of fake_lcn are scaled log10(PGV)
            fake_lcn = torch.mean(x_syn[:, 3:, :], dim=-1)
            
            # convert scaled log10(PGV) to real PGV (m/s)
            fake_lcn = torch.pow(10, fn_to_real(fake_lcn))
            
            # first 3 components of x_syn are normalized 3C components 
            fake_wfs = x_syn[:, :3, :]
            
            # combine normalized waveforms and PGV
            fake_wfs_all = fake_wfs * fake_lcn.unsqueeze(2)
            
    else:
        fake_wfs_all = torch.zeros((len(v_all), 3, ndim))
        with torch.no_grad():
            for i in range(len(v_all)):
                label = v_all[i].unsqueeze(0).unsqueeze(2)
                label = label.repeat(1, 1, 1).permute(0, 2, 1).float()
                x_syn = G(grf.sample(label.shape[0]).to(device), label.to(device))
                fake_lcn = torch.mean(x_syn[:, 3:, :], dim=-1)
                fake_lcn = torch.pow(10, fn_to_real(fake_lcn))
                fake_wfs = x_syn[:, :3, :]

                fake_wfs = fake_wfs * fake_lcn.unsqueeze(2)
                fake_wfs_all[i,:,:] = fake_wfs
                  
    # move generated waveforms to CPU 
    fake_wfs_all = fake_wfs_all.detach().cpu()
    
    # if velocity, differentiate to get accelerogram
    if velocity:
        time_step = 0.01
        fake_wfs_all[:,:,1:] = (fake_wfs_all[:,:,1:] - fake_wfs_all[:,:,:-1])/time_step  
        fake_wfs_all[:,:,0] = 0
    
    # detrend waveforms
    fake_wfs_scen_detrend = detrend(fake_wfs_all.numpy(), type='linear')
    fake_wfs_scen_detrend = detrend(fake_wfs_scen_detrend, type='constant')

    return fake_wfs_scen_detrend
        

def baseline_correction(time,wfs_scen):
    """
    Baseline correction method, apply a 7-order polynominal function to fit the daa to forces the motion (acc, vel, disp) to stop at the end. 
    """
    wfs_scen_corrected = np.zeros_like(wfs_scen)
    
    for i in range(len(wfs_scen)):
        for j in range(3):
            _, _, vel_nm, disp_nm = NewmarkIntegation(time, wfs_scen[i,j,:], int_type='midle point')
            _, acc_bs, vel_bs, disp_bs = BaselineCorrection(time, vel_nm, disp_nm, n=7, f_taper_beg=0.05, f_taper_end=0.05)
            wfs_scen_corrected[i,j,:] = acc_bs
            
    return wfs_scen_corrected



def plot_one_example(plt_wfs, v_names):
    """
    take in one 3-C waveform and description
    """
    t = np.arange(0, plt_wfs.shape[1]) * 0.01 # sampling is 100Hz
    box = dict(boxstyle='round', facecolor='white', alpha=1)
    text_loc = [0.02, 0.87]

    plt.figure(figsize=(12,9))
    
    if v_names['tectonic_value'] == 0:
        tectonic_type = 'Subduction'
    elif v_names['tectonic_value'] == 1:
        tectonic_type = 'Shallow Crustal'
    plt.suptitle('Magnitude = {}, Rrup = {} km, Vs30 = {} m/s, {}'.format(v_names['magnitude'], v_names['rrup'], v_names['vs30'], tectonic_type), fontsize=18)
    plt.subplot(311)
    plt.plot(t, plt_wfs[0, :], 'k', label='E', linewidth=0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    tmp_min = np.min(plt_wfs[0, :])
    tmp_max = np.max(plt_wfs[0, :])

    plt.ylabel('Amplitude $(m/s^2)$',fontsize='large')
    plt.legend(loc='upper right', fontsize='medium')
    plt.gca().set_xticklabels([])

    plt.subplot(312)
    plt.plot(t, plt_wfs[1,:], 'k', label='N', linewidth=0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    tmp_min = np.min(plt_wfs[1,:])
    tmp_max = np.max(plt_wfs[1,:])

    plt.ylabel('Amplitude $(m/s^2)$',fontsize='large')
    plt.legend(loc='upper right', fontsize='medium')
    plt.gca().set_xticklabels([])
    plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center',
            transform=plt.gca().transAxes, fontsize="medium", fontweight="normal", bbox=box)

    plt.subplot(313)
    plt.plot(t, plt_wfs[2,:], 'k', label='Z', linewidth=0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    tmp_min = np.min(plt_wfs[2,:])
    tmp_max = np.max(plt_wfs[2,:])

    plt.ylabel('Amplitude $(m/s^2)$',fontsize='large')
    plt.legend(loc='upper right', fontsize='medium')
    #plt.gca().set_xticklabels([])
    plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='center',
            transform=plt.gca().transAxes, fontsize="medium", fontweight="normal", bbox=box)

    plt.legend(loc='upper right', fontsize='medium', ncol=2)
    plt.xlabel('Time (s)',fontsize='large' )
    plt.tight_layout()
    plt.gcf().align_labels()