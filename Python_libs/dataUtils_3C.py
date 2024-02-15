# %%
"""
This file is modified from Manuel Florez's code for
"Data-Driven Synthesis of Broadband Earthquake Ground Motions Using Artificial Intelligenc"
"""
import numpy as np
import pandas as pd

def rescale(v, v_min, v_max):
    """
    Rescale numpy array to be in the range [-1,1] for conditional variables 
    """
    dv = v_max-v_min
    vn = (v-v_min) / dv
    # rescale to [-1,1]
    vn = 2.0 * vn - 1.0
    return vn

def make_maps_scale(v_min, v_max):
    """
    rescale varaibles, min_max to [-1, 1], independent of input dimension, for PGA 
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


class SeisData(object):
    """
    Class to manage seismic data
    
    key_variables:
    --------------
    self.wfs: np.array
        normalized waveforms 
    self.df_meta_all : Dataframes
        all attributes, includes location
    self.df_meta : Dataframe
        attributes about conditioning variables
    self.vc_min : dict 
        store the min of conditional variables in order
    self.vc_max : dict
        store the max of the conditional variables in order
    self.vc_lst: tuple
        sotre the normalized conditional variables 
    
    """
    
    # -------------------------------------------------
    def __init__(self, data_file, attr_file, v_names, condv_min_max, batch_size, isel):
        """
        Parameters:
        -----------
        data_file : str
            location of waveforms data, load format ".npy", shape : [batch, 3, dimension] 
        attr_file : str
            location of attributes file, load format ".csv'
        v_names: tuple, list[str, ... str]
            list of conditional variables, like ['mag', 'Rrup', 'vs30',...]
        condv_min_max: tuple, list[(min, max), ... (min, max)]
            list of global normalization min max for v_names, same for training and validation dataset
        batch_size : int
            batch size for dataloader
        isel : np.array
            set of indices to use, separate training and validation dataset
        """
        
        self.data_file = data_file
        self.attr_file = attr_file
        
        self.v_names = v_names
        self.condv_min_max = condv_min_max
        self.batch_size = batch_size
        self.isel = isel 

        if not isinstance(v_names, list):
            assert False, "Please supply names of conditional variables"

        # load data
        print('Loading data ...')
        wfs = np.load(data_file)
        print('Loaded samples: ', wfs.shape[0])
        self.Ntrain = wfs.shape[0]
  
        # apply 3C normalization, self.wfs range from [min/max(abs), max/max(abs)]
        print('normalizing data ...')
        wfs_norm = np.max(np.abs(wfs), axis=2)     
        cnorms = wfs_norm.copy()
        wfs_norm = wfs_norm[:, :, np.newaxis] 
        self.wfs = wfs / wfs_norm

        # --- rescale norms -----
        pga_max = np.max(cnorms)
        pga_min = np.min(cnorms)
        log10_pga_max = np.log10(pga_max)
        log10_pga_min = np.log10(pga_min)
        print('max log pga:', log10_pga_max, 'min log pga:', log10_pga_min)
        
        # [-1, 1] normalization in log10 scale
        fn_to_scale_11, fn_to_real = make_maps_scale(log10_pga_min, log10_pga_max)     
        self.log10_PGA = fn_to_scale_11(np.log10(cnorms))
        self.fn_to_scale_11 = fn_to_scale_11
        self.fn_to_real = fn_to_real

        # load attributes for waveforms
        df = pd.read_csv(attr_file)
        self.df_meta_all = df.copy()
        self.df_meta = df[self.v_names]


        # ----- Configure conditional variables --------------
        # store normalization constants for conditional variables
        self._set_vc_max()
        # set values for conditional variables
        self._init_vcond()

        # partition the dataset
        Nsel = len(isel)
        self.Ntrain = Nsel
        
        # ----- sample a fracion of the dataset ------
        self.wfs = self.wfs[isel]
        self.log10_PGA = self.log10_PGA[isel]

        # get a list with the labels
        vc_b = []
        for v in self.vc_lst:
            vc_b.append(v[isel, :])
        self.vc_lst = vc_b

        # indices for the waveforms
        self.ix = np.arange(Nsel)
        # numpy array with conditional variables
        self.vc = np.hstack(self.vc_lst)

        self.Ntrain = Nsel
        print('Number selected samples: ', self.Ntrain)
        print('Class init done!')

        
    def to_real(self, vn, v_name): 
        """
        used for evaluation, same as to_real part of make_maps_scale
        """
        v_min = self.vc_min[v_name]
        v_max = self.vc_max[v_name]
        dv = v_max-v_min
        # rescale to [v_min, v_max]
        vn = (vn + 1.0)/2
        v = vn*dv + v_min
        return v

    def to_syn(self, vr, v_name):
        """
        used for evaluation, same as to_scale_11 part of make_maps_scale
        """
        v_min = self.vc_min[v_name]
        v_max = self.vc_max[v_name]
        dv = v_max-v_min
        # don't forget to rescale to [-1,1]
        vn = (vr-v_min) / dv
        vn = 2.0 * vn -1.0
        
        return vn


    def _set_vc_max(self):
        """
        store normalization constant for conditional variables in dict
        """
        self.vc_max = {}
        self.vc_min = {}
        for i, vname in enumerate(self.v_names):
            v_max = self.condv_min_max[i][1]
            self.vc_max[vname] = v_max
            
            v_min = self.condv_min_max[i][0]
            self.vc_min[vname] = v_min

    def _init_vcond(self):
        """
        Set values for conditional variables, each variable is normalized to be in [-1,1]
        """

        self.vc_lst = []
        for v_name in self.v_names:
            print('---------', v_name, '-----------')
            print('min ' + v_name, self.df_meta[v_name].min(), 'scale min', self.vc_min[v_name])
            print('max ' + v_name, self.df_meta[v_name].max(), 'scale max', self.vc_max[v_name])
            # 1. rescale variables to be between -1,1. we don't need the reverse
            v = rescale(self.df_meta[v_name].to_numpy(), self.vc_min[v_name], self.vc_max[v_name])
            # reshape conditional variables
            vc = np.reshape(v, (v.shape[0], 1))
            print('vc shape', vc.shape)
            # 3. store conditional variable
            self.vc_lst.append(vc)
        ## end method

    def _get_rand_idxs(self):
        """
        Randomly sample data
        :return: sorted random indeces for the batch that is going to be used
        """
        ib = np.random.choice(self.ix, size=self.batch_size, replace=False)
        ib.sort()
        return ib

    def get_rand_batch(self):
        """
        get a random sample from the data with batch_size 3C waveforms
        :return: wfs (numpy.ndarray), cond vars (list)
        """
        ib = self._get_rand_idxs()
        wfs_b = self.wfs[ib]
        # get the corresponding normalization constants
        log10_PGA_b = self.log10_PGA[ib]
        # get a list with tshe labels
        vc_b = []
        for v in self.vc_lst:
            vc_b.append(v[ib, :])

        return (wfs_b, log10_PGA_b, vc_b)

    def get_rand_cond_v(self):
        """
        Get a random sample of conditional variables
        """
        vc_b = []
        # sample conditional variables at random
        for v in self.vc_lst:
            ii = self._get_rand_idxs()
            vc_b.append(v[ii, :])

        return vc_b

    def get_batch_size(self):
        """
        Get the batch size used
        :return: int = batch size
        """
        return self.batch_size

    def __str__(self):
        return 'wfs data shape: ' + str(self.wfs.shape)

    def get_Ntrain(self):
        """
        Get total number of training samples in the seismic dataset
        :return: int
        """
        return self.Ntrain

    def get_Nbatches_tot(self):
        """
        get the total number of batches requiered for training: Ntrain/batch_size
        :return: int: number of batches
        """
        Nb_tot = np.floor(self.Ntrain / self.batch_size)
        return int(Nb_tot)

# %%

