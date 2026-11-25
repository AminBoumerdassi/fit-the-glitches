import numpy as np
import cupy as cp
import copy
import torch
import torch.utils.dlpack as dlpack

import pytorch_pfn_extras as ppe

import glob
from h5py import File

# #FEW imports
# import sys
# import os
# from numpy.random import default_rng
# from few.trajectory.inspiral import EMRIInspiral
# from few.waveform import Pn5AAKWaveform, GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
# from few.utils.geodesic import get_separatrix
# from few.utils.utility import get_p_at_t
# from few.utils.constants import YRSID_SI

#LISA tools imports
from lisatools.sensitivity import get_sensitivity, A1TDISens, T1TDISens
from lisatools.detector import EqualArmlengthOrbits, ESAOrbits
from lisatools.utils.constants import lisaLT

#fast lisa response imports
from fastlisaresponse import ResponseWrapper

#oise whitening/noise generation imports
from scipy.signal.windows import tukey
import scipy.signal as sp_signal
import cupyx.scipy.signal as cp_signal

class TDIDataGenerator(torch.utils.data.Dataset):
    'Generates data for PyTorch'
    def __init__(self, 
                 dim=2**21, 
                 dt=10.,  
                 TDI_channels="AE", 
                 seed=2023, 
                 use_gpu = True, 
                 random_sign_flips=False,
                 max_absolute=None,
                 ):
        'Initialization'
        self.dim = dim
        self.dt = dt
        self.TDI_channels=TDI_channels
        self.channels_dict= {"AET":["AE","AE","T"], "AE":["AE","AE"]}
        #For use in the noise generation and whitening functions
        self.n_channels = len(TDI_channels)
        self.add_noise= add_noise
        self.seed= seed
        self.use_gpu= use_gpu
        #For data augmentation
        self.random_sign_flips = random_sign_flips
        #Gpu check
        if self.use_gpu == True:
            self.xp = cp
            self.xp_signal = cp_signal
            self.backend_type= "cuda12x"
            self.device="cuda"
            #Ensure cupy mem pooling enabled
            self.xp.get_default_memory_pool()
            #Make cupy use the same stream as torch
            torch_stream = torch.cuda.current_stream().cuda_stream
            self.xp.cuda.ExternalStream(torch_stream)
            #Also share memory pool between torch and cp
            '''NOTE: this makes errors messages extremely verbose. Temporarily disable whenever debugging is needed'''
            #ppe.cuda.use_torch_mempool_in_cupy()
        else:
            self.xp =np
            self.xp_signal = sp_signal
            self.backend_type= "cpu"
            self.device="cpu"
        #For preprocessing
        if max_absolute is None:
            self.max_absolute = 1.0
        else:
            self.max_absolute = self.xp.array(max_absolute)[:,None]
        self.inverse = False
        #initialise RNG for noise generation with a fixed seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        # Initialise the PSD
        self.PSD_AE = noise_PSD_AE(self.dim,
                                   self.dt, 
                                   TDI = 'TDI2', 
                                   include_foreground=True, 
                                   model='scirdv1', 
                                   xp=self.xp
                                   )
        # Initialise the tukey window
        self.alpha = 1/8
        self.window = self.xp_signal.windows.tukey(self.dim, alpha=self.alpha)
        # Initialise the bandpass object
        lowcut = 1e-4
        highcut= None
        order = 4
        self.bandpass = init_bandpass(dt,
                                      lowcut,
                                      highcut=None,
                                      order=order,
                                      use_gpu=self.use_gpu)
        # Initialise the crop index
        self.crop_idx = int(self.alpha * self.dim)
        #Tune the luminosity distances to ensure the SNRs are in the target range
    def __len__(self):
        'Denotes the total number of samples in the dataset'
        return self.EMRI_params_set_size
    def __getitem__(self, index):
        'Generates one sample of data'
        #Get inspiral index
        plunge_idx = self.get_inspiral_idx(index)
        #Start with the clean EMRI i.e. the target
        clean_EMRI = self.get_EMRI(index)
        return clean_EMRI, plunge_idx
    def get_EMRI(self, index):
        'Generate a single noise-whitened TDI EMRI.'
        #Generate the list of waveforms, convert to array, then truncate to the correct length
        return None
    def get_glitch_bg_batch(self, batch_size):
        '''Only changing glitch backgrounds per-epoch due to long I/O times.'''
        #Create a randomised window of the 2y glitch background
        start_idx = self.xp.random.randint(0, self.glitches_td_AET.shape[-1] - self.dim - 1, size=batch_size)#X.shape[-1] - 1
        idxs = start_idx[:,None] + self.xp.arange(self.dim)#X.shape[-1]
        # end_idx = start_idx + X.shape[-1]
        #Crop the background to match the length of the input signal
        cropped_glitches_td_AET = self.glitches_td_AET[:, idxs]#start_idx:end_idx
        '''Lazy fix: swap axes on cropped_glitches_td_AET to get (BS, no. chans, len seq)'''
        cropped_glitches_td_AET = self.xp.swapaxes(cropped_glitches_td_AET, 0, 1)
        return cropped_glitches_td_AET

    def preprocess_batch(self, batch):
        '''Preprocess a batch of data'''
        # Separate targets and auxiliary info
        '''At this point, targets is a tuple of (A channel, E channel)'''
        targets, plunge_idxs = zip(*batch)
        # Stack targets into single array
        targets = self.xp.stack(targets, axis=0)
        # concatenate the input and target arrays
        combined = self.xp.concatenate([targets, targets], axis=0)
        # Define batch size
        bs = combined.shape[0]//2
        #Add stationary noise if wanted
        combined[:bs,...] += noise_td_AET_BATCHWISE(self.dim, 
                            self.dt, 
                            self.PSD_AE,
                            targets.shape[0],
                            return_cupy=self.use_gpu) if self.add_noise else 0.0

        #Bandpass
        combined = bandpass(self.bandpass,
                              combined,
                              use_gpu=self.use_gpu,
                              axis=-1
                              )

        #Whiten
        combined = noise_whiten_AET(combined, 
                              self.dt, 
                              self.PSD_AE, 
                              window=self.window,
                              axis=-1
                              )

        #Preprocess
        combined = preprocess_signal(combined, 
                              max_absolute=self.max_absolute,
                              inverse = self.inverse
                              )

        # Crop to remove whitening artefacts and tapering
        combined = combined[..., self.crop_idx:-self.crop_idx]

        #Convert data to float32 precision if needed
        copy=False
        combined = combined.astype(self.xp.float32, 
                    copy=copy
                    )
        #Split data into inputs and targets
        inputs, targets = self.xp.split(combined, 2, axis=0)

        #Convert data from xp arrays to PyTorch tensors
        return torch.from_dlpack(inputs), torch.from_dlpack(targets), plunge_idxs

    def declare_generator_params(self):
        #Declare generator parameters
        print("#################################")
        print("####DATASET PARAMETERS####")
        print("#Dataset size: ", self.EMRI_params_set_size)
        print("#Time in years:", self.T)
        print("#n_channels: ", self.n_channels)
        print("#dt in seconds: ",self.dt)
        print("#Length of timeseries:", self.dim)
        print("Noise background: ", self.add_noise)
        print("#################################")
    def set_seed_state(self, torch_rng_state, cuda_rng_state, numpy_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        torch.cuda.set_rng_state_all(cuda_rng_state)#[0].to("cpu")
        np.random.set_state(numpy_rng_state)
