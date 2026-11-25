'''
This is a custom Pytorch dataset for generating time-domain EMRIs from a given set of parameters.
It uses sets of EMRI parameters to generate and store time-domain EMRIs only for as long as is needed in a particular batch. 
'''
#---------------------------------------------------------------------------------------
#Adapted from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
#---------------------------------------------------------------------------------------

import numpy as np
import cupy as cp
import copy
import torch
import torch.utils.dlpack as dlpack

import pytorch_pfn_extras as ppe

import glob
from h5py import File

from .EMRI_analysis_tools import *

from .transforms import *

#FEW imports
import sys
import os
from numpy.random import default_rng
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import Pn5AAKWaveform, GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from few.utils.geodesic import get_separatrix
from few.utils.utility import get_p_at_t
from few.utils.constants import YRSID_SI

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

#pywavelet import
import pywavelet
# from pywavelet.backend import *
# from pywavelet import set_backend
# from pywavelet.types import FrequencySeries, Wavelet, TimeSeries

#Disable pywavelet logging of warnings
import logging
logging.getLogger("pywavelet").setLevel(logging.ERROR)



class EMRIGeneratorTDI(torch.utils.data.Dataset):
    'Generates data for PyTorch'
    def __init__(self, 
                 EMRI_params_and_SNRs,
                 waveform_model="FastSchwarzschildEccentricFlux", 
                 dim=2**21, 
                 dt=10.,  
                 TDI_channels="AE", 
                 target_SNR_range=[70,80], 
                 seed=2023, 
                 add_noise=False, 
                 add_glitches=False, 
                 use_gpu = True, 
                 random_windows=False, 
                 random_sign_flips=False,
                 max_absolute=None,
                 curriculum_rate=0.0, 
                 ):
        'Initialization'
        self.EMRI_params_and_SNRs= copy.copy(EMRI_params_and_SNRs)#shape: (no. EMRIs, no. parameters + SNR)
        self.EMRI_params_set_size= self.EMRI_params_and_SNRs.shape[0]
        self.reference_SNRs = copy.copy(EMRI_params_and_SNRs[:,-1])
        self.reference_luminosity_dists = copy.copy(EMRI_params_and_SNRs[:,6])
        self.dim = dim
        self.dt = dt
        self.TDI_channels=TDI_channels
        self.channels_dict= {"AET":["AE","AE","T"], "AE":["AE","AE"]}
        self.random_windows = random_windows
        #For use in the noise generation and whitening functions
        self.n_channels = len(TDI_channels)
        self.add_noise= add_noise
        self.seed= seed
        self.use_gpu= use_gpu
        #For use in the glitch generation
        self.add_glitches = add_glitches
        self.glitch_scale_factor = 1.0
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
        #For curriculum learning
        self.curriculum_rate = curriculum_rate
        #initialise RNG for noise generation with a fixed seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        # self.random.seed(self.seed)
        #Initialise the waveform generator
        inspiral_kwargs = {}
        sum_kwargs = {"pad_output": True}
        amplitude_kwargs = {}
        self.waveform_generator = GenerateEMRIWaveform(waveform_model, 
                                                  force_backend=self.backend_type, 
                                                  sum_kwargs=sum_kwargs, 
                                                  inspiral_kwargs=inspiral_kwargs, 
                                                  amplitude_kwargs=amplitude_kwargs
                                                  )
        #Initialise the corresponding trajectory module
        func_dict = {"FastSchwarzschildEccentricFlux" : "SchwarzEccFlux"}
        self.traj = EMRIInspiral(func=func_dict[waveform_model])
        #Set the time window to 2 years if we want randomised windows of EMRIs
        if self.random_windows:
            self.T= 2.0
            long_dim = int(self.T*365*86400/dt)
            #Init TDI wrapper
            self.TDI_wrapper= init_TDI(self.waveform_generator, 
                                       long_dim, 
                                       self.dt, 
                                       tdi_chan=TDI_channels, 
                                       orbit_obj=EqualArmlengthOrbits(), 
                                       use_gpu=use_gpu
                                       )
        else:
            self.T= (dim*dt/YRSID_SI)+0.005#A tiny bit extra on T to ensure output length =>dim
            #Initialise the TDI wrapper
            self.t0 = 0.0
            self.t_buffer = 20000
            self.TDI_wrapper= init_TDI(self.waveform_generator, 
                                       self.dim, 
                                       self.dt, 
                                       tdi_chan=TDI_channels, 
                                       orbit_obj=EqualArmlengthOrbits(), 
                                       force_backend=self.backend_type,
                                       t0=self.t0,
                                       t_buffer = self.t_buffer,
                                       )
        # Initialise the PSD
        self.PSD_AE = noise_PSD_AE(self.dim,#self.TDI_wrapper.response_model.num_pts,
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
        self.retune_SNRs(low=target_SNR_range[0], high=target_SNR_range[1])
        #Load a glitch background if needed, rescale, and resample
        if self.add_glitches:
            #Pick a particular glitch background
            self.background_idx=0
            #Load & resample it
            self.load_glitch_background(glitch_background_dir="/fred/oz303/aboumerd/EMRI_Glitches/data_files/glitch_bg_AET/max_glitch_SNR_inf/",
                                                            background_idx=self.background_idx)
            #Resample the cropped glitch background to match the 0.1Hz fs
            # self.glitches_td_AET = self.xp.array([self.xp_signal.resample_poly(self.glitches_td_AET[channel,:], 1,5, axis=-1) for channel in range(self.glitches_td_AET.shape[0])])
            #Initialise the rescaled glitches
            # self.retune_glitches(glitch_scale_factor=1.0)
            # self.glitch_scale_factor = 1.0
            # self.rescaled_glitches_td_AET = self.glitch_scale_factor * self.glitches_td_AET
        #Custom kernel for slicing glitch backgrounds
        self.add_glitch_kernel = cp.RawKernel(
            r'''
            extern "C" __global__
            void add_glitch_windows(
                float* inputs,
                const float* background,
                const long long* starts,
                const float* signs,
                const long long background_length,
                const long long dim,
                const long long channels,
                const long long batch_size
            ) {
                long long idx =
                    (long long)blockIdx.x * blockDim.x + threadIdx.x;

                long long total = batch_size * channels * dim;

                if (idx >= total) return;

                long long t = idx % dim;
                long long c = (idx / dim) % channels;
                long long b = idx / (dim * channels);

                long long glitch_idx =
                    c * background_length + starts[b] + t;

                inputs[idx] += signs[b] * background[glitch_idx];
            }
            ''',
            'add_glitch_windows'
        )
    def __len__(self):
        'Denotes the total number of samples in the dataset'
        return self.EMRI_params_set_size
    def __getitem__(self, index):
        'Generates one sample of data'
        #Get inspiral index
        plunge_idx = self.get_inspiral_idx(index)
        #Start with the clean EMRI i.e. the target
        clean_EMRI = self.get_EMRI(index)
        #Construct the input data, starting with the clean EMRI
        #X = y.copy()
        # #Add stationary noise if wanted
        # X += noise_td_AET(self.dim, 
        #                     self.dt, 
        #                     self.PSD_AE, 
        #                     return_cupy=self.use_gpu) if self.add_noise else 0.0
        # Add glitches to the data if wanted
        # if self.add_glitches:
        #     '''Only changing glitch backgrounds per-epoch due to long I/O times.'''
        #     #Create a randomised window of the 2y glitch background
        #     start_idx = np.random.randint(0, self.glitches_td_AET.shape[-1] - X.shape[-1] - 1)
        #     end_idx = start_idx + X.shape[-1]
        #     #Crop the background to match the length of the input signal
        #     cropped_glitches_td_AET = self.glitches_td_AET[: , start_idx:end_idx]
        #     #Randomly flip the sign of the cropped glitches
        #     cropped_glitches_td_AET *= self.xp.random.choice([1.0,-1.0], size=1) if self.random_sign_flips else 1.0
        #     '''Something weird: randomly dropout the glitches for curriculum learning'''
        #     if self.curriculum_rate is not None:
        #         X += self.xp.random.choice([0.0, 1.0], size=1, p=[self.curriculum_rate, 1-self.curriculum_rate]) * cropped_glitches_td_AET
        #     else:
        #         X += cropped_glitches_td_AET
        return clean_EMRI, plunge_idx#X
    def get_EMRI(self, index):
        'Generate a single noise-whitened TDI EMRI.'
        #Generate the list of waveforms, convert to array, then truncate to the correct length
        waveform= self.xp.asarray(generate_TDI_EMRI(self.EMRI_params_and_SNRs[index,:-1], self.TDI_wrapper))[:,:self.dim]
        #Randomly flip sign of the ORIGINAL waveform
        # if self.random_sign_flips: waveform *= self.xp.random.choice([1.0,-1.0], size=1)
        return waveform
        # #Initialise the input signal array
        # signal_AET = self.xp.zeros_like(waveform)
        # #Add the clean EMRI to the data
        # signal_AET += waveform
        # #Add the stationary noise to the data
        # if self.add_noise: signal_AET += 1.0 * noise_td_AET(self.dim, self.dt, self.PSD_AE, return_cupy=self.use_gpu)
        #Add glitches to the data
        # if self.add_glitches:
        #     '''Only changing glitch backgrounds per-epoch due to long I/O times.'''
        #     #Create a randomised window of the 2y glitch background
        #     start_idx = np.random.randint(0, self.glitches_td_AET.shape[-1] - signal_AET.shape[-1] - 1)
        #     end_idx = start_idx + signal_AET.shape[-1]
        #     #Crop the background to match the length of the input signal
        #     cropped_glitches_td_AET = self.glitches_td_AET[: , start_idx:end_idx]
        #     #Randomly flip the sign of the cropped glitches
        #     if self.random_sign_flips: cropped_glitches_td_AET *= self.xp.random.choice([1.0,-1.0], size=1)
        #     '''Something weird: randomly dropout the glitches for curriculum learning'''
        #     if self.curriculum_rate is not None:
        #         # '''This is purely randomised, no dependence on curriculum rate'''
        #         # rate = self.xp.random.uniform(low=0.1, high=0.9)
        #         # signal_AET += self.xp.random.choice([0.0, 1.0], size=1, p=[rate, 1-rate]) * cropped_glitches_td_AET
        #         signal_AET += self.xp.random.choice([0.0, 1.0], size=1, p=[self.curriculum_rate, 1-self.curriculum_rate]) * cropped_glitches_td_AET
        #     else:
        #         signal_AET += cropped_glitches_td_AET
        # return X, y, plunge_idx
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
        # #Randomly flip the sign of the cropped glitches
        # cropped_glitches_td_AET *= self.xp.random.choice([1.0,-1.0], size=1) if self.random_sign_flips else 1.0
        # '''Something weird: randomly dropout the glitches for curriculum learning'''
        if self.curriculum_rate is not None:
            cropped_glitches_td_AET *= self.xp.random.choice([0.0, 1.0], size=batch_size, p=[self.curriculum_rate, 1-self.curriculum_rate])[...,None,None]
        return cropped_glitches_td_AET

    def get_inspiral_idx(self, index):
        #Get the length of the inspiral in seconds
        inspiral_length = self.get_inspiral_length(self.traj,
                                                   self.EMRI_params_and_SNRs[index,:],
                                                   self.dt,
                                                   self.T,
                                                   )
        #Get the index corresponding to the end of the inspiral
        plunge_idx= int(self.dim * inspiral_length / (self.T*YRSID_SI))
        #Because we are cropping the signal, modify the plunge idx accordingly
        if plunge_idx > int(self.dim * 6/8):
            plunge_idx = int(self.dim * 6/8)
        else:
            plunge_idx -= int(self.dim * 1/8)
        return plunge_idx

    def preprocess_batch(self, batch):
        '''Preprocess a batch of data'''
        # Separate targets and auxiliary info
        '''At this point, targets is a tuple of (A channel, E channel)'''
        targets, plunge_idxs = zip(*batch)
        # Stack targets into single array
        targets = self.xp.stack(targets, axis=0)
        # concatenate the input and target arrays
        combined = self.xp.concatenate([targets, targets], axis=0)
        # # Stack samples
        # combined = self.xp.stack(combined, axis=0)
        # Define batch size
        bs = combined.shape[0]//2
        #Randomly flip the sign of EMRI
        EMRI_sign = self.xp.random.choice([1.0,-1.0], size=bs) if self.random_sign_flips else 1.0
        combined[:bs,...] *= EMRI_sign[...,None,None]
        combined[bs:,...] *= EMRI_sign[...,None,None]        
        #Add stationary noise if wanted
        combined[:bs,...] += noise_td_AET_BATCHWISE(self.dim, 
                            self.dt, 
                            self.PSD_AE,
                            targets.shape[0],
                            return_cupy=self.use_gpu) if self.add_noise else 0.0

        #Add glitches if wanted and randomly flip sign
        '''combined[:bs,...] = self.add_glitch_background(combined[:bs,...])'''
        glitch_sign = self.xp.random.choice([1.0,-1.0], size=targets.shape[0]) if self.random_sign_flips else 1.0
        combined[:bs,...] += glitch_sign[...,None,None] * self.get_glitch_bg_batch(bs) if self.add_glitches else 0.0

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

    def add_glitch_background(self, inputs):

        batch_size, channels, dim = inputs.shape

        background = self.glitches_td_AET

        # Random starting positions
        starts = self.xp.random.randint(
            0,
            background.shape[-1] - dim + 1,
            size=batch_size,
            dtype=self.xp.int64
        )

        # Random signs, if enabled
        if self.random_sign_flips:
            signs = self.xp.random.choice(
                self.xp.asarray([-1.0, 1.0], dtype=self.xp.float32),
                size=batch_size
            )
        else:
            signs = self.xp.ones(
                batch_size,
                dtype=self.xp.float32
            )

        threads = 256
        total = batch_size * channels * dim
        blocks = (total + threads - 1) // threads

        self.add_glitch_kernel(
            (blocks,),
            (threads,),
            (
                inputs,
                background,
                starts,
                signs,
                background.shape[-1],
                dim,
                channels,
                batch_size
            )
        )

        return inputs
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
    def retune_SNRs(self, low=70, high=80):
        #Calculate the luminosity distances that make the SNRs uniformly distributed in the target range
        target_SNRs = np.random.uniform(low=low, high=high, size=self.EMRI_params_set_size)
        target_luminosity_dist = get_target_luminosity_dist(target_SNRs, self.reference_SNRs, self.reference_luminosity_dists)
        #Reset the luminosity distances in the parameters
        self.EMRI_params_and_SNRs[:,6] = target_luminosity_dist
    def load_glitch_background(self, glitch_background_dir=None, background_idx=0):
        #Glob the glitch background files
        glitch_background_files = glob.glob(glitch_background_dir + "BG_*.h5")
        #Choose the specified one
        glitch_background_file = glitch_background_files[background_idx]
        #Load file
        with File(glitch_background_file, 'r') as f:
            '''
            Note: the cupy implementation of resample_poly is broken
            for multidimensional arrays so use it channel by channel.
            '''
            self.glitches_td_AET = self.xp.array([f["A"][()], f["E"][()]])#
            glitch_dt= f["dt"][()]#Original fs of 0.5Hz
        #Resample the glitch background
        self.glitches_td_AET = self.xp.array([self.xp_signal.resample_poly(self.glitches_td_AET[channel,:], 1,5, axis=-1) for channel in range(self.glitches_td_AET.shape[0])])
    def random_flip_on_glitch_background(self):
        '''Randomly flip the glitch background by +-1'''
        self.glitches_td_AET *= self.xp.random.choice([1.0,-1.0], size=1)
    # def retune_glitches(self, glitch_scale_factor=1.0):
    #     '''Rescale the glitch backgrounds by multiplying by alpha'''
    #     #Change the scale factor
    #     self.glitch_scale_factor = glitch_scale_factor
    #     self.rescaled_glitches_td_AET = self.glitch_scale_factor * self.glitches_td_AET
    def set_seed_state(self, torch_rng_state, cuda_rng_state, numpy_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        torch.cuda.set_rng_state_all(cuda_rng_state)#[0].to("cpu")
        np.random.set_state(numpy_rng_state)
        # random.setstate(checkpoint["python_rng_state"])
        # g.set_state(checkpoint["generator_rng_state"].to("cpu"))
    def get_inspiral_length(self, traj, params, dt, T):
        '''Returns end of inspiral or end of observing window in seconds,
        whichever is sooner.'''
        t, _, _, _, _, _, _ = traj(params[0],
                                   params[1],
                                   params[2],
                                   params[3],
                                   params[4],
                                   params[5],
                                   Phi_phi0 = params[-4],
                                   Phi_theta0= params[-3],
                                   Phi_r0 = params[-2],
                                   dt=dt,
                                   T=T,
                                   )
        return t[-1]
