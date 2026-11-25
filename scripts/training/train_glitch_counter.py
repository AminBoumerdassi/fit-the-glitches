'''
This script trains a neural network to count the no. of glitches in some observing window.
'''

#numpy and matplotlib imports
import numpy as np
import matplotlib.pyplot as plt

#torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from ema_pytorch import EMA
from autoclip.torch import QuantileClip

#sklearn imports
from sklearn.model_selection import train_test_split

#Other imports
from few.utils.constants import YRSID_SI
import os
import copy
import random
import datetime

# #pywavelet imports
# import pywavelet
# from pywavelet.types.wavelet_bins import compute_bins

# #Disable pywavelet logging of warnings
# import logging
# logging.getLogger("pywavelet").setLevel(logging.ERROR)

#Set the default precisions, backends and device for torch
torch.set_default_dtype(torch.float32)###torch.float64
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# #Set the default precisions/backends for pywavelet
# pywavelet.set_backend("cupy", "float32") if use_cuda else pywavelet.set_backend("numpy", "float32")#"float64","float64"

# Data generator, training loop and net architecture imports
'''These imports need rewriting'''
from src.tools.EMRI_generator_TDI import EMRIGeneratorTDI
from src.tools.test_and_train_loop import *
from src.tools.model_architecture import *
from src.tools.losses import *
from src.tools.transforms import *

#For debugging (ONLY turn on when needed)
torch.autograd.set_detect_anomaly(False)

# Using mixed precision
use_amp=True#False
scaler = torch.amp.GradScaler(device, enabled=use_amp)

# Reproducibility & determinism for pytorch, numpy and cupy
seed=2026
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # numpy random generator
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
g = torch.Generator()
g.manual_seed(seed)

# Initialise the model and move it to device
model= ???.to(device)

#Trick: compile the model
model = torch.compile(model)

#Initialise autoclip object
clipper = QuantileClip(model.parameters(), quantile=0.1, history_length=1000)

# Initialise loss function and optimizer
lr=1e-3#
weight_decay = 0.0#1e-5
loss_fn= criterion().to(device)
optimizer= torch.optim.RAdam(params=model.parameters(), 
                             lr=lr,
                             decoupled_weight_decay=True)

#Defining Lr scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)#5000

# Continuing training? Set here!!!!!!!!
checkpoint_path = None

if checkpoint_path is not None:
    print("Continuing training!")
    #Load checkpoint dict
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    #load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    #Load optimiser state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #Load scheduler state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #Load scaler state
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    #Set the most recent losses
    train_history=[]
    val_history=[checkpoint['loss']]
    #Set the most recent metrics
    #Set model to train mode
    model.train()
    #Get previous random states
    torch.set_rng_state(checkpoint["torch_rng_state"].to("cpu")),
    torch.cuda.set_rng_state_all([checkpoint["cuda_rng_state"][0].to("cpu")])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    g.set_state(checkpoint["generator_rng_state"].to("cpu"))
else:
    print("Training from scratch!")
    start_epoch = 0
    #initialise training and validation histories
    train_history=[]
    val_history=[]
    #Initialise metric histories

#Setting generator parameters
len_seq= 2**18#23#2**22 gives around 1.3 years, 2**23 around 2.6 years
dt=10.#10
fs=1/dt
T= len_seq*dt/round(YRSID_SI)#Not actually input into the generator, it already calculates this and stores as an attribute
TDI_channels="AE"
n_channels=len(TDI_channels)

#Setting training hyperparameters
batch_size=32
epochs=2000#200#100
test_size=0.3

#Preprocessing parameters
max_absolute_A_E = np.load("/fred/oz303/aboumerd/EMRI_denoising/preprocessing/max_absolute_A_E.npy")

#Load dataset of EMRI parameters and apply test-train split
EMRI_params_dir="training_data/EMRI_params.npy"
EMRI_params_and_SNRs= np.load(EMRI_params_dir, allow_pickle=True)
train_params, val_params= train_test_split(EMRI_params_and_SNRs, test_size=test_size, random_state=seed)

#Initialise the glitch generators for the training and val data
training_set= ???(train_params, 
                               dim=len_seq, 
                               dt=dt, 
                               TDI_channels=TDI_channels, 
                               add_noise=add_noise, 
                               add_glitches=add_glitches, 
                               seed=seed, 
                               target_SNR_range=training_target_SNR_range, 
                               use_gpu=use_cuda, 
                               random_windows=random_windows, 
                               random_sign_flips=random_sign_flips,
                               max_absolute=max_absolute_A_E,
                               curriculum_rate=0.5#None,#None#0.5 seems to be best
                               )

validation_set= ???(val_params, 
                                dim=len_seq, 
                                dt=dt, 
                                TDI_channels=TDI_channels, 
                                add_noise=add_noise, 
                                add_glitches=add_glitches, 
                                seed=seed, 
                                target_SNR_range=validation_target_SNR_range, 
                                use_gpu=use_cuda, 
                                random_windows=False, 
                                random_sign_flips=False,
                                max_absolute=max_absolute_A_E,
                                curriculum_rate=None,#
                                )

#Get current states so we can reset the val set at each epoch
val_torch_rng_state = torch.get_rng_state()
val_cuda_rng_state = torch.cuda.get_rng_state_all()
val_numpy_rng_state = np.random.get_state()
val_python_rng_state = random.getstate()

#Set the current state for the EMRI generators
training_set.set_seed_state(val_torch_rng_state, 
                            val_cuda_rng_state, 
                            val_numpy_rng_state)

validation_set.set_seed_state(val_torch_rng_state, 
                            val_cuda_rng_state, 
                            val_numpy_rng_state)

#Apply Repeated Sampling of Random Subsets on training data to save time
samples_per_epoch = 1024
sampler = torch.utils.data.RandomSampler(training_set, 
                                         num_samples=samples_per_epoch, 
                                         replacement=False, 
                                         generator=g)

#Use a fixed subset of the validation to save time
no_val_samples= 1024
validation_subset= torch.utils.data.Subset(validation_set, 
                                           np.arange(no_val_samples).tolist())

#Initialise training dataloader
training_dataloader= torch.utils.data.DataLoader(training_set, 
                                                 batch_size=batch_size, 
                                                 sampler=sampler, 
                                                 shuffle=None, 
                                                 drop_last=True, 
                                                 generator=g,
                                                 collate_fn=training_set.???)

#Initialise validation dataloader
'''Generator needs a fixed state with NO data augmentation'''
val_g = torch.Generator()
val_g.manual_seed(seed)
validation_dataloader= torch.utils.data.DataLoader(validation_subset, 
                                                   batch_size=batch_size, 
                                                   shuffle=True, 
                                                   drop_last=True, 
                                                   generator=val_g,
                                                   collate_fn=training_set.???)

#See the architecture of the model
'''Internally cropping the signal'''
summary(model, input_size=(batch_size, n_channels, len_seq - 2*training_set.crop_idx))#len_seq

#Declare generator's parameters
training_set.declare_generator_params()

#Declare hyperparameters
print("#################################")
print("####TRAINING HYPERPARAMETERS####")
print("#Batch size: ", batch_size)
print("#Learning rate:", lr)
print("#Training proportion of dataset: ", 1-test_size)
print("#No. epochs: ", epochs)
print("#################################")

#Initialise a summary writer for tensorboard
record_run = False
log_dir = "/fred/oz303/aboumerd/EMRI_denoising/runs_convtasnet/"
logfolder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
comment=''
filename_suffix=''

if record_run:
    os.mkdir(log_dir+logfolder)
    writer = SummaryWriter(log_dir= log_dir+logfolder, 
                        filename_suffix=filename_suffix, 
                        comment=comment)

#Training loop 
for t in range(start_epoch, epochs):
    print(f"-----------------------------------\n\t\tEpoch {t+1}/{epochs}\n-----------------------------------")
    #Initialise variables for measuring time of an epoch
    start= torch.cuda.Event(enable_timing=True)
    end= torch.cuda.Event(enable_timing=True)
    #Apply per-epoch data augmentations on training but NOT validation set
    # training_set.retune_SNRs(low=validation_target_SNR_range[0], high=validation_target_SNR_range[1])
    training_glitch_background_idx = np.random.randint(1,128)
    print(f"Training set has glitch BG no. {training_glitch_background_idx}")
    training_set.load_glitch_background(glitch_background_dir=glitch_background_dir,
                                                        background_idx=training_glitch_background_idx)
    #Start training loop    
    start.record()
    train_loop(training_dataloader, 
               model, 
               loss_fn, 
               optimizer, 
               batch_size, 
               train_history, 
               scaler, 
               str(device), 
               use_amp=use_amp, 
               max_abs_tensor=None, 
               ema=ema,
               pred_overlap_history=train_pred_overlap_history,
               residual_overlap_history=train_residual_overlap_history,
               clipper=clipper,
               )
    #Reset the seed and state of the validation dataloader before each loop so the validation data doesn't change
    val_g.manual_seed(seed)
    validation_set.set_seed_state(val_torch_rng_state, 
                                val_cuda_rng_state, 
                                val_numpy_rng_state)
    #Start val. loop
    val_loop(validation_dataloader, 
             model, 
             loss_fn, 
             val_history, 
             scaler, 
             str(device), 
             use_amp=use_amp, 
             max_abs_tensor=None, 
             pred_overlap_history=val_pred_overlap_history, 
             input_overlap_history=input_overlap_history,
             ema=ema,
             val_history_EMA=val_history_EMA,
             pred_overlap_history_EMA=val_pred_overlap_history_EMA,
             residual_overlap_history=val_residual_overlap_history,
             residual_overlap_history_EMA=val_residual_overlap_history_EMA,
             )
    #Step the scheduler after each val loop
    if scheduler is not None: scheduler.step()#Can need extra arguments depending on the scheduler
    end.record()

    #Print time for 1 epoch
    torch.cuda.synchronize()
    print("Epoch time: {:.2f}s\n".format(start.elapsed_time(end)/1000))

    #Save a checkpoint for further training if needed
    checkpoint = {
        'epoch': t,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': val_history[-1],
        "pred_overlap_history" : val_pred_overlap_history[-1],
        #For reproducibility
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "generator_rng_state": g.get_state()
    }
    torch.save(checkpoint, "model_checkpoint.tar")
    

    #Record training, val. losses, and reconstructions for tensorboard
    if record_run:
        writer.add_scalar("Loss/train", train_history[t-start_epoch], t)
        writer.add_scalar("Loss/validation", val_history[t-start_epoch], t)

#Close the tensorboard writer
if record_run:
    writer.flush()
    writer.close()
print("Done!")