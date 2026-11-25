import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.tools.EMRI_analysis_tools import *

class criterion(nn.Module):

  def __init__(self):#, alpha=1.0, beta=0.1
    super(criterion, self).__init__()
    # self.alpha = alpha  # Weight for MSE component
    # self.beta = beta  # Weight for L1 penalty component
    self.si_sdr = SI_SDR_loss()
    self.stft_mag_loss = STFTMagnitudeLoss(log=True, log_eps=1e-13)#log_eps=1e-8
    self.l1_loss = torch.nn.MSELoss()#torch.nn.L1Loss()#
    self.overlap_loss = OverlapLossTD()
    self.SNR_loss = OptSNRLossTD()
    self.det_SNR_loss = DetSNRLossTD()
    self.fft_mag_loss = FFT_mag_loss()

  def forward(self, predictions, targets, dt, plunge_idx=None):
    #Initialise total loss
    out = 0.0
    #Define weights
    if plunge_idx is None:
        weights=1.0
    else:
        weights = torch.ones_like(predictions)
        weights[plunge_idx:] = 0.0
    #Add waveform loss
    # out = out + 1.0 * self.masked_mse_loss(predictions, targets, plunge_idx)
    out = out + 10.0 * self.l1_loss(predictions, targets)#1.0#weights*
    #Add SI-SDR loss
    '''weighting the prediction and target'''
    # out = out + 0.0 * self.si_sdr(weights*predictions, weights*targets)#0.01
    # Add overlap loss
    '''Data is whitened so PSD argument set to 1.0'''
    out = out + 1.0 * self.overlap_loss(predictions, targets, dt, 1.0)#0.1
    # Add FFT loss
    out = out + 0.1 * self.fft_mag_loss(predictions, targets)
    # Add variance loss
    '''Potential improvement could be to constrain the variance calculation
    to the region where the signal actually is i.e. before plunge'''
    var_preds = self.masked_var(predictions, plunge_idx, dim=-1, unbiased=True, eps=1e-8)
    var_targets = self.masked_var(targets, plunge_idx, dim=-1, unbiased=True, eps=1e-8)
    out = out + 1.0 * self.l1_loss(var_preds, var_targets)
    #out = out + self.l1_loss(torch.var(predictions[...,:plunge_idx], dim=-1), torch.var(targets[...,:plunge_idx], dim=-1))
    # Add det SNR loss
    # out = out + self.det_SNR_loss(predictions, targets, dt, 1.0)
    # Add SNR loss
    # out = out + self.SNR_loss(predictions, targets, dt, 1.0)
    # Add stft loss
    # '''Have to reshape data prior to stft'''
    # b, c, l = predictions.shape
    # targets_flat = targets.reshape(-1, l)
    # predictions_flat = predictions.reshape(-1, l)
    # '''STFT the flattened data'''
    # pred_mag= torch.stft(predictions_flat, 512, return_complex=True).abs()#512 bins
    # target_mag= torch.stft(targets_flat, 512, return_complex=True).abs()#512 bins
    # '''Then unflatten'''
    # target_mag = target_mag.unflatten(0, (b, c))
    # pred_mag = pred_mag.unflatten(0, (b, c))
    # loss_stft_mag = self.stft_mag_loss(pred_mag, target_mag)
    # out = out + 0.01 * loss_stft_mag
    return out
  def masked_var(self, x, plunge_idx, dim=-1, unbiased=True, eps=1e-8):
    """
        x: Tensor of shape (B, C, L)
        mask: Boolean or float tensor broadcastable to x (e.g., (B, 1, L))
        """
    #Create mask
    mask = torch.ones_like(x)
    mask[:,:,plunge_idx:] = 0.0

    # Count valid elements in the region
    count = mask.sum(dim=dim, keepdim=True)

    # 1. Calculate masked mean
    mean = (x * mask).sum(dim=dim, keepdim=True) / (count + eps)

    # 2. Calculate squared deviations, zeroing out elements outside the mask
    sq_diff = ((x - mean) ** 2) * mask

    # 3. Divide by N (biased) or N - 1 (unbiased)
    denom = (count - 1).clamp(min=1) if unbiased else count.clamp(min=1)
    variance = sq_diff.sum(dim=dim) / (denom.squeeze(dim) + eps)

    return variance
  def masked_mse_loss(self, pred: torch.Tensor, target: torch.Tensor, plunge_idx: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    #Create mask
    mask = torch.ones_like(pred)
    mask[:,:,plunge_idx:] = 0.0
    
    # Compute element-wise squared error
    squared_error = F.mse_loss(pred, target, reduction='none')
    
    # Zero out masked positions and normalize by the count of valid elements
    masked_error = squared_error * mask
    return masked_error.sum() / (mask.sum() + eps)

class FFT_mag_loss(torch.nn.Module):
    """
    """

    def __init__(self, eps=1e-4):#,  reduction="mean"
        super(FFT_mag_loss, self).__init__()
        self.eps = eps
        # self.reduction = reduction

    def forward(self, input, target):
        #Convert input and target to float64 precision
        input_64= input#.double()
        target_64= target#.double()
        #Calculate FFTs
        input_fft = torch.fft.rfft(input_64)#FFT_AET(input_64, zero_pad_data=True, window=None, fft_out=None, axis=-1)
        target_fft = torch.fft.rfft(target_64)#FFT_AET(target_64, zero_pad_data=True, window=None, fft_out=None, axis=-1)
        #Calculate log mags
        input_logmag = torch.log(input_fft.abs() + self.eps)
        target_logmag = torch.log(target_fft.abs() + self.eps)
        #MSE between the log magnitude
        losses = F.mse_loss(input_logmag, target_logmag, reduction='none')
        '''Optionally, we could apply a binary weighting to only consider frequencies within what we're interested in.'''
        #Reduce the loss
        losses = torch.mean(losses)
        return losses


class SI_SDR_loss(nn.Module):
    '''Computes the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).'''
    def __init__(self, eps: float = 1e-8, zero_mean=False) -> None:
        super(SI_SDR_loss, self).__init__()
        self.eps = eps
        self.zero_mean = zero_mean
    def si_sdr(self, preds, target):#, weights=None
        '''Returns SI-SDR.'''

        # 1. Zero-mean the signals
        if self.zero_mean:
            '''The predictions are clean so it may not make sense to subtract the mean!'''
            target = target - torch.mean(target, dim=(-1), keepdim=True)
            preds = preds - torch.mean(preds, dim=(-1), keepdim=True)

        # 2. Project target onto preds to find the scale (alpha)
        # alpha = <preds, target> / ||target||^2
        alpha = (preds * target).sum((-1)) / (((target ** 2).sum((-1))) + self.eps)
        target = target * alpha.unsqueeze(-1)
        res = preds - target
        out = 10 * torch.log10(
            (target ** 2).sum((-1)) / ((res ** 2).sum((-1)) + self.eps) + self.eps
        )
        return out
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate SI-SDR
        si_sdr = self.si_sdr(preds, target)
        # Return negative for minimization
        return -torch.mean(si_sdr)

'''Borrowed from auraloss'''
class STFTMagnitudeLoss(torch.nn.Module):
    """STFT magnitude loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719)
    and [Engel et al., 2020](https://arxiv.org/abs/2001.04643v1)

    Log-magnitudes are calculated with `log(log_fac*x + log_eps)`, where `log_fac` controls the
    compression strength (larger value results in more compression), and `log_eps` can be used
    to control the range of the compressed output values (e.g., `log_eps>=1` ensures positive
    output values). The default values `log_fac=1` and `log_eps=0` correspond to plain log-compression.

    Args:
        log (bool, optional): Log-scale the STFT magnitudes,
            or use linear scale. Default: True
        log_eps (float, optional): Constant value added to the magnitudes before evaluating the logarithm.
            Default: 0.0
        log_fac (float, optional): Constant multiplication factor for the magnitudes before evaluating the logarithm.
            Default: 1.0
        distance (str, optional): Distance function ["L1", "L2"]. Default: "L1"
        reduction (str, optional): Reduction of the loss elements. Default: "mean"
    """

    def __init__(self, log=True, log_eps=0.0, log_fac=1.0, distance="L1", reduction="mean"):
        super(STFTMagnitudeLoss, self).__init__()

        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac

        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            x_mag = torch.log(self.log_fac * x_mag + self.log_eps)
            y_mag = torch.log(self.log_fac * y_mag + self.log_eps)
        return self.distance(x_mag, y_mag)

class OverlapLossTD(torch.nn.Module):
    """
    Time domain mismatch as a loss function.
    NOTE: overlap and mismatch are bounded between [-1,1] NOT [0,1]
    because you can have anticorrelated signals producing negative overlaps.
    """

    def __init__(self, eps=1e-4):#,  reduction="mean"
        super(OverlapLossTD, self).__init__()
        self.eps = eps
        # self.reduction = reduction

    def forward(self, input, target, dt, PSD):
        '''If the data is whitened, set PSD argument to 1.0.'''
        '''Also, this may be better done in float64 precision!'''
        #Convert input and target to float64 precision
        input_64= input.double()
        target_64= target.double()
        #Calculate required inner products
        inner_prod_h1_h2= inner_prod_AET_batchwise(input_64, target_64, dt, PSD)
        inner_prod_h1_h1= inner_prod_AET_batchwise(input_64, input_64, dt, PSD)
        inner_prod_h2_h2= inner_prod_AET_batchwise(target_64, target_64, dt, PSD)
        #Calculate the denominator of the overlap
        '''This needs to be computed carefully to avoid overflows'''
        denom = torch.rsqrt(inner_prod_h1_h1 + self.eps) * torch.rsqrt(inner_prod_h2_h2 + self.eps)
        #Then calculate the channel-wise overlap
        '''Let's try the network overlap again. But first, confirm the correct
        formula'''
        overlap= (inner_prod_h1_h2) * denom
        # overlap= (inner_prod_h1_h2) * torch.rsqrt(inner_prod_h1_h1 * inner_prod_h2_h2)#+ self.eps
        # #Reduce to network overlap
        # n_chans=input.shape[-2]
        # network_overlap = torch.sqrt(torch.sum(overlap**2, dim=-1)/n_chans)
        #xp.sqrt(xp.sum(overlap**2)/n_chans)
        #Reduce across batch
        losses = 1 - overlap# 1 - network_overlap#
        #Reduce the loss
        losses = torch.mean(losses)
        return losses

class OptSNRLossTD(torch.nn.Module):
    """
    Optimal SNR as a loss function. (h_pred|h_pred) = (h_true|h_true)
    """

    def __init__(self, eps=1e-4):#,  reduction="mean"
        super(OptSNRLossTD, self).__init__()
        self.eps = eps
        # self.reduction = reduction

    def forward(self, input, target, dt, PSD):
        '''If the data is whitened, set PSD argument to 1.0.'''
        '''Also, this may be better done in float64 precision!'''
        #Calculate required inner products
        # inner_prod_h1_h2= inner_prod_AET_batchwise(input, target, dt, PSD)
        inner_prod_h1_h1= inner_prod_AET_batchwise(input, input, dt, PSD)
        inner_prod_h2_h2= inner_prod_AET_batchwise(target, target, dt, PSD)
        #Then calculate
        losses = torch.abs(torch.sqrt(inner_prod_h1_h1) - torch.sqrt(inner_prod_h2_h2))
        #Reduce the loss
        losses = torch.mean(losses)
        return losses

class DetSNRLossTD(torch.nn.Module):
    """
    Det SNR as a loss function. Minimise (det SNR - opt SNR)
    """

    def __init__(self, eps=1e-4):#,  reduction="mean"
        super(DetSNRLossTD, self).__init__()
        self.eps = eps
        # self.reduction = reduction

    def forward(self, input, target, dt, PSD):
        '''If the data is whitened, set PSD argument to 1.0.'''
        '''Also, this may be better done in float64 precision!'''
        #Calculate required inner products
        inner_prod_h1_h2= inner_prod_AET_batchwise(input, target, dt, PSD)
        inner_prod_h1_h1= inner_prod_AET_batchwise(input, input, dt, PSD)
        inner_prod_h2_h2= inner_prod_AET_batchwise(target, target, dt, PSD)
        #Calculate opt SNR and Det SNR
        opt_SNR = torch.sqrt(inner_prod_h2_h2)
        det_SNR = inner_prod_h1_h2 / opt_SNR
        #Minimise the difference
        losses = torch.abs(det_SNR - opt_SNR)
        #Reduce the loss
        losses = torch.mean(losses)
        return losses

#-----------------------------OLD STUFF

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    '''Returns charbonnier loss between prediction and target without any reduction.'''
    return torch.sqrt(torch.pow((pred - target), 2) + eps)# - eps


#From LapSRN-PyTorch github repo
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, pred: torch.Tensor, target: torch.Tensor, region_weighting = None, sample_weighting = None, normalise=False) -> torch.Tensor:
        #Compute unreduced and unweighted loss
        loss = charbonnier_loss(pred, target, eps= self.eps)#torch.sqrt(torch.pow((pred - target), 2) + self.eps)
        if region_weighting is not None:
            loss = region_weighting * loss
        if sample_weighting is not None:
            loss = sample_weighting * loss
        loss = torch.mean(loss)
        if normalise:
            norm = torch.mean(torch.sqrt(torch.pow((target), 2) + self.eps))
            loss = loss / (norm + self.eps)#torch.std(torch.abs(target))#(torch.mean(torch.abs(target)) + self.eps)
        # '''Here, we are reducing by summing rather than averaging!!!'''
        # loss = torch.sum(torch.sqrt(torch.pow((pred - target), 2) + self.eps), dim=(-1,-2,-3))
        # loss = loss.mean()
        return loss

#Adapted from LapSRN-PyTorch github repo
class SubBandCharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super(SubBandCharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, pred: torch.Tensor, target: torch.Tensor, region_weighting = None, sample_weighting = None, normalise=False) -> torch.Tensor:
        loss = charbonnier_loss(pred, target, eps= self.eps)#torch.sqrt(torch.pow((pred - target), 2) + self.eps)
        if region_weighting is not None:
            loss = region_weighting * loss
        if sample_weighting is not None:
            loss = sample_weighting * loss
        '''#Reduce by summation across all channels except batch
        loss = torch.sum(loss, dim=(1, 2, 3))
        #Reduce by averaging across batches
        loss = torch.mean(loss)'''
        #Reduce over time and channel axes
        loss = torch.mean(loss, (1,3))
        #Then average over all frequency bins
        loss = torch.mean(loss)#torch.sum(loss)#torch.mean(loss)#torch.sum(loss)
        if normalise:
            norm = torch.mean(torch.sqrt(torch.pow((target), 2) + self.eps))
            loss = loss / (norm + self.eps)#torch.std(torch.abs(target))#(torch.mean(torch.abs(target)) + self.eps)
        # '''Here, we are reducing by summing rather than averaging!!!'''
        # loss = torch.sum(torch.sqrt(torch.pow((pred - target), 2) + self.eps), dim=(-1,-2,-3))
        # loss = loss.mean()
        return loss


class LogMagnitudeL1Loss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(LogMagnitudeL1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x_mag, y_mag, region_weighting = None):
        """
        Computes the L1 distance between log-magnitudes of predictions and targets.
        
        Args:
            x_mag (torch.Tensor): Predicted magnitude spectrogram.
            y_mag (torch.Tensor): Target magnitude spectrogram.
            
        Returns:
            torch.Tensor: The mean log-magnitude L1 loss.
        """
        #Cast objects to float64 precision
        input_64 = x_mag.double()
        target_64 = y_mag.double()
        #Calculate each scaled modulus
        abs_input = torch.abs(input_64)#input, self.scale_factor*
        abs_target = torch.abs(target_64)#target, self.scale_factor*
        #Calculate logs separately to avoid product underflow (input * target)
        # log|input * target| = log|input| + log|target|
        # log|target * target| = 2 * log|target|
        log_y = torch.log(abs_target.clamp(min=self.epsilon))#, min=1.0
        log_x = torch.log(abs_input.clamp(min=self.epsilon))#, min=1.0

        # Add epsilon to prevent log(0) and stabilize the gradient
        # log_x = torch.log(x_mag + self.epsilon)
        # log_y = torch.log(y_mag + self.epsilon)
        
        # Calculate the L1 loss on the log-magnitudes
        '''We could try switching this to Charbonnier loss.'''
        loss = charbonnier_loss(log_x, log_y, eps = 1e-6)
        
        if region_weighting is not None:
            loss = region_weighting * loss

        loss = torch.mean(loss)

        #loss = F.l1_loss(log_x, log_y, weight = region_weighting)#, reduction='none'

        # #Reduce by summation over all axes except batch size
        # loss = torch.sum(loss, (1,2,3))

        # #Reduce by averaging over batch size
        # loss = torch.mean(loss)
        
        #Cast to float32 precision
        return loss.float()

class OverlapLoss(torch.nn.Module):
    """
    Wavelet domain mismatch as a loss function.
    NOTE: overlap and mismatch are bounded between [-1,1] NOT [0,1]
    because you can have anticorrelated signals producing negative overlaps.
    """

    def __init__(self, eps=1e-4):#,  reduction="mean"
        super(OverlapLoss, self).__init__()
        self.eps = eps
        # self.reduction = reduction

    def forward(self, input, target, region_weighting = None):
        '''BIG NOTE: we are doing some of this in float64 precision instead!!!!'''
        #Cast objects to float64 precision
        input_64 = input.double()
        target_64 = target.double()
        if region_weighting is None:
            weights = 1.0
        else:
            # Match the weighting tensor precision to float64 as well
            weights = region_weighting.double()
                
        input_dot_target = torch.nansum(weights*input_64*target_64, dim=(-2,-1))#input*target
        target_dot_target = torch.nansum(weights*target_64*target_64, dim=(-2,-1))#target*target
        input_dot_input = torch.nansum(weights*input_64*input_64, dim=(-2,-1))#input*input
        #Calculate denominator terms individually
        input_norm  = torch.sqrt(input_dot_input + self.eps)#torch.sqrt(torch.clamp(input_dot_input, min=self.eps))
        target_norm = torch.sqrt(target_dot_target + self.eps)#torch.sqrt(torch.clamp(target_dot_target, min=self.eps))
        #Calculate the match
        match = input_dot_target / (input_norm * target_norm)#input_dot_target * torch.rsqrt(target_dot_target * input_dot_input + self.eps)
        #Clamp match in case of rounding errors
        match = torch.clamp(match, -1.0, 1.0)
        losses = 1 - match#(1 - match)**2#
        losses = torch.mean(losses)
        #Finally, cast the loss back to float32 precision
        return losses.float()
    
class DetSNRLoss(torch.nn.Module):
    """
    Detected SNR as a loss function.
    Charbonnier loss between the detected SNR and optimal SNR
    """

    def __init__(self, eps=1e-8):#,  reduction="mean"
        super(DetSNRLoss, self).__init__()
        self.eps = eps
        # self.reduction = reduction

    def forward(self, input, target):
        opt_SNR = torch.sqrt(torch.nansum((target * target), dim=(-2,-1)))
        det_SNR = torch.nansum((input * target), dim=(-2,-1)) / (opt_SNR + self.eps)
        losses = torch.mean(torch.sqrt(torch.pow(det_SNR - opt_SNR, 2) + self.eps))
        return losses

class CrossWaveletPowerLoss(torch.nn.Module):
    """
    XWT power as a loss function.
    Charbonnier loss between the log|w_xy| and log|w_yy|
    eps: a floor on the log|w_xy|
    """

    def __init__(self, scale_factor=1.0, eps=1e-8):#,  reduction="mean"
        super(CrossWaveletPowerLoss, self).__init__()
        self.eps = eps
        self.scale_factor = scale_factor
        # self.reduction = reduction

    def forward(self, input, target):
        '''The below logs can go to -inf if epsilon is too small.
        Need to be careful (potentially adaptive) with the choice of eps'''
        '''We can also try clamping with an epsilon instead of adding.'''
        #Cast objects to float64 precision
        input_64 = input.double()
        target_64 = target.double()
        #Calculate each scaled modulus
        abs_input = torch.abs(input_64)#input, self.scale_factor*
        abs_target = torch.abs(target_64)#target, self.scale_factor*
        #Calculate logs separately to avoid product underflow (input * target)
        # log|input * target| = log|input| + log|target|
        # log|target * target| = 2 * log|target|
        log_abs_target = torch.log(abs_target.clamp(min=self.eps))#, min=1.0
        log_abs_input = torch.log(abs_input.clamp(min=self.eps))#, min=1.0
        log_XWT_power_yy = 2.0 * log_abs_target
        log_XWT_power_xy = log_abs_input + log_abs_target
        # log_XWT_power_yy = torch.log(torch.abs(target * target) + self.eps)
        # log_XWT_power_xy = torch.log(torch.abs(input * target) + self.eps)
        #Compute Charbonnier loss between each XWT power
        '''Different epsilon for the charbonnier loss since we're already comparing logged values at that point'''
        charbonnier_eps = 1e-6
        '''The below can produce nans if the logs have gone to -inf.
        Not a problem with the loss between the input and targets.'''
        XWT_diff = charbonnier_loss(log_XWT_power_xy, log_XWT_power_yy, eps = charbonnier_eps)#torch.sqrt(torch.pow(torch.log(XWT_power_yy + self.eps) - torch.log(XWT_power_xy + self.eps), 2) + self.eps)
        #Reduction by averaging
        '''Averaging losses that contain NaNs will obviously produce a nan, hence
        a final loss of Nan. So maybe try nanmean'''
        losses = torch.mean(XWT_diff)
        #Finally, cast to float32 precision
        return losses.float()


class OptSNRLoss(torch.nn.Module):
    """
    Optimal SNR as a loss function.
    Inner product of the difference between prediction and target.
    """

    def __init__(self, eps=1e-8):#,  reduction="mean"
        super(OptSNRLoss, self).__init__()
        self.eps = eps
        # self.reduction = reduction

    def forward(self, input, target):
        residual = target - input
        inn_prod = torch.nansum(residual*residual, dim=(-2,-1))
        losses = torch.mean(inn_prod)
        return losses


class CorrelationLoss(torch.nn.Module):
    """
    pearson correlation as a loss function.
    Bounded between -1 and 1.
    """

    def __init__(self, eps=1e-8):#,  reduction="mean"
        super(CorrelationLoss, self).__init__()
        self.eps = eps
        # self.reduction = reduction
    
    def pearson_correlation(self, x, y):
        mean_x = torch.mean(x, dim=(-2,-1), keepdim=True)
        mean_y = torch.mean(y, dim=(-2,-1), keepdim=True)
        xm = x - mean_x
        ym = y - mean_y
        # Use torch.sum for the numerator (covariance)
        cov = torch.nansum(xm * ym, dim=(-2,-1))
        # Use torch.norm for stability and avoiding potential NaNs with sqrt(sum(...))
        # Add a small epsilon for numerical stability in the denominator
        norm_xm = torch.linalg.matrix_norm(xm, dim=(-2,-1))
        norm_ym = torch.linalg.matrix_norm(ym, dim=(-2,-1))
        denominator = norm_xm * norm_ym + self.eps
        pcc = cov / denominator
        return pcc

    def forward(self, input, target):
        losses = 1 - self.pearson_correlation(input, target)
        losses = torch.mean(losses)
        return losses

#Borrowed from https://github.com/mmany/pytorch-GDL/blob/main/custom_loss_functions.py
class GradientDifferenceLoss(nn.Module):
    '''Configured for data that is to be logged and modulus'd'''
    def __init__(self, weight=None, size_average=True, alpha=1.0, eps=1e-4):
        super(GradientDifferenceLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, inputs, targets):
        #Convert input to float64
        input_64 = inputs.double()
        target_64 = targets.double()
        #Log and modulus to eliminate extreme variance and negative values
        input_64 = torch.log(input_64.abs().clamp(min=self.eps))
        target_64 = torch.log(target_64.abs().clamp(min=self.eps))
        # gradient_diff = (inputs.diff(dim=-2, prepend=torch.tensor(0.0))-targets.diff(dim=-2, prepend=torch.tensor(0.0))).pow(2) + (inputs.diff(dim=-1, prepend=torch.tensor(0.0))-targets.diff(dim=-1, prepend=torch.tensor(0.0))).pow(2)
        term_1 = torch.diff(input_64, dim=-2).abs() - torch.diff(target_64, dim=-2).abs() #(inputs.diff(dim=-2)-targets.diff(dim=-2)).pow(2)
        term_2 = torch.diff(input_64, dim=-1).abs() - torch.diff(target_64, dim=-1).abs()  #(inputs.diff(dim=-1)-targets.diff(dim=-1)).pow(2)
        gradient_diff = torch.abs(term_1[:,:,:,:-1])**self.alpha + torch.abs(term_2[:,:,:-1,:])**self.alpha #term_1[:,:,:,:-1] + term_2[:,:,:-1,:]
        loss_gdl = torch.mean(gradient_diff)#gradient_diff.sum()/inputs.numel()#Maybe should be averaged just over batch size?
        #Finally cast back to float32 precision
        return loss_gdl.float()

# import math
# import torch
# from torch import autograd as autograd
# from torch import nn as nn
# from torch.nn import functional as F

# # # from basicsr.archs.vgg_arch import VGGFeatureExtractor
# # from basicsr.utils.registry import LOSS_REGISTRY
# # from .loss_util import weighted_loss

# # _reduction_modes = ['none', 'mean', 'sum']

# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

# class CharbonnierLoss(nn.Module):
#     """Charbonnier loss (one variant of Robust L1Loss, a differentiable

#     variant of L1Loss).



#     Described in "Deep Laplacian Pyramid Networks for Fast and Accurate

#         Super-Resolution".



#     Args:

#         loss_weight (float): Loss weight for L1 loss. Default: 1.0.

#         reduction (str): Specifies the reduction to apply to the output.

#             Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.

#         eps (float): A value used to control the curvature near zero. Default: 1e-12.

#     """

#     def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
#         super(CharbonnierLoss, self).__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

#         self.loss_weight = loss_weight
#         self.reduction = reduction
#         self.eps = eps

#     def forward(self, pred, target,  **kwargs):#weight=None,
#         """

#         Args:

#             pred (Tensor): of shape (N, C, H, W). Predicted tensor.

#             target (Tensor): of shape (N, C, H, W). Ground truth tensor.

#             weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.

#         """
#         return self.loss_weight * charbonnier_loss(pred, target, eps=self.eps, reduction=self.reduction)#weight,
