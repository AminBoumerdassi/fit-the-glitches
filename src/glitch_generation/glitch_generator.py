import jax.numpy as jnp
import jaxglitches as jg
from jax import random

from jaxglitches.catalog_generator import *
from jaxglitches.data import *

import matplotlib.pyplot as plt
import numpy as np

class GlitchBackgroundGenerator():
    def __init__(self, t_obs, dt=10.0, tdi=2, basis = "AET")
    self.tdi=tdi
    self.basis=basis
    self.t_obs = t_obs
    self.dt = dt
    #Initialise array of t
    self.t = jnp.arange(0, self.t_obs, self.dt)
    #Initialise the batched version of clean_signal_t
    self.generate_many_glitches = jax.vmap(
    lambda params: clean_signal_t(
        params,
        self.t,
        tdi = self.tdi,
        basis = self.basis,
    ))
    @jax.jit
    def generate_glitch_background(self, params):
        return generate_many_glitches(params.swapaxes(0,1)).sum(axis=0)


