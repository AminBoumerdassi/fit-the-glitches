import jax.numpy as jnp
import jaxglitches as jg
from jax import random
from jax import jit

from jaxglitches.catalog_generator import *
from jaxglitches.data import *

import matplotlib.pyplot as plt
import numpy as np

'''
#Can force the backend like so
import os

os.environ["JAX_PLATFORMS"] = "cpu"  # or "gpu", "tpu"
'''

# create a root PRNG key for reproducibility
key = random.key(0)

#Choose the batch size
no_backgrounds=50

#set parameters
dt=10.0#s
t_obs= 0.3 * 365 * 86400.#in seconds
t= jnp.arange(0, t_obs, dt)

#Define the batched background generation
generate_many_glitches = jax.vmap(
    lambda params: clean_signal_t(
        params,
        t,
        tdi = 2,
        basis = "AET",
    ))

@jax.jit
def generate_glitch_background(params):
    return generate_many_glitches(params.swapaxes(0,1)).sum(axis=0)

#run 10 times
for k in range(no_backgrounds):
    print(f"BG {k}")
    # New subkey
    subkey = jax.random.split(key)
    # Generate catalogue
    catalog = run_catalog(t_obs, key=subkey[0], run_type = "ordinary", rate=1/86400,
                    smooth = True, signed = True,
                    sort = True)#inj_points=INJ_POINTS
    # Generate background
    background = generate_glitch_background(jnp.array([catalog["t0"],catalog["Deltav"],catalog["tau"]]))
    # background = jnp.zeros((t.shape[0],3))
    # for i in range(catalog["n_glitches"]):
    #     params = jnp.array([catalog["t0"][i],catalog["Deltav"][i],catalog["tau"][i]])
    #     background += clean_signal_t(params, t, tdi = 2, basis = "AET")#T: float = T_ARM_s,

#plot
plt.figure()
plt.plot(np.asarray(t), np.asarray(background[:,0]))
plt.savefig("test.png")