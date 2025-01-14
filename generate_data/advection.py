import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap
import argparse

P = 20
M = 252 # M+1 spatial points
N = 378 # N+1 time points 
NUMBER_OF_SENSORS = 64

def sech(x):
    return 1/jnp.cosh(x)

def u_soliton(x, t, key = random.key(0), P=20):    
    key_cs, key_ds = random.split(key, 2)
    c = random.uniform(key_cs, minval=0.5, maxval=1.5, shape=())
    d = random.uniform(key_ds, minval=0, maxval=1, shape=())
    
    return 2 * c**2 * sech(c * ((x-c*t+P/2-P*d) % P - P/2))**2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate advection data")
    parser.add_argument("--running_on", type=str, choices=["local", "idun"], default="local", help="Specify where the code is running")
    running_on = parser.parse_args().running_on
    
    if running_on=="local":
        PATH = "C:/Users/eirik/OneDrive - NTNU/5. klasse/prosjektoppgave/eirik_prosjektoppgave/data/"
    elif running_on=="idun":
        PATH = "/cluster/work/eirikaf/data/"
    else:
        raise ValueError("running_on must be either 'local' or 'idun'")
    
    x = jnp.linspace(0, 20, 256, endpoint=False)
    t = jnp.linspace(0, 3, (256*3)//2, endpoint=False)

    NUM_SAMPLES = 1000

    keys = random.split(random.key(0), NUM_SAMPLES)
    data = vmap(vmap(u_soliton, (None, 0, None)), (None, None, 0))(x,t,keys)

    jnp.savez(PATH + "advection", x=x, t=t, data=data)