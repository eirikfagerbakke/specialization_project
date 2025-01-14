import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import numpy as np
key = random.key(0)
import argparse
import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from traditional_solvers import Dx, Dxx, implicit_midpoint, gauss_legendre_4, gauss_legendre_6

###############INITIAL CONDITION####################

def sech(x): return 1/jnp.cosh(x)

def initial_condition_kdv(x, key, η=6., P=20):
    """
    Generate the initial condition for the Korteweg-de Vries (KdV) equation.
    Parameters:
        x (float or array-like) : A single point or array in the spatial domain. 
        key (jax.random.PRNGKey): The random key for generating random numbers.
        η (float, optional): The coefficient for the KdV equation. Default is 6.
        P (float, optional): The period of the spatial domain. Default is 20.
    Returns:
        array-like: The initial condition for the KdV equation.
    """
    
    key_cs, key_ds = random.split(key, 2)
    c1, c2 = random.uniform(key_cs, minval=0.5, maxval=1.5, shape=(2,))
    d1, d2 = random.uniform(key_ds, minval=0, maxval=1, shape=(2,))
    
    u0 = (-6./-η)*2 * c1**2 * sech(c1 * ((x+P/2-P*d1) % P - P/2))**2
    u0 += (-6./-η)*2 * c2**2 * sech(c2 * ((x+P/2-P*d2) % P - P/2))**2
    return u0

#################PARAMETERS####################

η = 6.0
γ = 1.0
P = 20 # period (and end of the domain)
M = 256 # M points on [0,P)
N = (256*3)//2 # N points on [0,T) 

t0 = 0.0 # initial time
t_final = 3.0 # end time

x0 = 0.0 # initial position

x = jnp.linspace(x0, P, M, endpoint=False) # domain
t = jnp.linspace(t0, t_final, N, endpoint=False) # time domain

dt = t_final / N # time step
dx = P / M # space step

args = {"η" : η, "γ": γ, "dx" : dx}
SPATIAL_ORDER = 6

#############EQUATION####################
def f(t, u, args):
    η, γ, dx = args["η"], args["γ"], args["dx"]
    return -Dx(η/2*u**2 + γ**2 * Dxx(u, dx, SPATIAL_ORDER), dx, SPATIAL_ORDER)

#############HAMILTONIANS################
def H_energy(u, args):
    η, γ, dx = args["η"], args["γ"], args["dx"]
    integrand = -η/6*u**3 + 0.5*γ**2*Dx(u, dx)**2
    return dx*jnp.sum(integrand)

def H_mass(u, args):
    return dx * jnp.sum(u)

def H_momentum(u, args):
    return dx * jnp.sum(u**2)


#############SOLVING####################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate kdv data")
    parser.add_argument("--running_on", type=str, choices=["local", "idun"], default="local", help="Specify where the code is running")
    running_on = parser.parse_args().running_on
    
    if running_on=="local":
        PATH = "C:/Users/eirik/OneDrive - NTNU/5. klasse/prosjektoppgave/eirik_prosjektoppgave/data/"
    elif running_on=="idun":
        PATH = "/cluster/work/eirikaf/data/"
    else:
        raise ValueError("running_on must be either 'local' or 'idun'")
    
    NUM_SAMPLES_PER_LOOP = 100 # Number of samples to generate per loop
    NUM_LOOPS = 10 # Number of loops
    NUM_SAMPLES = NUM_SAMPLES_PER_LOOP*NUM_LOOPS # Total number
    
    atol, rtol = 1e-12, 1e-12
    
    """
    data = jnp.empty((NUM_SAMPLES, N+1, M))
    
    key = random.key(0)
    for i in range(NUM_LOOPS):
        key, subkey = random.split(key)
        keys = random.split(subkey, NUM_SAMPLES_PER_LOOP)

        a = jax.vmap(initial_condition_kdv, (None, 0))(x, keys)

        data = data.at[i*NUM_SAMPLES_PER_LOOP:(i+1)*NUM_SAMPLES_PER_LOOP, ...].set(jax.vmap(gauss_legendre_6, (None, 0, None, None, None, None, None))(f, a, dt, t, args, rtol, atol))
    """
    keys = random.split(random.key(0), NUM_SAMPLES)
    a_s = jax.vmap(initial_condition_kdv, (None, 0))(x, keys)
    data = jax.lax.map(lambda a : gauss_legendre_6(f, a, dt, t, args, rtol, atol), a_s, batch_size=NUM_SAMPLES_PER_LOOP)
    
    np.savez_compressed(PATH + "kdv", x=x, t=t, data=data)