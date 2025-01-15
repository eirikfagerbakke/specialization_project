import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import jax
#jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax import vmap
import networks
from utils import train_val_test_split, scale_data

import argparse

parser = argparse.ArgumentParser(description='Hyperparameter optimization with Optuna')
parser.add_argument('--problem', type=str, default='advection', choices=['advection', 'kdv'], help='Problem type')
args = parser.parse_args()

problem = args.problem

data_path = "C:/Users/eirik/OneDrive - NTNU/5. klasse/prosjektoppgave/eirik_prosjektoppgave/data/"

if problem == "advection":
    data = jnp.load(data_path + "advection.npz")
elif problem == "kdv":
    data = jnp.load(data_path + "kdv.npz")
else:
    raise ValueError("Invalid problem")

# SPLIT DATA
train_val_test = train_val_test_split(jnp.array(data['data']), 0.8, 0.1, 0.1)

# SCALE DATA
scaled_data = scale_data(jnp.array(data['x']), jnp.array(data['t']), train_val_test)

jnp.savez(data_path + problem + "_z_score.npz", **scaled_data)