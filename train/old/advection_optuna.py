import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import equinox as eqx
import jax.numpy as jnp
from jax import random
import optax 
from jax import vmap
import jax_dataloader as jdl
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
#jax.config.update("jax_enable_x64", True)

from networks import *
from utils import *

import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as jshard

#PATH = "/cluster/work/eirikaf/"
PATH = r"C:\Users\eirik\orbax"
#PATH = None

# LOAD DATA
#data = jnp.load("/cluster/home/eirikaf/phlearn-summer24/eirik_prosjektoppgave/data/advection.npz")
data = jnp.load(r"C:\Users\eirik\OneDrive - NTNU\5. klasse\prosjektoppgave\eirik_prosjektoppgave\data\advection.npz")
x = jnp.array(data['x'])
t = jnp.array(data['t'])
a_train = jnp.array(data['a_train'])
u_train = jnp.array(data['u_train'])
a_val = jnp.array(data['a_val'])
u_val = jnp.array(data['u_val'])
a_test = jnp.array(data['a_test'])
u_test = jnp.array(data['u_test'])

# SET PARAMETERS
P = x[-1]
T = t[-1]
M = len(x) - 1
N = len(t) - 1
NUMBER_OF_SENSORS = M + 1
N = len(t)-1

# NORMALIZE DATA
u_normalizer = UnitGaussianNormalizer(u_train)
a_normalizer = UnitGaussianNormalizer(a_train)
x_normalizer = UnitGaussianNormalizer(x)
t_normalizer = UnitGaussianNormalizer(t)

u_train_n = u_normalizer.encode(u_train)
a_train_n = a_normalizer.encode(a_train)

u_val_n = u_normalizer.encode(u_val)
a_val_n = a_normalizer.encode(a_val)

x_n = x_normalizer.encode(x)
t_n = t_normalizer.encode(t)

# Set x and t as class attributes, since they are constant throughout the training
Trainer.x = x_n
Trainer.t = t_n

def compute_loss(model, a, u, key, num_query_points=100):
    """Computes the loss of the model.
    Returns the l2 loss, averaged over the batch. The loss is computed by randomly selecting query points from the input data and evaluating the model at those points.

    Args:
        model (eqx.Module): The model to evaluate.
        a (batch, number_of_sensors): The input data.
        u (batch, num_query_points): The ground truth data at the query points.

    Returns:
        loss (scalar): The loss of the model for the given batch.
    """
    batch_size, Np1, Mp1 = u.shape
            
    # Select random query indices
    t_key, x_key = random.split(key, 2)
    t_idx = random.randint(t_key, (batch_size, num_query_points), 0, Np1)
    x_idx = random.randint(x_key, (batch_size, num_query_points), 0, Mp1)

    # Select the ground truth data at the query points
    # Has shape (batch_size, num_query_points)
    u_at_query_points = u[jnp.arange(batch_size)[:, None], t_idx, x_idx]
    
    # For each input function, compute the prediction of the model at the query points. (inner vmap)
    # Do this for each sample in the batch. (outer vmap)
    # Has shape (batch_size, num_query_points)
    u_pred = vmap(vmap(model, (None, 0, 0)))(a, Trainer.x[x_idx], Trainer.t[t_idx])
    
    #mse loss
    if model.self_adaptive:
        loss = jnp.mean(model.self_adaptive(t_idx, x_idx) * jnp.square((u_pred - u_at_query_points))) #MSE
    else:
        loss = jnp.mean(jnp.square(u_pred - u_at_query_points))

    return loss

Trainer.compute_loss = staticmethod(compute_loss)

@eqx.filter_jit(donate="all")
def make_step(model, opt_state, a, u, key):
    """Performs a single optimization step."""
    model, opt_state = eqx.filter_shard((model, opt_state), Trainer.replicated)
    a, u = eqx.filter_shard((a,u), (Trainer.sharding_a, Trainer.sharding_u))
    
    loss, grads = eqx.filter_value_and_grad(Trainer.compute_loss)(model, a, u, key)
    updates, opt_state = Trainer.opt.update([grads], opt_state, value=loss)
    model = eqx.apply_updates(model, updates[0])
    
    if model.self_adaptive:
        # normalize λ
        model = eqx.tree_at(lambda m : m.self_adaptive.λ, model, model.self_adaptive.λ/jnp.mean(model.self_adaptive.λ))        
    
    model, opt_state = eqx.filter_shard((model, opt_state), Trainer.replicated)
    
    return model, opt_state, loss

Trainer.make_step = staticmethod(make_step)

@eqx.filter_jit(donate="all-except-first")
def evaluate(model, a, u, key):
    model = eqx.filter_shard(model, Trainer.replicated)
    a, u = eqx.filter_shard((a,u), (Trainer.sharding_a, Trainer.sharding_u))
    return compute_loss(model, a, u, key)

Trainer.evaluate = staticmethod(evaluate)

# Create mesh for sharding (autoparallelism)
num_devices = len(jax.devices())
devices = mesh_utils.create_device_mesh((num_devices, 1))
sharding_u = jshard.PositionalSharding(devices).reshape(num_devices, 1, 1)
sharding_a = jshard.PositionalSharding(devices)
sharding = {"a": sharding_a, "u": sharding_u}
replicated = sharding_a.replicate()

# Set sharding and replicated as class attributes, since they are constant throughout the training
Trainer.sharding_a = sharding_a
Trainer.sharding_u = sharding_u
Trainer.replicated = replicated

# Dataloaders
train_loader = jdl.DataLoader(jdl.ArrayDataset(a_train_n, u_train_n), batch_size=16, shuffle=True, backend='jax', drop_last=True)
val_loader = jdl.DataLoader(jdl.ArrayDataset(a_val_n, u_val_n), batch_size=16, shuffle=True, backend='jax', drop_last=True)

num_trials = 10

study = optuna.create_study(
    study_name='test12345',
    storage=f"sqlite:///test_problems/optuna/optuna.db",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50),
    sampler = optuna.samplers.TPESampler(seed=2),
    load_if_exists=True,
)

with ProgressBar(mode = "trial") as p:
    p.pbar.update(p.trials_id, total = num_trials)
    study.optimize(modified_deeponet.HparamTuning(train_loader, 
                                                  val_loader, 
                                                  NUMBER_OF_SENSORS, 
                                                  (x_n[-1]-x_n[0]).item(),
                                                  (N+1, M+1),
                                                  max_epochs=30), 
                    n_trials=num_trials, callbacks = [lambda study, trial: p.pbar.advance(p.trials_id)])