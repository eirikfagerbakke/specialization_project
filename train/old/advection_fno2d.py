import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import equinox as eqx
import jax.numpy as jnp
from jax import random
import optax 
from optax.contrib import reduce_on_plateau
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

u_test_n = u_normalizer.encode(u_test)
a_test_n = a_normalizer.encode(a_test)

x_n = x_normalizer.encode(x)
t_n = t_normalizer.encode(t)

# Pre-repeat data 
a_train_n = jnp.repeat(a_train_n[:, None, :], N+1, axis=1)
a_val_n = jnp.repeat(a_val_n[:, None, :], N+1, axis=1)
a_test_n = jnp.repeat(a_test_n[:, None, :], N+1, axis=1)

x_n = jnp.repeat(x_n[None, :], N+1, axis=0)
t_n = jnp.repeat(t_n[:, None], M+1, axis=1)

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

    u_pred = vmap(model, (0, None, None))(a, Trainer.x, Trainer.t)
    
    #mse loss
    if model.self_adaptive:
        loss = jnp.mean(model.self_adaptive() * jnp.square((u_pred - u))) #MSE
    else:
        loss = jnp.mean(jnp.square(u_pred - u))

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
devices = mesh_utils.create_device_mesh((num_devices, 1, 1))
sharding = jshard.PositionalSharding(devices)
replicated = sharding.replicate()

# Set sharding and replicated as class attributes, since they are constant throughout the training
Trainer.sharding_a = sharding
Trainer.sharding_u = sharding
Trainer.replicated = replicated

# Dataloaders
train_loader = jdl.DataLoader(jdl.ArrayDataset(a_train_n, u_train_n), batch_size=16, shuffle=True, backend='jax', drop_last=True)
val_loader = jdl.DataLoader(jdl.ArrayDataset(a_val_n, u_val_n), batch_size=16, shuffle=True, backend='jax', drop_last=True)

# Define hyperparameters
hparams = fno2d.Hparams(n_blocks=4,
                        hidden_dim=100,
                        modes_max=10,)

# Define model
model = FNO2d(hparams)
model = eqx.filter_shard(model, replicated)

# Dataloaders
train_loader = jdl.DataLoader(jdl.ArrayDataset(a_train_n, u_train_n), batch_size=16, shuffle=True, backend='jax', drop_last=True)
val_loader = jdl.DataLoader(jdl.ArrayDataset(a_val_n, u_val_n), batch_size=16, shuffle=True, backend='jax', drop_last=True)

# OPTIMIZER 
PATIENCE = 5 # Number of epochs with no improvement after which learning rate will be reduced
COOLDOWN = 0 # Number of epochs to wait before resuming normal operation after the learning rate reduction
FACTOR = 0.5  # Factor by which to reduce the learning rate:
RTOL = 1e-4  # Relative tolerance for measuring the new optimum:
ACCUMULATION_SIZE = 200 # Number of iterations to accumulate an average value:
LEARNING_RATE = 1e-3

θ_learning_rate = 1e-3

opt = optax.chain(
    optax.adam(θ_learning_rate),
    reduce_on_plateau(
        patience=PATIENCE,
        cooldown=COOLDOWN,
        factor=FACTOR,
        rtol=RTOL,
        accumulation_size=ACCUMULATION_SIZE,
    ),
)

opt_state = opt.init(eqx.filter([model], eqx.is_array))

trainer = Trainer(model, 
                  opt, 
                  opt_state, 
                  train_loader, 
                  val_loader, 
                  hparams = hparams, 
                  animate=True,
                  save_path=PATH)
trainer(10)