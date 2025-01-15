import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from utils import *

import jax
import equinox as eqx
import jax.numpy as jnp
from jax import random
import optax 
from jax import vmap
import jax_dataloader as jdl
from dataclasses import dataclass
import optuna
import matplotlib.pyplot as plt
optuna.logging.set_verbosity(optuna.logging.WARNING)
#jax.config.update("jax_enable_x64", True)

from networks import *
from utils import *

import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as jshard

PATH = "/cluster/work/eirikaf/"

data = jnp.load("/cluster/work/eirikaf/kdv.npz")
x = jnp.array(data['x'])
x = jnp.concatenate([x, jnp.array([20.])])
t = jnp.array(data['t'])
a_train = jnp.array(data['a_train'])
u_train = jnp.array(data['u_train'])
a_val = jnp.array(data['a_val'])
u_val = jnp.array(data['u_val'])
a_test = jnp.array(data['a_test'])
u_test = jnp.array(data['u_test'])

P = x[-1]
T = t[-1]
M = len(x) - 1
N = len(t) - 1
NUMBER_OF_SENSORS = M + 1
N = len(t)-1
NUMBER_OF_SENSORS = M+1

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
    u_shape = u.shape
    batch_size = u_shape[0]
    N = u_shape[1]-1
    M = u_shape[2]-1
            
    # Select random query indices
    t_key, x_key = random.split(key, 2)
    t_idx = random.randint(t_key, (batch_size, num_query_points), 0, N+1)
    x_idx = random.randint(x_key, (batch_size, num_query_points), 0, M+1)

    # Select the ground truth data at the query points
    # Has shape (batch_size, num_query_points)
    u_at_query_points = u[jnp.arange(batch_size)[:, None], t_idx, x_idx]
    
    # For each input function, compute the prediction of the model at the query points. (inner vmap)
    # Do this for each sample in the batch. (outer vmap)
    # Has shape (batch_size, num_query_points)
    u_pred = vmap(vmap(model, (None, 0, 0)))(a, x_n[x_idx], t_n[t_idx])
    
    #mse loss
    if model.self_adaptive:
        loss = jnp.mean(model.self_adaptive(t_idx, x_idx) * jnp.square((u_pred - u_at_query_points))) #MSE
       
    else:
        loss = jnp.mean(jnp.square(u_pred - u_at_query_points))

    return loss

Trainer.compute_loss = staticmethod(compute_loss)

@eqx.filter_jit(donate="all")
def make_step(model, opt_state, opt, a, u, key, sharding):
    """Performs a single optimization step."""
    sharding_a, sharding_u = sharding["a"], sharding["u"]
    
    replicated = sharding_a.replicate()
    model, opt_state = eqx.filter_shard((model, opt_state), replicated)
    a = eqx.filter_shard(a, sharding_a)
    u = eqx.filter_shard(u, sharding_u)
    
    loss, grads = eqx.filter_value_and_grad(Trainer.compute_loss)(model, a, u, key)
    updates, opt_state = opt.update([grads], opt_state)
    model = eqx.apply_updates(model, updates[0])
    
    if model.self_adaptive:
        # normalize λ
        model = eqx.tree_at(lambda m : m.self_adaptive.λ, model, model.self_adaptive.λ/jnp.mean(model.self_adaptive.λ))        
    
    model, opt_state = eqx.filter_shard((model, opt_state), replicated)
    
    return model, opt_state, loss

Trainer.make_step = staticmethod(make_step)

num_devices = len(jax.devices())
devices = mesh_utils.create_device_mesh((num_devices, 1))
sharding_u = jshard.PositionalSharding(devices).reshape(num_devices, 1, 1)
sharding_a = jshard.PositionalSharding(devices)
sharding = {"a": sharding_a, "u": sharding_u}
replicated = sharding_a.replicate()

hparams = {
    "number_of_sensors": NUMBER_OF_SENSORS,
    "width": 100,
    "depth": 7,
    "learning_rate": 1e-3,
    "interact_size": 15,
    "λ_learning_rate": 1e-3,
    "λ_mask": "soft_relu",
    "λ_shape": (N+1, M+1),
}

hparams = modified_deeponet.Hparams(**hparams)

model = ModifiedDeepONet(hparams)
model = eqx.filter_shard(model, replicated)


train_loader = jdl.DataLoader(jdl.ArrayDataset(a_train_n, u_train_n), batch_size=16, shuffle=True, backend='jax')
val_loader = jdl.DataLoader(jdl.ArrayDataset(a_val_n, u_val_n), batch_size=16, shuffle=True, backend='jax')

θ_learning_rate = 1e-3
λ_learning_rate = 1e-2

θ_optimizer = optax.adam(θ_learning_rate)
λ_optimizer = optax.chain(optax.adam(λ_learning_rate), optax.scale(-1.))
opt = optax.multi_transform({'θ': θ_optimizer, 'λ': λ_optimizer}, param_labels=param_labels)
opt_state = opt.init(eqx.filter([model], eqx.is_array))

trainer = Trainer(model, opt, opt_state, train_loader, val_loader, hparams = hparams, animate=True, max_epochs=30, save_path=PATH, sharding=sharding)
trainer(3000)