import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import jax
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp
from jax import vmap
import jax_dataloader as jdl
import networks
from utils import *

import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as jshard
import argparse
from optax.contrib import reduce_on_plateau

parser = argparse.ArgumentParser(description='Hyperparameter optimization with Optuna')
parser.add_argument('--problem', type=str, default='advection', choices=['advection', 'kdv'], help='Problem type')
parser.add_argument('--network', type=str, default='modified_deeponet', choices=['deeponet', 'modified_deeponet', 'fno1d', 'fno2d', "fno_timestepping"], help='Network type')
parser.add_argument('--running_on', type=str, default='local', choices=['local', 'idun'], help='Running environment')
parser.add_argument('--num_epochs', type=int, default=2000, help='Number of epochs')
parser.add_argument('--save', default = True, action=argparse.BooleanOptionalAction)
parser.add_argument('--track_progress', default = False, action=argparse.BooleanOptionalAction)
parser.add_argument('--load_trainer', default = False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

problem = args.problem
network = args.network
running_on = args.running_on
num_epochs = args.num_epochs
save = args.save
track_progress = args.track_progress
load_trainer = args.load_trainer

print("Running with the following settings:")
print(f"Problem: {problem}")
print(f"Network: {network}")
print(f"Running on: {running_on}")
print(f"Number of epochs: {num_epochs}")
print(f"Save: {save}")
print(f"Track progress: {track_progress}")
print(f"Load trainer: {load_trainer}")

if running_on == "local":
    data_path = "C:/Users/eirik/OneDrive - NTNU/5. klasse/prosjektoppgave/eirik_prosjektoppgave/data/"
    hparams_path = "C:/Users/eirik/OneDrive - NTNU/5. klasse/prosjektoppgave/eirik_prosjektoppgave/hyperparameters/"
    checkpoint_path = "C:/Users/eirik/orbax/checkpoints/" if save else None
elif running_on == "idun":
    data_path = "/cluster/work/eirikaf/data/"
    hparams_path = "/cluster/home/eirikaf/phlearn-summer24/eirik_prosjektoppgave/hyperparameters/"
    checkpoint_path = "/cluster/work/eirikaf/checkpoints/" if save else None
else:
    raise ValueError("Invalid running_on")


scaled_data = jnp.load(data_path + problem + "_z_score.npz")
a_train_s = jnp.array(scaled_data["a_train_s"])
u_train_s = jnp.array(scaled_data["u_train_s"])
a_val_s = jnp.array(scaled_data["a_val_s"])
u_val_s = jnp.array(scaled_data["u_val_s"])

x_train_s = jnp.array(scaled_data["x_train_s"])
t_train_s = jnp.array(scaled_data["t_train_s"])

z_score_data = {
    "u_std" : scaled_data["u_std"].item(),
    "u_mean" : scaled_data["u_mean"].item(),
    "x_std" : scaled_data["x_std"].item(),
    "x_mean" : scaled_data["x_mean"].item(),
    "t_std" : scaled_data["t_std"].item(),
    "t_mean" : scaled_data["t_mean"].item()
}

# DATALOADERS
train_loader = jdl.DataLoader(jdl.ArrayDataset(a_train_s, u_train_s, asnumpy = False), batch_size=16, shuffle=True, backend='jax', drop_last=True)
val_loader = jdl.DataLoader(jdl.ArrayDataset(a_val_s, u_val_s, asnumpy = False), batch_size=16, shuffle=True, backend='jax', drop_last=True)

# AUTOPARALLELISM
sharding_a, sharding_u, replicated = create_device_mesh()

# IMPORT WANTED NETWORK ARCHITECTURE
if network == "deeponet":
    from networks.deeponet import *
    from networks import DeepONet as Network
elif network == "modified_deeponet":
    from networks.modified_deeponet import *
    from networks import ModifiedDeepONet as Network
elif network == "fno1d":
    from networks import FNO1d as Network
    from networks.fno1d import *
elif network == "fno2d":
    from networks import FNO2d as Network
    from networks.fno2d import *
elif network == "fno_timestepping":
    from networks import FNOTimeStepping as Network
    from networks.fno_timestepping import *
else:
    raise ValueError("Invalid network")

Trainer.compute_loss = staticmethod(compute_loss)

# IMPORT HYPERPARAMETERS
with open(hparams_path + f"{network}_{problem}.json", "rb") as f:
    hparams_dict = json.load(f) | z_score_data
    hparams = Hparams(**hparams_dict)

# OPTIMIZERS
PATIENCE = 5 # Number of epochs with no improvement after which learning rate will be reduced
COOLDOWN = 0 # Number of epochs to wait before resuming normal operation after the learning rate reduction
FACTOR = 0.5  # Factor by which to reduce the learning rate:
RTOL = 1e-4  # Relative tolerance for measuring the new optimum:
ACCUMULATION_SIZE = 200 # Number of iterations to accumulate an average value:

if network in ["fno1d", "fno2d", "fno_timestepping"]:
    θ_optimizer = optax.chain(
        conjugate_grads_transform(),
        optax.adamw(hparams.learning_rate),
        reduce_on_plateau(
            patience=PATIENCE,
            cooldown=COOLDOWN,
            factor=FACTOR,
            rtol=RTOL,
            accumulation_size=ACCUMULATION_SIZE,
        ),
    )
else:
    θ_optimizer = optax.chain(
        optax.adamw(hparams.learning_rate),
        reduce_on_plateau(
            patience=PATIENCE,
            cooldown=COOLDOWN,
            factor=FACTOR,
            rtol=RTOL,
            accumulation_size=ACCUMULATION_SIZE,
        ),
    )

if hparams.λ_learning_rate: # Self-adaptive weights are enabled 
    λ_optimizer = optax.chain(optax.adam(hparams.λ_learning_rate), optax.scale(-1.))
    opt = optax.multi_transform({'θ': θ_optimizer, 'λ': λ_optimizer}, param_labels=param_labels)
else:
    opt = θ_optimizer

# INITIALIZE MODEL AND OPT STATE
model = Network(hparams)
if replicated:
    model = eqx.filter_shard(model, replicated)
    
if load_trainer:
    trainer = Trainer.from_checkpoint(checkpoint_path+f"{network}_{problem}", 
                                            Network,
                                            opt = opt, 
                                            Hparams=Hparams,
                                            train_loader=train_loader, 
                                            val_loader=val_loader,
                                            x = x_train_s,
                                            t = t_train_s, 
                                            save_path = checkpoint_path, 
                                            save_path_prefix = network + "_" + problem + "_",
                                            sharding_a = sharding_a,
                                            sharding_u = sharding_u,
                                            replicated = replicated,)
else:
    opt_state = opt.init(eqx.filter([model], eqx.is_array))

    # INITIALIZE TRAINER
    trainer = Trainer(model, 
                    opt, 
                    opt_state, 
                    train_loader, 
                    val_loader,
                    x = x_train_s,
                    t = t_train_s,
                    hparams = hparams, 
                    save_path = checkpoint_path, 
                    save_path_prefix = network + "_" + problem + "_",
                    sharding_a = sharding_a,
                    sharding_u = sharding_u,
                    replicated = replicated,)

# TRAIN
trainer(num_epochs, track_progress = track_progress)