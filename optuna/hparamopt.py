import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax_dataloader as jdl
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import networks
from utils import *
import math

import argparse

parser = argparse.ArgumentParser(description='Hyperparameter optimization with Optuna')
parser.add_argument('--problem', type=str, default='advection', choices=['advection', 'kdv'], help='Problem type')
parser.add_argument('--network', type=str, default='modified_deeponet', choices=['deeponet', 'deeponet_mse_loss', 'deeponet_no_he','modified_deeponet', 'fno_timestepping' ,'fno1d', 'fno2d'], help='Network type')
parser.add_argument('--running_on', type=str, default='local', choices=['local', 'idun'], help='Running environment')
parser.add_argument('--num_trials', type=int, default=200, help='Number of trials')
parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('--track_progress', default = False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

problem = args.problem
network = args.network
running_on = args.running_on
num_trials = args.num_trials
num_epochs = args.num_epochs
track_progress = args.track_progress

if running_on == "local":
    data_path = "C:/Users/eirik/OneDrive - NTNU/5. klasse/prosjektoppgave/eirik_prosjektoppgave/data/"
    optuna_path = "sqlite:///optuna/optuna.db"
    hparams_path = "C:/Users/eirik/OneDrive - NTNU/5. klasse/prosjektoppgave/eirik_prosjektoppgave/hyperparameters/"
elif running_on == "idun":
    data_path = "/cluster/work/eirikaf/data/"
    optuna_path = "sqlite:///phlearn-summer24/eirik_prosjektoppgave/optuna/optuna.db"
    hparams_path = "/cluster/home/eirikaf/phlearn-summer24/eirik_prosjektoppgave/hyperparameters/"
else:
    raise ValueError("Invalid running_on")

#LOAD DATA
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

# LOAD HYPERPARAMETERS (the ones that are not being optimized)
with open(hparams_path + f"{network}_{problem}.json", "rb") as f:
    hparams_dict = json.load(f)

# IMPORT WANTED NETWORK ARCHITECTURE
if network == "deeponet":
    from networks.deeponet import HparamTuning, compute_loss
elif network == "modified_deeponet":
    from networks.modified_deeponet import HparamTuning, compute_loss
elif network == "fno_timestepping":
    from networks.fno_timestepping import HparamTuning, compute_loss
elif network == "fno1d":
    from networks.fno1d import HparamTuning, compute_loss
elif network == "fno2d":
    from networks.fno2d import HparamTuning, compute_loss
else:
    raise ValueError("Invalid network")

Trainer.compute_loss = staticmethod(compute_loss)

study = optuna.create_study(
    study_name=network + "_" + problem + "_SA",
    storage=optuna_path,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50),
    sampler = optuna.samplers.TPESampler(),
    load_if_exists=True,
)

"""
def keep_this_trial(trial):
    # Keep trials that are COMPLETE, "self_adaptive" is False, and value is finite
    return (
        trial.state == optuna.trial.TrialState.COMPLETE
        and trial.params.get("self_adaptive", None) is False
        and trial.value is not None
        and math.isfinite(trial.value)
    )

old_trials = old_study.get_trials(deepcopy=False)
correct_trials = [optuna.trial.create_trial(
        params=t.params,
        distributions=t.distributions,
        value=t.value,
    ) for t in old_trials if keep_this_trial(t)]
study = optuna.create_study(study_name=network + "_" + problem,
    storage=optuna_path[:-3]  + "_new" + ".db",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50),
    sampler = optuna.samplers.TPESampler(),
    load_if_exists=True,
)
study.add_trials(correct_trials)
"""

study.optimize(HparamTuning(train_loader, 
                            val_loader,
                            z_score_data,
                            hparams = hparams_dict,
                            x = x_train_s,
                            t = t_train_s,
                            sharding_a = sharding_a,
                            sharding_u = sharding_u,
                            replicated = replicated,
                            max_epochs = num_epochs,
                            track_progress = track_progress), 
                n_trials=num_trials,
                show_progress_bar=track_progress,)