import optuna
import jax
#jax.config.update("jax_enable_x64", True)
from jax import random, vmap
import jax.numpy as jnp
import json
import numpy as np
import equinox as eqx
from ._progress_bar import ProgressBar, CustomProgress
from .checkpoint import *

import orbax.checkpoint as ocp
from dataclasses import asdict
from datetime import datetime
from etils import epath
from typing import Optional, Type
from jaxtyping import PyTree
import optax
import jax_dataloader as jdl
from time import perf_counter

import sys 
sys.path.append("..")
from networks.self_adaptive import get_self_adaptive, normalize_self_adaptive

class Trainer:
    def __init__(self, 
                 model : eqx.Module = None, 
                 opt : optax.GradientTransformation = None, 
                 opt_state : PyTree = None, 
                 train_loader : jdl.DataLoader = None, 
                 val_loader : jdl.DataLoader = None, 
                 save_path : Optional[str] = None,
                 **kwargs):
        
        # Initialize the model, optimizer, and data loaders
        self.model = model
        Trainer.opt = opt
        self.opt_state = opt_state
        self.train_loader = train_loader
        self.val_loader = val_loader
        if self.train_loader:
            self.train_batches = len(train_loader)
        if self.val_loader:
            self.val_batches = len(val_loader)
        self.hparams = kwargs.get("hparams", {})
        
        self.max_epochs = kwargs.get("max_epochs", 300)
        self.val_every_n_epoch  = kwargs.get("val_every_n_epoch ", 5)
        
        # Early stopping parameters
        self.stopped_at_epoch = self.max_epochs
        self.early_stopping = kwargs.get("early_stopping", False)
        self.min_epochs = kwargs.get("min_epochs", 50)
        self.early_stopping_patience = kwargs.get("early_stopping_patience", 25)
        self.early_stopping_counter = self.early_stopping_patience
        self.best_val_loss = jnp.inf
        
        # Auto-parallelism
        Trainer.sharding_a = kwargs.get("sharding_a")
        Trainer.sharding_u = kwargs.get("sharding_u")
        Trainer.replicated = kwargs.get("replicated")
        if Trainer.replicated:
            Trainer.multi_device = True
        else:
            Trainer.multi_device = False
            
        # Grid
        Trainer.x = kwargs.get("x")
        Trainer.t = kwargs.get("t")
        
        # Used for optuna (hyperparameter optimization)
        self.trial = kwargs.get("trial")
        
        # Lists for storing the loss history
        self.epochs_trained = 0
            
        # Initialize progress bar
        self.track_progress=kwargs.get("track_progress", True)
        
        # Check if model is self-adaptive

        self.λ_history = jnp.array([0.])
        self.save_λ_every_n_epoch = kwargs.get("save_λ_every_n_epoch", 5)
        if self.model.is_self_adaptive:
            self.save_λ_every_n_epoch = kwargs.get("save_λ_every_n_epoch", 5)
            all_λ_with_mask = jnp.array([x.all_with_mask() for x in get_self_adaptive(self.model)]) # list of self-adaptive weights
            self.λ_history = jnp.empty((self.max_epochs//self.save_λ_every_n_epoch, *all_λ_with_mask.shape))
            self.λ_history = self.λ_history.at[0].set(all_λ_with_mask)
        
        # Save intermediate models
        self.save_path = save_path
        if self.save_path:
            save_path_prefix = kwargs.get("save_path_prefix", "")
            # Create a folder for saving the model, and save the hyperparameters
            self.save_path = epath.Path(self.save_path)
            folder_name = save_path_prefix + datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_path = self.save_path / folder_name
            self.save_path.mkdir()
            
            with open(self.save_path / 'hparams.json', 'w') as f:
                # check if hparams is a dataclass or dict
                if isinstance(self.hparams, dict):
                    json.dump(self.hparams, f, indent=4)
                else:
                    json.dump(asdict(self.hparams), f, indent=4)
                    
    
    @eqx.filter_jit(donate="all")
    @staticmethod
    def make_step(model, opt_state, a, u, key):
        """Advances the model one step. 
        Args:
            model: the model to update
            opt_state: the optimizer state
            inputs: inputs to the model
            ground_truth: the ground truth
            key: key for genenerating random numbers 
        """
        if Trainer.multi_device:
            model, opt_state = eqx.filter_shard((model, opt_state), Trainer.replicated)
            a, u = eqx.filter_shard((a,u), (Trainer.sharding_a, Trainer.sharding_u))
        
        loss, grads = eqx.filter_value_and_grad(Trainer.compute_loss)(model, a, u, key)
        updates, opt_state = Trainer.opt.update([grads], opt_state, value=loss, params=eqx.filter([model], eqx.is_array))
        model = eqx.apply_updates(model, updates[0])
        
        if model.is_self_adaptive:
            model = eqx.tree_at(get_self_adaptive, 
                                model, 
                                replace_fn = normalize_self_adaptive)
            
        if Trainer.multi_device:
            model, opt_state = eqx.filter_shard((model, opt_state), Trainer.replicated)
        
        return model, opt_state, loss
    
    @staticmethod
    def compute_loss(model, a, u, key):
        """Computes the loss for one batch.
        Args:
            model: the model to update
            opt_state: the optimizer state
            opt: the optimizer to use
            inputs: inputs to the model
            ground_truth: the ground truth 
            key: key for genenerating random numbers
        
        Needs to be overwritten.
        """
        raise NotImplementedError
    
    @eqx.filter_jit(donate="all-except-first")
    @staticmethod
    def evaluate(model, a, u, key):
        """Evaluates the model on the validation set.
        Same loss function across all methods (on the whole grid).
        
        Args:
            model: the model to update
            inputs: input function to the model
            ground_truth: the ground truth
            key: key for genenerating random numbers
        """
        if Trainer.multi_device:
            model = eqx.filter_shard(model, Trainer.replicated)
            a, u = eqx.filter_shard((a,u), (Trainer.sharding_a, Trainer.sharding_u))
            
        model = eqx.nn.inference_mode(model)
            
        u_pred = model.predict_whole_grid_batch(a, Trainer.x, Trainer.t)
        
        #compute the loss
        batch_size = len(u) 
        u_norms = jnp.linalg.norm(u.reshape(batch_size,-1), 2, 1)
        diff_norms = jnp.linalg.norm((u - u_pred).reshape(batch_size,-1), 2, 1)
            
        loss = jnp.mean(diff_norms/u_norms)
        a_pred = u_pred[:,0]
        ic_loss = jnp.mean(jnp.linalg.norm(a - a_pred, 2, 1)/jnp.linalg.norm(a, 2, 1))
        return loss + ic_loss*0.5
    
    def _init_loss_history(self, max_epochs):
        """Initializes placeholders for the loss history arrays."""
        train_loss_history = np.zeros(max_epochs) # training loss history per epoch
        val_loss_history = np.zeros(max_epochs//self.val_every_n_epoch) # validation loss history per epoch
        train_loss_history_batch = np.zeros(self.train_batches*max_epochs)  # training loss history per batch
        val_loss_history_batch = np.zeros(self.val_batches*max_epochs//self.val_every_n_epoch)  # validation loss history per batch
        
        if self.epochs_trained:
            # Model has already been trained, need to append to the existing loss history
            self.train_loss_history = np.concatenate((self.train_loss_history, train_loss_history))
            self.val_loss_history = np.concatenate((self.val_loss_history, val_loss_history))
            self.train_loss_history_batch = np.concatenate((self.train_loss_history_batch, train_loss_history_batch))
            self.val_loss_history_batch = np.concatenate((self.val_loss_history_batch, val_loss_history_batch))
        else:
            # Initialize the loss history
            self.train_loss_history = train_loss_history
            self.val_loss_history = val_loss_history
            self.train_loss_history_batch = train_loss_history_batch
            self.val_loss_history_batch = val_loss_history_batch
        
    def __call__(self, max_epochs : Optional[int] = None, key = random.key(0), track_progress = None):
        """Trains the model.

        Args:
            max_epochs (Optional[int]): Override the number of epochs to train.
            key: A JAX PRNG key for random number generation, which may be used in the loss function.
        """
        assert self.model is not None, "Model is not initialized."
        assert Trainer.opt is not None, "Optimizer is not initialized."
        assert self.opt_state is not None, "Optimizer state is not initialized."
        assert self.train_loader is not None, "Training data loader is not initialized."
        assert self.val_loader is not None, "Validation data loader is not initialized."
        
        if max_epochs: self.max_epochs = max_epochs
        
        if self.save_path:
            checkpoint_options = ocp.CheckpointManagerOptions(save_interval_steps=5, 
                                                              max_to_keep=5,
                                                              best_fn=lambda x: x["current_val_loss"],
                                                              best_mode='min')
            
            checkpoint_manager = ocp.CheckpointManager(self.save_path, options=checkpoint_options)
            
        self._init_loss_history(self.max_epochs)
        
        pbar = None
        if track_progress or (track_progress is None and self.track_progress):
            pbar = CustomProgress(10)
            pbar.start()
            
            epochs_id = pbar.add_task("[bold green]Epochs...", total=self.max_epochs)
            pbar.train_id = pbar.add_task("[cyan]Train epoch...", total=self.train_batches)
            pbar.val_id = pbar.add_task("[cyan]Validation epoch...", total=self.val_batches)
        val_loss = 1.
        for epoch_idx in range(self.epochs_trained, self.epochs_trained+self.max_epochs):
            key, train_key, val_key = random.split(key, 3)
            
            train_loss = self._train_epoch(epoch_idx, train_key, pbar)
            
            if (epoch_idx+1) % self.val_every_n_epoch == 0:
                val_loss = self._val_epoch(epoch_idx, val_key, pbar)
            
                if self.trial:
                    self.trial.report(val_loss, step=epoch_idx)
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                if self.early_stopping:
                    if self.min_epochs <= epoch_idx and self._should_stop_early(val_loss):
                        self.stopped_at_epoch = epoch_idx
                        if self.model.is_self_adaptive:
                            self.λ_history = self.λ_history[:epoch_idx//self.save_λ_every_n_epoch]
                        break
                    
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                
                
            # save λ every n epochs
            if self.model.is_self_adaptive and epoch_idx % self.save_λ_every_n_epoch == 0:                    
                all_λ_with_mask = jnp.array([x.all_with_mask() for x in get_self_adaptive(self.model)]) # list of self-adaptive weights
                self.λ_history = self.λ_history.at[epoch_idx//self.save_λ_every_n_epoch].set(all_λ_with_mask)
                
            # save intermediate models in case of crash   
            self.epochs_trained += 1
            if self.save_path:
                save_checkpoint(self, self.epochs_trained, checkpoint_manager, val_loss)
                
            if track_progress:
                pbar.advance(epochs_id)
                pbar.reset(pbar.train_id)
                pbar.reset(pbar.val_id)
                pbar.update_table((f"{self.epochs_trained}", f"{train_loss}", f"{val_loss}"))
    
        if track_progress: pbar.stop()
        if self.save_path:
            checkpoint_manager.wait_until_finished()
                    
    def _train_epoch(self, epoch_idx, key, pbar):        
        total_train_loss = 0.
        model, opt_state = self.model, self.opt_state
        keys = random.split(key, self.train_batches)
        for i, (batch_a, batch_u) in enumerate(self.train_loader):
            if pbar: pbar.advance(pbar.train_id) 

            batch_a, batch_u = eqx.filter_shard((batch_a, batch_u), (Trainer.sharding_a, Trainer.sharding_u))
            model, opt_state, train_loss_batch = Trainer.make_step(model, opt_state, batch_a, batch_u, keys[i])
            
            train_loss_batch_item = train_loss_batch.item()
            self.train_loss_history_batch[epoch_idx*self.train_batches + i] = train_loss_batch_item
            total_train_loss += train_loss_batch_item
    
        train_loss_epoch = total_train_loss / self.train_batches
        self.train_loss_history[epoch_idx] = train_loss_epoch
        
        self.model, self.opt_state = model, opt_state
        return train_loss_epoch
    
    def _val_epoch(self, epoch_idx, key, pbar):
        epoch_idx = epoch_idx // self.val_every_n_epoch
        
        total_val_loss = 0.
        
        keys = random.split(key, self.val_batches)
        for i, (batch_inputs, batch_ground_truth) in enumerate(self.val_loader): 
            if pbar: pbar.advance(pbar.val_id) 
                        
            val_loss_batch = Trainer.evaluate(self.model, batch_inputs, batch_ground_truth, keys[i])
            
            val_loss_batch_item = val_loss_batch.item()
            self.val_loss_history_batch[epoch_idx*self.val_batches + i] = val_loss_batch_item
            total_val_loss += val_loss_batch_item
        
        val_loss_epoch = total_val_loss / self.val_batches
        self.val_loss_history[epoch_idx] = val_loss_epoch
        
        return val_loss_epoch
    
    def _should_stop_early(self, val_loss):
        if val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = self.early_stopping_patience  # reset patience
            return False
        else:
            self.early_stopping_counter -= 1
            return self.early_stopping_counter == 0
    
    @classmethod
    def from_checkpoint(cls, path : str,
                    Network : Type[eqx.Module], 
                    opt : optax.GradientTransformation = None,
                    epoch_idx: Optional[int] = None,
                    **kwargs):
        """Loads a trainer from a checkpoint, using orbax.

        Args:
            Network (Type[eqx.Module])
            opt (optax.GradientTransformation)
            abstract_opt_state (PyTree)
            train_loader (jdl.DataLoader)
            val_loader (jdl.DataLoader)
            path (str)
            epoch_idx (Optional[int], optional): Which epoch to load. Defaults to the last saved step.
            kwargs: Additional arguments to pass to the Trainer.

        Returns:
            Trainer: the loaded trainer.
        """
        return _from_checkpoint(path, Network, opt, epoch_idx, **kwargs)

def _from_checkpoint(path : str,
                    Network : Type[eqx.Module], 
                    opt : optax.GradientTransformation = None,
                    epoch_idx: Optional[int] = None,
                    Hparams : Type[dataclass] = None,
                    **kwargs):
    path = epath.Path(path)

    # First load the hparams
    with open(path / 'hparams.json', "rb") as f:
        hparams = json.load(f)
        hparams = Hparams(**hparams) if Hparams else hparams

    # Initialize the model with the given hparams
    abstract_model = Network(hparams)
    if replicated:=kwargs.get("replicated"):
        abstract_model = eqx.filter_shard(abstract_model, replicated)

    mngr = ocp.CheckpointManager(path)
    epoch_idx = mngr.latest_step() if epoch_idx is None else epoch_idx

    if opt:
        abstract_opt_state = opt.init(eqx.filter([abstract_model], eqx.is_array))
        restored = mngr.restore(
            epoch_idx, 
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(eqx.filter(abstract_model, eqx.is_array)),
                opt_state=ocp.args.StandardRestore(abstract_opt_state),
                training_info=ocp.args.PyTreeRestore(
                    restore_args={
                    'train_loss_history': ocp.RestoreArgs(restore_type=np.ndarray),
                    'val_loss_history': ocp.RestoreArgs(restore_type=np.ndarray),
                    'train_loss_history_batch': ocp.RestoreArgs(restore_type=np.ndarray),
                    'val_loss_history_batch': ocp.RestoreArgs(restore_type=np.ndarray),
                    'λ_history': ocp.RestoreArgs(restore_type=np.ndarray),
                    'epochs_trained': ocp.RestoreArgs(restore_type=int),    
                    }
                ),
            )
        )
        opt_state = restored.opt_state
    else:
        restored = mngr.restore(
            epoch_idx, 
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(eqx.filter(abstract_model, eqx.is_array)),
                training_info=ocp.args.PyTreeRestore(
                    restore_args={
                    'train_loss_history': ocp.RestoreArgs(restore_type=np.ndarray),
                    'val_loss_history': ocp.RestoreArgs(restore_type=np.ndarray),
                    'train_loss_history_batch': ocp.RestoreArgs(restore_type=np.ndarray),
                    'val_loss_history_batch': ocp.RestoreArgs(restore_type=np.ndarray),
                    'λ_history': ocp.RestoreArgs(restore_type=np.ndarray),
                    'epochs_trained': ocp.RestoreArgs(restore_type=int),
                    }
                ),
            )
        )
        opt_state = None

    model = eqx.combine(restored.model, abstract_model)
        
    trainer = Trainer(model, opt, opt_state, hparams = hparams, **kwargs) #save_path = path.parent

    trainer.train_loss_history = np.trim_zeros(restored.training_info["train_loss_history"], 'b')
    trainer.val_loss_history = np.trim_zeros(restored.training_info["val_loss_history"], 'b')
    trainer.train_loss_history_batch = np.trim_zeros(restored.training_info["train_loss_history_batch"], 'b')
    trainer.val_loss_history_batch = np.trim_zeros(restored.training_info["val_loss_history_batch"], 'b')

    trainer.epochs_trained = restored.training_info["epochs_trained"]
    trainer.λ_history = jnp.array(restored.training_info["λ_history"])
    
    return trainer    