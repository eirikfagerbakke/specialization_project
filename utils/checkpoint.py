from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .trainer import Trainer

import orbax.checkpoint as ocp
from dataclasses import dataclass, asdict
from typing import Type, Optional
import equinox as eqx
from jaxtyping import PyTree
import json
import os
import jax_dataloader as jdl
from etils import epath
from datetime import datetime
import optax

class EquinoxCheckpointHandler(ocp.CheckpointHandler):
    """Checkpoint handler for Equinox models, which utilizes the built-in tree serialisation of Equinox.
    Copied from https://github.com/google/orbax/issues/741.
    """
    def save(
        self,
        directory: epath.Path,
        args: "EquinoxStateSave",
    ):
        full_path = directory / "model.eqx"
        eqx.tree_serialise_leaves(str(full_path), args.item, is_leaf=eqx.is_array)

    def restore(
        self,
        directory: epath.Path,
        args: "EquinoxStateRestore",
    ) -> eqx.Module:
        full_path = directory / "model.eqx"
        loaded = eqx.tree_deserialise_leaves(str(full_path), args.item, is_leaf=eqx.is_array)
        return loaded


@ocp.args.register_with_handler(EquinoxCheckpointHandler, for_save=True)
@dataclass
class EquinoxStateSave(ocp.args.CheckpointArgs):
    item: eqx.Module


@ocp.args.register_with_handler(EquinoxCheckpointHandler, for_restore=True)
@dataclass
class EquinoxStateRestore(ocp.args.CheckpointArgs):
    item: eqx.Module

def save_checkpoint(trainer : Trainer, epoch_idx : int, mngr : ocp.CheckpointManager, val_loss : float):
    """Uses orbax to checkpoint the trainer, optimizer state and training history.

    Args:
        trainer (Trainer)
        epoch_idx (int)
        save_path (str)
    """
    
    # Use orbax to checkpoint the rest of the training state
    mngr.save(
        epoch_idx,
        metrics={"current_val_loss" : val_loss},
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(eqx.filter(trainer.model, eqx.is_array)),
            opt_state=ocp.args.StandardSave(trainer.opt_state),
            training_info=ocp.args.StandardSave({
                "train_loss_history": trainer.train_loss_history,
                "val_loss_history": trainer.val_loss_history,
                "train_loss_history_batch": trainer.train_loss_history_batch,
                "val_loss_history_batch": trainer.val_loss_history_batch,
                "λ_history": trainer.λ_history,
                "epochs_trained": trainer.epochs_trained,
                "time_trained": trainer.time_trained/trainer.epochs_trained,
                "time_val": trainer.time_val/trainer.epochs_trained*trainer.val_every_n_epoch,
            }),
        )
    )