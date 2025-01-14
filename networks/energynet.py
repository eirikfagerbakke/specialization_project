import jax
import equinox as eqx
from jax import random, grad, value_and_grad, vmap
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type
from ._abstract_operator_net import AbstractOperatorNet, AbstractHparams
from .self_adaptive import SelfAdaptive, SAHparams
import sys
sys.path.append("..")
from utils.model_utils import init_he, init_F

@dataclass(frozen=True, kw_only=True)
class EnergyNetHparams(AbstractHparams):
    depth: int
    width: int
    energy_penalty: float
    num_query_points: int = 100
    
class EnergyNet(AbstractOperatorNet):
    mlp : eqx.nn.MLP
    energy_penalty : float
    num_query_points : int

    def __init__(self, hparams):
        if isinstance(hparams, dict):
            hparams = EnergyNetHparams(**hparams)
        super().__init__(hparams)
   
        activation = jax.nn.gelu
        key = random.key(hparams.seed)

        self.mlp = init_he(eqx.nn.MLP(
            in_size=2, #takes u and u_x as inputs
            out_size='scalar',
            width_size=hparams.width,
            depth=hparams.depth,
            activation=activation,
            key=key,
        ), key)
        
        self.energy_penalty = hparams.energy_penalty
        self.num_query_points = hparams.num_query_points
        
    def __call__(self, u, u_x):
        return self.mlp(jnp.array([u, u_x]))
    
    def predict_whole_grid(self, u, u_x):
        return vmap(self)(u.ravel(), u_x.ravel()).reshape(u.shape)
    
    def predict_whole_grid_batch(self, u, u_x):
        """u and u_x are 3D arrays with shape (batch_size, t_dim, x_dim)"""
        return vmap(self.predict_whole_grid)(u, u_x)