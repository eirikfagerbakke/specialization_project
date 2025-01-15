import jax
import equinox as eqx
from jax import random, grad, value_and_grad, vmap, jacfwd
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type
from ..networks._abstract_operator_net import AbstractOperatorNet, AbstractHparams
from ..networks.self_adaptive import SelfAdaptive
from ..networks.energynet import EnergyNet, EnergyNetHparams
import sys
sys.path.append("..")
from utils.trainer import Trainer
from scipy.integrate import cumulative_simpson

from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt

@dataclass(frozen=True, kw_only=True)
class Hparams:
    energy_net: EnergyNetHparams
    operator_net: AbstractHparams

class HamiltonianNeuralOperatorFNO2d(eqx.Module):
    """A Hamiltonian PDE can be written on the form u_t = 𝒢 δℋ/δu.
    This network predicts the function 𝒢 δℋ/δu (x,t), which should be equal to u_t (x,t). 
    The final prediction can then be achieved by integrating the output of the network.
    
    This gives us a hard constraint, as the final prediction is computed via the Hamiltonian structure.
    """
    F : EnergyNet
    u : AbstractOperatorNet
    
    def __init__(self, energy_net : eqx.Module, operator_net: AbstractOperatorNet):
        self.F = energy_net
        self.u = operator_net

    def __call__(self, a, x, t):
        u = lambda x : self.u(a, x, t)    
        u_x = lambda x : self.u.Dx(a, x, t)
        
        # as F takes scalar input, we have to vmap over both inputs
        #dFdu = lambda x : (vmap(grad(self.F, 0))(u(x).ravel(), u_x(x).ravel())).reshape((len(t), len(x))) # ∂F/∂u(x), function
        #dFdu_x = lambda x : (vmap(grad(self.F, 1))(u(x).ravel(), u_x(x).ravel())).reshape((len(t), len(x))) # ∂F/∂u_x(x), function
        dFdu = lambda x : grad(lambda u, u_x : jnp.sum(vmap(self.F)(u, u_x)), 0)(u(x).ravel(), u_x(x).ravel()).reshape((len(t), len(x))) # ∂F/∂u(x), function
        dFdu_x = lambda x : grad(lambda u, u_x : jnp.sum(vmap(self.F)(u, u_x)), 1)(u(x).ravel(), u_x(x).ravel()).reshape((len(t), len(x))) # ∂F/∂u_x(x), function
        
        # to take gradients, we need to compute jacobian as they are vector-valued
        δℋ = lambda x : dFdu(x) - jacfwd(lambda x : jnp.sum(dFdu_x(x), axis=1))(x) # δℋ = ∂F/∂u - (∂F/∂u_x)_x, function
        𝒢δℋ = -jacfwd(lambda x : jnp.sum(δℋ(x), axis=1))(x) # 𝒢 δℋ/δu , value
        
        return 𝒢δℋ
    
    def hamiltonian(self, a, x, t):
        u = self.u(a, x, t)    
        u_x = self.u.Dx(a, x, t)
        # integrates F over x 
        F = vmap(self.F)(u.ravel(), u_x.ravel()).reshape((len(t), len(x)))
        return jnp.trapezoid(F, dx = x[1] - x[0], axis=1)

    def predict_whole_grid(self, a, x, t):
        """Since the model predicts 𝒢δℋ (=u_t),
        we have to integrate the prediction to get u.
        Does so using cumulative_simpson from scipy.integrate.

        Args:
            a (Mp1,): initial condition
            x (Mp1,): spatial grid
            t (Np1,): temporal grid

        Returns:
            u (Np1, Mp1): prediction for the given grid
        """
        # predict 𝒢δℋ (=u_t) over the whole grid (vmap over temporal and spatial dimensions)
        u_t = self(a, x, t)
        return a + cumulative_simpson(u_t, t, axis=0, initial=0)
    
    def get_self_adaptive(self):
        """Retrieves the self-adaptive weights instance, stored in u."""
        return self.u.self_adaptive
    
    def self_adaptive(self, t_idx):
        """Retrieves the self-adaptive weights, with mask applied."""
        return self.u.self_adaptive(t_idx)
    
    @property
    def is_self_adaptive(self):
        return self.u.is_self_adaptive

def compute_loss_FNO2d(model, a, u, key, num_query_points=100):
    """Computes the loss of the model.
    Returns the MSE loss, averaged over the batch. The loss is computed by randomly selecting query points from the input data and evaluating the model at those points.

    Args:
        model (eqx.Module): The model to evaluate.
        a (batch, number_of_sensors): The input data.
        u (batch, num_query_points): The ground truth data at the query points.

    Returns:
        loss (scalar): The loss of the model for the given batch.
    """
    batch_size, Np1, Mp1 = u.shape
    skips = 6
    a = a[:,::skips]
    x = Trainer.x[::skips]
    t = Trainer.t[::skips]
    
    # vmap over a_batch
    𝒢δℋ = vmap(model, (0, None, None))(a, x, t)
    u_t = vmap(lambda a : jnp.diagonal(jacfwd(model.u, 2)(a, x, t)))(a)

    if model.is_self_adaptive:
        # retrieve the self-adaptive weights (with applied masking function)
        λ = model.self_adaptive(jnp.arange(0, Np1, skips)[:,None])
        loss = jnp.mean(λ * jnp.square(u_t-𝒢δℋ))
    else:
        loss = jnp.mean(jnp.square(u_t-𝒢δℋ))

    return loss