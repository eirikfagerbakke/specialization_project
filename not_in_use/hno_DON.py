import jax
#jax.config.update("jax_enable_x64", True)
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

class HNO_DON(eqx.Module):
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
        """The input and output of the HNO is the same as the input of the OperatorNet.
        In all cases, a is an array of point-evaluations of the function a.
        """

        u = lambda x : self.u(a, x, t) # u(x), function in x only (for convenience)
        u_x = grad(u) # u_x(x), function
        
        dFdu = lambda x : grad(self.F, 0)(u(x), u_x(x)) # ∂F/∂u(x), function
        dFdu_x = lambda x : grad(self.F, 1)(u(x), u_x(x)) # ∂F/∂u_x(x), function
        δℋ = lambda x : dFdu(x) - grad(dFdu_x)(x) # δℋ/δu(x), function
        
        𝒢δℋ = -grad(δℋ)(x) # 𝒢 δℋ/δu , value
        return 𝒢δℋ
    
    def predict_whole_grid(self, a, x, t):
        """Since the model predicts 𝒢δℋ (=u_t),
        we have to integrate the prediction to get u.
        Does so using an ODE solver (Tsit5() in Diffrax)

        Args:
            a (Mp1,): initial condition
            x (Mp1,): spatial grid
            t (Np1,): temporal grid

        Returns:
            u (Np1, Mp1): prediction for the given grid
        """
        def u_t(t, y, x):
            # vmap over the spatial dimension 
            # (model takes scalar values. t is scalar, but x is an array)
            return vmap(self, (None, 0, None))(a[::4], x, t)
        
        term = ODETerm(u_t)
        solver = Tsit5()
        saveat = SaveAt(ts=t)
        solution = diffeqsolve(term, 
                               solver, 
                               t0=t[0],
                               t1=t[-1], 
                               dt0=t[1]-t[0], 
                               y0=a, 
                               args=x, 
                               saveat=saveat)
        return solution.ys

    def predict_whole_grid2(self, a, x, t):
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
        u_t = vmap(vmap(self, (None, 0, None)), (None, None, 0))(a, x, t)
        return a + cumulative_simpson(u_t, t, axis=0, initial=0)
    
    def get_self_adaptive(self):
        """Retrieves the self-adaptive weights instance, stored in u."""
        return self.u.self_adaptive
    
    def self_adaptive(self, t_idx = None):
        """Retrieves the self-adaptive weights, with mask applied."""
        return self.u.self_adaptive(t_idx)
    
    @property
    def is_self_adaptive(self):
        return self.u.is_self_adaptive
        
    
def compute_loss_DON(model, a, u, key, num_query_points=100):
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
    
            
    # Select random query indices
    t_key, x_key = random.split(key, 2)
    t_idx = random.randint(t_key, (batch_size, num_query_points), 0, Np1)
    x_idx = random.randint(x_key, (batch_size, num_query_points), 0, Mp1)

    # Select the ground truth data at the query points
    # Has shape (batch_size, num_query_points)
    
    # For each input function, compute the prediction of the model at the query points. (inner vmap)
    # Do this for each sample in the batch. (outer vmap)
    # Has shape (batch_size, num_query_points)   
    𝒢δℋ = vmap(vmap(model, (None, 0, 0)))(a, Trainer.x[x_idx], Trainer.t[t_idx])
    u_t = vmap(vmap(grad(model.u, 2), (None, 0, 0)))(a, Trainer.x[x_idx], Trainer.t[t_idx])
    
    #mse loss
    u_norms = jnp.linalg.norm(u_t, 2, 1)
    if model.is_self_adaptive:
        λ = model.self_adaptive(t_idx) 
        diff_norms = jnp.sqrt(jnp.sum(λ * jnp.square(u_t - 𝒢δℋ), axis=1))
    else:
        diff_norms = jnp.linalg.norm((u_t - 𝒢δℋ), 2, 1)
        
    loss = jnp.mean(diff_norms/u_norms)
    return loss

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
    # vmap over a_batch
    𝒢δℋ = vmap(model, (0, None, None))(a, Trainer.x[::4], Trainer.t[::4])
    u_t = vmap(grad(model.u, 2), (0, None, None))(a, Trainer.x[::4], Trainer.t[::4])

    if model.self_adaptive:
        # retrieve the self-adaptive weights (with applied masking function)
        λ = model.self_adaptive(jnp.arange(0, Np1, 4)[:,None])
        loss = jnp.mean(λ * jnp.square(u_t - 𝒢δℋ))
    else:
        loss = jnp.mean(jnp.square(u_t - 𝒢δℋ))

    return loss