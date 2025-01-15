import jax
#jax.config.update("jax_enable_x64", True)
import equinox as eqx
from jax import random, grad, vmap, jacfwd, value_and_grad, jacrev
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type
from .energynet import EnergyNet, EnergyNetHparams
from .fno_timestepping import FNOTimeStepping as OperatorNet, compute_loss as compute_operator_loss, Hparams as OperatorHparams
import sys
sys.path.append("..")
from utils.trainer import Trainer

from scipy.integrate import cumulative_simpson
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt

@dataclass(frozen=True, kw_only=True)
class Hparams:
    energy_net: EnergyNetHparams
    operator_net: OperatorHparams

class HNO(eqx.Module):
    """A Hamiltonian PDE can be written on the form u_t = 𝒢 δℋ/δu.
    The operator network predicts the function u(x,t). From this function, we compute
    u_t and 𝒢 δℋ/δu using automatic differentiation.
    These terms should be equal, and can be added as a penalty term to the loss function.
    
    This gives us a network that is "informed" by the Hamiltonian structure of the PDE.
    """
    F : EnergyNet
    u : OperatorNet
    
    is_self_adaptive: bool = False
    
    def __init__(self, hparams = None, 
                 energy_net : eqx.Module = None, 
                 operator_net: OperatorNet = None):
        if hparams is not None:
            energy_net = EnergyNet(hparams["energy_net"])
            operator_net = OperatorNet(hparams["operator_net"])
                
        self.F = energy_net
        self.u = operator_net
        
        self.is_self_adaptive = self.F.is_self_adaptive or self.u.is_self_adaptive

    def __call__(self, a, x, t):
        """The input and output is the same as the input of the OperatorNet.
        Input:
            x: (Nx,) array
            t: (Nt, array)
        Output:
            𝒢δℋ(x,t) (= u_t(x,t)): (Nt, Nx)
        """
        u = self.u.decode_u(self.u(a, x, t)).ravel()
        u_x = self.u.u_x(a,x,t).ravel()
        u_xx = self.u.u_xx(a,x,t).ravel()
        u_xxx = self.u.u_xxx(a,x,t).ravel()
        
        # Notation: write u=y and u_x=z.
        # dF/du is then F_y, dF/du_x is F_z, etc.
        
        F_z = grad(self.F, 1)
        
        F_zz = grad(F_z, 1)
        F_zy = grad(F_z)
        
        F_zy_val, (F_zyy_val, _) = vmap(value_and_grad(F_zy, (0,1)))(u, u_x)
        F_zz_val, (F_zzy_val, F_zzz_val) = vmap(value_and_grad(F_zz, (0,1)))(u, u_x)
        
        F_yy_val = vmap(grad(grad((self.F))))(u, u_x)
        
        F_yx_val = F_yy_val * u_x  + F_zy_val * u_xx
        
        F_zxx = F_zy_val*u_xx +\
                F_zz_val*u_xxx +\
                F_zyy_val*u_x**2 +\
                2*F_zzy_val*u_x*u_xx+\
                F_zzz_val * u_xx**2
                                    
        𝒢δℋ = - F_yx_val + F_zxx
        return 𝒢δℋ.reshape((len(t), len(x)))
    
    def predict_whole_grid(self, a, x, t):
        """When we want to predict on the whole grid, we simply use the operator network's output, without the energy net.

        Args:
            a (M+1,): input function
            x (M+1,): spatial grid
            t (N+1,): temporal grid

        Returns:
            u_pred (N+1, M+1): prediction at the given grid points.
        """
        𝒢δℋ = self(a, x, t) # shape (Nt, Nx)
        
        return 𝒢δℋ
    
    def predict_whole_grid_batch(self, a, x, t):
        return vmap(self.predict_whole_grid, (0, None, None))(a,x,t)
    
    
    def integrate(self, a, x, t):
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
        
        vector_field = lambda t, y, args: vmap(self, (None, 0, None))(a, args, t)
        term = ODETerm(vector_field)
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

    def integrate2(self, a, x, t):
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
        # predict 𝒢δℋ (=u_t) over the whole grid
        u_t = self.predict_whole_grid(a, x, t)
        return a + cumulative_simpson(u_t, x=t, axis=0, initial=0)
    
def compute_energy_loss(model, a, u, key):
    """Computes the loss of the model.
    Returns the l2 loss, averaged over the batch. The loss is computed by randomly selecting query points from the input data and evaluating the model at those points.

    Args:
        model (eqx.Module): The model to evaluate.
        a (batch, number_of_sensors): The input data.
        u (batch, num_query_points): The ground truth data at the query points.

    Returns:
        loss (scalar): The loss of the model for the given batch.
    """
    batch_size = len(u)
            
    # cannot sample x and t randomly, since they are inputs to the operator net, which assumes equidistant grids
    
    𝒢δℋ = vmap(model.predict_whole_grid, (0, None, None))(a, Trainer.x, Trainer.t) # (batch, Nt, Nx)
    u_t = vmap(model.u.u_t, (0, None, None))(a, Trainer.x, Trainer.t) # (batch, Nt, Nx)
    if model.F.is_self_adaptive:
        # we need to sample the time points in accordance with the self-adaptive weights
        λ = model.self_adaptive.all_with_mask()[None,:,None]
        energy_loss = jnp.mean(jnp.sqrt(jnp.sum((λ * jnp.square(u_t - 𝒢δℋ)).reshape(batch_size,-1), axis=1)))
    else:
        energy_loss = jnp.mean(jnp.linalg.norm((u_t - 𝒢δℋ).reshape(batch_size,-1), 2, 1))
        
    return energy_loss

def compute_loss(model, a, u, key):
    return compute_operator_loss(model.u, a, u, key) + compute_energy_loss(model, a, u, key)*model.F.energy_penalty

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
    # each has shape (batch_size, Nt, Nx)
    u_pred = model.u.predict_whole_grid_batch(a, Trainer.x, Trainer.t)
    𝒢δℋ = model.predict_whole_grid_batch(a, Trainer.x, Trainer.t)
    u_t = vmap(model.u.u_t, (0, None, None))(a, Trainer.x, Trainer.t) # compute u_t in original scale
    
    #compute the loss 
    batch_size = len(u)
    u_norms = jnp.linalg.norm(u.reshape(batch_size, -1), 2, 1)
    diff_norms = jnp.linalg.norm((u - u_pred).reshape(batch_size, -1), 2, 1)
        
    operator_loss = jnp.mean(diff_norms/u_norms)
        
    energy_loss = jnp.mean(jnp.linalg.norm((u_t-𝒢δℋ).reshape(batch_size, -1), 2, 1))
        
    loss = operator_loss + energy_loss*model.F.energy_penalty
    return loss