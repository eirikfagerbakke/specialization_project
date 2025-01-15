import jax
#jax.config.update("jax_enable_x64", True)
import equinox as eqx
from jax import random, grad, value_and_grad, vmap
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type
from ..networks.energynet import EnergyNet, EnergyNetHparams
from ..networks.deeponet import DeepONet, compute_loss
from ..networks.deeponet import Hparams as DeepONetHparams
from ..networks.modified_deeponet import ModifiedDeepONet
from ..networks.modified_deeponet import Hparams as ModifiedDeepONetHparams
import sys
sys.path.append("..")
from utils.trainer import Trainer

@dataclass(frozen=True, kw_only=True)
class Hparams:
    energy_net: EnergyNetHparams
    operator_net: DeepONetHparams | ModifiedDeepONetHparams

class HINO_DON(eqx.Module):
    """A Hamiltonian PDE can be written on the form u_t = 𝒢 δℋ/δu.
    The operator network predicts the function u(x,t). From this function, we compute
    u_t and 𝒢 δℋ/δu using automatic differentiation.
    These terms should be equal, and can be added as a penalty term to the loss function.
    
    This gives us a network that is "informed" by the Hamiltonian structure of the PDE.
    """
    F : EnergyNet
    u : DeepONet | ModifiedDeepONet
    
    is_self_adaptive: bool = False
    
    def __init__(self, hparams = None, energy_net : eqx.Module = None, operator_net: DeepONet = None):
        if hparams is not None:
            energy_net = EnergyNet(hparams["energy_net"])
            operator_net = DeepONet(hparams["operator_net"])
        self.F = energy_net
        self.u = operator_net
        
        self.is_self_adaptive = self.F.is_self_adaptive or self.u.is_self_adaptive

    def __call__(self, a, x, t):
        """The input and output of the HINO is the same as the input of the OperatorNet.
        Input:
            x: scalar
            t: scalar
        Output:
            u_t(x,t): scalar, at x=x and t=t
        """     
        u_t = grad(self.u, 2)(a, x, t) # u and u_t, values   
        u = lambda x : self.u(a, x, t) # u(x), function in x only (for convenience)
        u_x = grad(u) # u_x(x), function
        
        dFdu = lambda x : grad(self.F, 0)(u(x), u_x(x)) # ∂F/∂u(x), function
        dFdu_x = lambda x : grad(self.F, 1)(u(x), u_x(x)) # ∂F/∂u_x(x), function
        δℋ = lambda x : dFdu(x) - grad(dFdu_x)(x) # δℋ/δu(x), function
        
        𝒢δℋ =  -grad(δℋ)(x) # 𝒢 δℋ/δu , value
        return 𝒢δℋ, u_t
    
    def multiple_query_points_one_a2(self, a, x, t):
        """The input and output of the HINO is the same as the input of the OperatorNet.
        Input:
            x: scalar
            t: scalar
        Output:
            u_t(x,t): scalar, at x=x and t=t
        """
        u_eval = self.u(a, x, t)
        def sum_multiple_query_points_one_a(a,x,t):
            return jnp.sum(self.u.multiple_query_points_one_a(a,x,t))
        u_t = grad(sum_multiple_query_points_one_a,2)(a,x,t) # u and u_t, values
        u = lambda x : self.u.multiple_query_points_one_a(a, x, t)
        u_x = lambda x : grad(sum_multiple_query_points_one_a,1)(a,x,t)
        
        dFdu = lambda x : grad(lambda u: jnp.sum(self.F(u, u_x(x))))(u(x)) # ∂F/∂u(x), function
        dFdu_x = lambda x : grad(lambda u_x: jnp.sum(self.F(u(x), u_x)))(u_x(x)) # ∂F/∂u_x(x), function
        δℋ = lambda x : dFdu(x) - grad(dFdu_x)(x) # δℋ/δu(x), function
        
        𝒢δℋ =  -grad(lambda x : jnp.sum(δℋ(x)))(x) # 𝒢 δℋ/δu , value
        return 𝒢δℋ, u_t
    
    def multiple_query_points_one_a(self, a, x, t):
        return vmap(self, (None, 0, 0))(a, x, t)
    
    def predict_whole_grid(self, a, x, t):
        """When we want to predict on the whole grid, we simply use the operator network's output, without the energy net.

        Args:
            a (M+1,): input function
            x (M+1,): spatial grid
            t (N+1,): temporal grid

        Returns:
            u_pred (N+1, M+1): prediction at the given grid points.
        """
        xx, tt = jnp.meshgrid(x, t)
        𝒢δℋ, u_t = self.multiple_query_points_one_a(a, xx.ravel(), tt.ravel())
        
        return 𝒢δℋ.reshape(xx.shape), u_t.reshape(xx.shape)
    
    def predict_whole_grid_batch(self, a, x, t):
        return vmap(self.predict_whole_grid, (0, None, None))(a,x,t)
    
def compute_loss_hino(model, a, u, key):
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
    
    operator_key, energy_key = random.split(key, 2)
    
    # Compute the operator loss
    operator_loss = compute_loss(model.u, a, u, operator_key)
            
    # Select random query indices
    t_key, x_key = random.split(energy_key, 2)
    # Randomly sample 100 x-values between [0, 1]
    x_samples = random.uniform(x_key, (batch_size, model.F.num_query_points))
    t_samples = random.uniform(t_key, (batch_size, model.F.num_query_points))
    
    
    
    𝒢δℋ, u_t = vmap(vmap(model, (None, 0, 0)))(a, x_samples, t_samples)    
    #if model.F.is_self_adaptive:
        #λ = model.F.self_adaptive(t_idx)
    #    energy_loss = jnp.mean(λ*jnp.square(u_t-𝒢δℋ))
    #else:
    energy_loss = jnp.mean(jnp.square(u_t-𝒢δℋ))
        
    loss = operator_loss + energy_loss*model.F.energy_penalty
    return loss

def evaluate_hino(model, a, u, key):
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
        
    # each has shape (batch_size, (N+1) * (M+1))
    u_pred = model.u.predict_whole_grid_batch(a, Trainer.x, Trainer.t)
    𝒢δℋ, u_t = model.predict_whole_grid_batch(a, Trainer.x, Trainer.t)
    
    #compute the loss 
    u_norms = jnp.linalg.norm(u.reshape(len(u), -1), 2, 1)
    diff_norms = jnp.linalg.norm((u - u_pred).reshape(len(u), -1), 2, 1)
        
    operator_loss = jnp.mean(diff_norms/u_norms)
        
    energy_loss = jnp.mean(jnp.square(u_t-𝒢δℋ))
        
    loss = operator_loss + energy_loss*model.F.energy_penalty
    return loss