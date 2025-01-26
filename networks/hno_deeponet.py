import jax
#jax.config.update("jax_enable_x64", True)
import equinox as eqx
from jax import random, grad, value_and_grad, vmap
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type
from .energynet import EnergyNet, EnergyNetHparams
from .deeponet import DeepONet as OperatorNet, compute_loss as compute_operator_loss, Hparams as OperatorHparams
import sys
sys.path.append("..")
from utils.trainer import Trainer

from quadax import cumulative_simpson, simpson
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
from scipy.interpolate import Akima1DInterpolator
from traditional_solvers import Dx

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
        """
        Input:
            x: scalar
            t: scalar
        Output:
            𝒢δℋ(x,t): scalar, at x=x and t=t
        """
        
        u = lambda x : self.u.decode_u(self.u(a, x, t)) # u in original scale (function of x for ease of notation)
        u_x = lambda x : self.u.u_x(a, x, t) # u_x in original scale
        
        dFdu = grad(self.F) # ∂F/∂u in original scale
        dFdu_x = lambda x : grad(self.F, 1)(u(x), u_x(x)) # ∂F/∂u in original scale
        
        δℋ = lambda x : dFdu(u(x), u_x(x)) - grad(dFdu_x)(x)/self.u.x_std # δℋ/δu in original scale
        𝒢δℋ =  -grad(δℋ)(x)/self.u.x_std  # 𝒢 δℋ/δu in original scale
        
        return 𝒢δℋ
    
    def multiple_query_points_one_a(self, a, x, t):
        return vmap(self, (None, 0, 0))(a, x, t)
    
    def 𝒢δℋ_whole_grid(self, a, x, t):
        """Predicts the solution at the whole grid.
        Args:
            a (M+1,): input function
            x (M+1,): spatial grid
            t (N+1,): temporal grid

        Returns:
            𝒢δℋ (N+1, M+1): prediction at the given grid points.
        """
        y = jnp.stack(jnp.meshgrid(x, t), axis=-1).reshape(-1, 2)
        return jax.lax.map(lambda y: self(a, y[0], y[1]), y, batch_size=1000).reshape(len(t), len(x))
    
    def 𝒢δℋ_whole_grid_jit(self, a, x, t):
        """Predicts the solution at the whole grid.
        Args:
            a (M+1,): input function
            x (M+1,): spatial grid
            t (N+1,): temporal grid

        Returns:
            𝒢δℋ (N+1, M+1): prediction at the given grid points.
        """
        y = jnp.stack(jnp.meshgrid(x, t), axis=-1).reshape(-1, 2)
        return jax.lax.map(eqx.filter_jit(lambda y: self(a, y[0], y[1])), y, batch_size=1000).reshape(len(t), len(x))
    
    def 𝒢δℋ_whole_grid_batch(self, a, x, t):
        return vmap(self.𝒢δℋ_whole_grid, (0, None, None))(a,x,t)

    def u_integrated_simpson(self, a, x, t):
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
        𝒢δℋ = self.𝒢δℋ_whole_grid(a, x, t)
        u0 = self.u.decode_u(a)[None,:]
        dt = self.u.decode_t(t[1])
        return cumulative_simpson(𝒢δℋ, dx=dt, axis=0, initial=u0)
    
    def u_integrated_akima(self, a, x, t):
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
        𝒢δℋ = self.𝒢δℋ_whole_grid(a, x, t)
        u0 = self.u.decode_u(a)
        𝒢δℋ_interp = Akima1DInterpolator(t, 𝒢δℋ, axis=0)
        return 𝒢δℋ_interp.antiderivative()(t) + u0
    
    def u_integrated_gauss(self, a, x, t):
        """Since the model predicts 𝒢δℋ (=u_t),
        we have to integrate the prediction to get u.

        Args:
            a (Mp1,): initial condition
            x (Mp1,): spatial grid
            t (Np1,): temporal grid

        Returns:
            u (Np1, Mp1): prediction for the given grid
        """
        # predict 𝒢δℋ (=u_t) for all x, at scalar t
        𝒢δℋ = lambda t : vmap(self, (None, 0, None))(a, x, self.u.encode_t(t))
        u0 = self.u.decode_u(a)
        return gauss_legendre_6(𝒢δℋ, u0, self.u.decode_t(t))
    
    def u_integrated_gauss2(self, a, x, t):
        """Since the model predicts 𝒢δℋ (=u_t),
        we have to integrate the prediction to get u.

        Args:
            a (Mp1,): initial condition
            x (Mp1,): spatial grid
            t (Np1,): temporal grid

        Returns:
            u (Np1, Mp1): prediction for the given grid
        """
        # predict 𝒢δℋ (=u_t) for all x, at scalar t

        dx = self.u.decode_x(x[1])        
        def 𝒢δℋ(t, u):
            u_x = Dx(u, dx)  
            dFdu = vmap(grad(self.F))(u, u_x) # ∂F/∂u in original scale
            dFdu_x = vmap(grad(self.F, 1))(u, u_x) # ∂F/∂u in original scale
            
            δℋ = dFdu - Dx(grad(dFdu_x), dx)/self.u.x_std # δℋ/δu in original scale
            return -Dx(δℋ, dx)/self.u.x_std  # 𝒢 δℋ/δu in original scale
            
        u0 = self.u.decode_u(a)
        return gauss_legendre_6(𝒢δℋ, u0, self.u.decode_t(t))
    
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
    batch_size, Np1, Mp1 = u.shape
            
    # Select random query indices
    t_key, x_key = random.split(key, 2)
    # Randomly sample 100 x-values between [0, 1]
    
    x_samples = model.u.encode_x(random.uniform(x_key, (batch_size, model.F.num_query_points), maxval=20.)) #(batch, num_query_points)
    if model.F.is_self_adaptive:
        # we need to sample the time points in accordance with the self-adaptive weights
        λ_idx = random.randint(t_key, (batch_size, model.F.num_query_points), 0, model.F.self_adaptive.λ_shape) # (batch, num_query_points)
        t_samples = model.u.encode_t(jnp.linspace(0, 2, model.F.self_adaptive.λ_shape, endpoint=False)[λ_idx]) # (batch, num_query_points)
        λ = model.F.self_adaptive(λ_idx) # (batch, num_query_points)
        
        𝒢δℋ = vmap(model.multiple_query_points_one_a)(a, x_samples, t_samples)
        u_t = vmap(vmap(model.u.u_t, (None, 0, 0)))(a, x_samples, t_samples)
        
        energy_loss = jnp.sqrt(jnp.sum((λ * jnp.square(u_t - 𝒢δℋ)).reshape(batch_size,-1), axis=1))
    else:
        t_samples = model.u.encode_t(random.uniform(t_key, (batch_size, model.F.num_query_points), maxval=2.)) #(batch, num_query_points)
    
        𝒢δℋ = vmap(model.multiple_query_points_one_a)(a, x_samples, t_samples)
        u_t = vmap(vmap(model.u.u_t, (None, 0, 0)))(a, x_samples, t_samples) # compute u_t in original scale   

        energy_loss = jnp.mean(jnp.linalg.norm((u_t - 𝒢δℋ).reshape(batch_size,-1), 2, 1))
        
    return energy_loss

def compute_loss(model, a, u, key):
    operator_key, energy_key = random.split(key)
    return compute_operator_loss(model.u, a, u, operator_key) + compute_energy_loss(model, a, u, energy_key)*model.F.energy_penalty

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
    
    # each has shape (batch_size, Nt, Nx)
    u_pred = model.u.predict_whole_grid_batch(a, Trainer.x, Trainer.t)
    𝒢δℋ = model.𝒢δℋ_whole_grid_batch(a, Trainer.x, Trainer.t)
    u_t = vmap(model.u.u_t_whole_grid, (0, None, None))(a, Trainer.x, Trainer.t) # compute u_t in original scale
    
    #compute the loss 
    batch_size = len(u)
    u_norms = jnp.linalg.norm(u.reshape(batch_size, -1), 2, 1)
    diff_norms = jnp.linalg.norm((u - u_pred).reshape(batch_size, -1), 2, 1)
        
    operator_loss = jnp.mean(diff_norms/u_norms)
        
    energy_loss = jnp.mean(jnp.linalg.norm((u_t-𝒢δℋ).reshape(batch_size, -1), 2, 1))
        
    loss = operator_loss + energy_loss*model.F.energy_penalty
    return loss


def gauss_legendre_6(f, u0, t):
    """
    Integrates the ODE system using the Gauss-Legendre method of order 6.
    Implementation follows "IV.8 Implementation of Implicit Runge-Kutta Methods" in 
    "Solving Ordinary Differential Equations II" by Hairer and Wanner

    Args:
      f: The right-hand side function of the ODE system.
      u0: Initial condition.
      dt: Time step.
      t: Array of time points.
      args: Additional arguments to pass to f.
      rtol: Relative tolerance for the nonlinear solver.
      atol: Absolute tolerance for the nonlinear solver.
      

    Returns:
      An array of solution values at the given time points.
    """
    dt = t[1] - t[0]
    
    c = jnp.array([0.5 - jnp.sqrt(15)/10, 0.5, 0.5 + jnp.sqrt(15)/10])
    A = jnp.array([[5/36, 2/9-jnp.sqrt(15)/15, 5/36-jnp.sqrt(15)/30],
                    [5/36+jnp.sqrt(15)/24, 2/9, 5/36-jnp.sqrt(15)/24],
                    [5/36+jnp.sqrt(15)/30, 2/9+jnp.sqrt(15)/15, 5/36]])
    d = jnp.array([5/3, -4/3, 5/3])
    
    @jax.jit
    def step(un, tn): 
        z0 = dt*(A[0,0]*f(tn + c[0]*dt) + A[0,1]*f(tn + c[1]*dt) + A[0,2]*f(tn + c[2]*dt))
        z1 = dt*(A[1,0]*f(tn + c[0]*dt) + A[1,1]*f(tn + c[1]*dt) + A[1,2]*f(tn + c[2]*dt))
        z2 = dt*(A[2,0]*f(tn + c[0]*dt) + A[2,1]*f(tn + c[1]*dt) + A[2,2]*f(tn + c[2]*dt))
        z_next = jnp.array([z0, z1, z2])
    
        u_next = un + jnp.dot(d, z_next)
            
        return u_next, un

    _, u_arr = jax.lax.scan(step, u0, t)
    return u_arr