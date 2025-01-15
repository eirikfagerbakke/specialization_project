import jax
#jax.config.update("jax_enable_x64", True)
import equinox as eqx
from jax import random, grad, value_and_grad, vmap
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type
from .energynet import EnergyNet, EnergyNetHparams
from .modified_deeponet import ModifiedDeepONet as OperatorNet, Hparams as OperatorHparams, compute_loss as compute_operator_loss
import sys
sys.path.append("..")
from utils.trainer import Trainer

from quadax import cumulative_simpson, simpson
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
from scipy.interpolate import Akima1DInterpolator

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
        """The input and output of the HINO is the same as the input of the OperatorNet.
        Input:
            x: scalar
            t: scalar
        Output:
            u_t(x,t): scalar, at x=x and t=t
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
        return jax.lax.map(eqx.filter_jit(lambda y: self(a, y[0], y[1])), y, batch_size=1000).reshape(len(t), len(x))
    
    def 𝒢δℋ_x_whole_grid(self, a, x, t):
        """Predicts the solution at the whole grid.
        Args:
            a (M+1,): input function
            x (M+1,): spatial grid
            t (N+1,): temporal grid

        Returns:
            𝒢δℋ (N+1, M+1): prediction at the given grid points.
        """
        y = jnp.stack(jnp.meshgrid(x, t), axis=-1).reshape(-1, 2)
        return jax.lax.map(eqx.filter_jit(lambda y: grad(self, 1)(a, y[0], y[1])), y, batch_size=1000).reshape(len(t), len(x))/self.u.x_std
    
    def 𝒢δℋ_whole_grid_batch(self, a, x, t):
        return vmap(self.𝒢δℋ_whole_grid, (0, None, None))(a,x,t)
    
    def 𝒢δℋ_x_whole_grid_batch(self, a, x, t):
        return vmap(self.𝒢δℋ_x_whole_grid, (0, None, None))(a,x,t)

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
    
    def u_x_integrated_simpson(self, a, x, t):
        """Since the model predicts 𝒢δℋ (=u_t),
        we have to integrate 𝒢δℋ_x to get u.
        Does so using cumulative_simpson from scipy.integrate.

        Args:
            a (Mp1,): initial condition
            x (Mp1,): spatial grid
            t (Np1,): temporal grid

        Returns:
            u (Np1, Mp1): prediction for the given grid
        """
        # predict 𝒢δℋ (=u_t) over the whole grid (vmap over temporal and spatial dimensions)
        𝒢δℋ_x = self.𝒢δℋ_x_whole_grid(a, x, t)
        u0_x = vmap(self.u.u_x, (None, 0, None))(a, x, t[0])[None,:]
        dt = self.u.decode_t(t[1])
        return cumulative_simpson(𝒢δℋ_x, dx=dt, axis=0, initial=u0_x)
    
    def Hamiltonian_simpson(self, a, x, t):
        u_integrated = self.u_integrated_simpson(a, x, t)
        u_x_integrated = self.u_x_integrated_simpson(a, x, t)
        
        energy_density = self.F.predict_whole_grid(u_integrated, u_x_integrated)
        return simpson(energy_density, dx=self.u.decode_x(x[1]), axis=1)
    
    
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
    
    def u_x_integrated_akima(self, a, x, t):
        """Since the model predicts 𝒢δℋ (=u_t),
        we have to integrate 𝒢δℋ_x to get u.
        Does so using cumulative_simpson from scipy.integrate.

        Args:
            a (Mp1,): initial condition
            x (Mp1,): spatial grid
            t (Np1,): temporal grid

        Returns:
            u (Np1, Mp1): prediction for the given grid
        """
        # predict 𝒢δℋ (=u_t) over the whole grid (vmap over temporal and spatial dimensions)
        𝒢δℋ_x = self.𝒢δℋ_x_whole_grid(a, x, t)
        u0_x = vmap(self.u.u_x, (None, 0, None))(a, x, t[0])
        𝒢δℋ_x_interp = Akima1DInterpolator(t, 𝒢δℋ_x, axis=0)
        return 𝒢δℋ_x_interp.antiderivative()(t) + u0_x
    
    def Hamiltonian_akima(self, a, x, t):
        u_integrated = self.u_integrated_akima(a, x, t)
        u_x_integrated = self.u_x_integrated_akima(a, x, t)
        
        energy_density = self.F.predict_whole_grid(u_integrated, u_x_integrated)
        return simpson(energy_density, dx=self.u.decode_x(x[1]), axis=1)
    
    def u_integrated_gauss(self, a, x, t):
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
        # predict 𝒢δℋ (=u_t) for all x, at scalar t
        𝒢δℋ = lambda t : vmap(self, (None, 0, None))(a, x, t)
        u0 = self.u.decode_u(a)
        return gauss_legendre_4(𝒢δℋ, u0, t)
    
    def u_x_integrated_gauss(self, a, x, t):
        """Since the model predicts 𝒢δℋ (=u_t),
        we have to integrate 𝒢δℋ_x to get u.
        Does so using cumulative_simpson from scipy.integrate.

        Args:
            a (Mp1,): initial condition
            x (Mp1,): spatial grid
            t (Np1,): temporal grid

        Returns:
            u (Np1, Mp1): prediction for the given grid
        """
        # predict 𝒢δℋ (=u_t) over spatial points, at scalar t
        𝒢δℋ_x = lambda t : vmap(grad(self, 1), (None, 0, None))(a, x, t)
        u0_x = vmap(self.u.u_x, (None, 0, None))(a, x, t[0])
        return gauss_legendre_4(𝒢δℋ_x, u0_x, t)
    
    def Hamiltonian_gauss(self, a, x, t):
        u_integrated = self.u_integrated_gauss(a, x, t)
        u_x_integrated = self.u_x_integrated_gauss(a, x, t)
        
        energy_density = self.F.predict_whole_grid(u_integrated, u_x_integrated)
        return simpson(energy_density, dx=self.u.decode_x(x[1]), axis=1)
    
    def u_integrated_diffrax(self, a, x, t):
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
        # predict 𝒢δℋ (=u_t) for all x, at scalar t
        𝒢δℋ = lambda t, y, args : vmap(self, (None, 0, None))(a, x, t)
 
        term = ODETerm(𝒢δℋ)
        solver = Tsit5()
        u0 = self.u.decode_u(a)
        solution = diffeqsolve(term, solver, t0=t[0].item(), t1=t[-1].item(), dt0=t[1].item()-t[0].item(), y0=u0, saveat=SaveAt(ts=t))
        return solution.ys
    
    def u_x_integrated_diffrax(self, a, x, t):
        """Since the model predicts 𝒢δℋ (=u_t),
        we have to integrate 𝒢δℋ_x to get u.
        Does so using cumulative_simpson from scipy.integrate.

        Args:
            a (Mp1,): initial condition
            x (Mp1,): spatial grid
            t (Np1,): temporal grid

        Returns:
            u (Np1, Mp1): prediction for the given grid
        """
        # predict 𝒢δℋ (=u_t) over spatial points, at scalar t
        𝒢δℋ_x = lambda t, y, args : vmap(grad(self, 1), (None, 0, None))(a, x, t)
        term = ODETerm(𝒢δℋ_x)
        solver = Tsit5()
        u0_x = vmap(self.u.u_x, (None, 0, None))(a, x, t[0])
        solution = diffeqsolve(term, solver, t0=t[0].item(), t1=t[-1].item(), dt0=t[1].item()-t[0].item(), y0=u0_x, saveat=SaveAt(ts=t))
        return solution.ys
    
    def Hamiltonian_diffrax(self, a, x, t):
        u_integrated = self.u_integrated_diffrax(a, x, t)
        u_x_integrated = self.u_x_integrated_diffrax(a, x, t)
        
        energy_density = self.F.predict_whole_grid(u_integrated, u_x_integrated)
        return simpson(energy_density, dx=self.u.decode_x(x[1]), axis=1)
        
    
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
        
        energy_loss = jnp.mean(jnp.sqrt(jnp.sum((λ * jnp.square(u_t - 𝒢δℋ)).reshape(batch_size,-1), axis=1)))
    else:
        t_samples = model.u.encode_t(random.uniform(t_key, (batch_size, model.F.num_query_points), maxval=2.)) #(batch, num_query_points)
    
        𝒢δℋ = vmap(model.multiple_query_points_one_a)(a, x_samples, t_samples)
        u_t = vmap(vmap(model.u.u_t, (None, 0, 0)))(a, x_samples, t_samples) # compute u_t in original scale   

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
    u_t = vmap(model.u.u_t_whole_grid, (0, None, None))(a, Trainer.x, Trainer.t) # compute u_t in original scale
    
    #compute the loss 
    batch_size = len(u)
    u_norms = jnp.linalg.norm(u.reshape(batch_size, -1), 2, 1)
    diff_norms = jnp.linalg.norm((u - u_pred).reshape(batch_size, -1), 2, 1)
        
    operator_loss = jnp.mean(diff_norms/u_norms)
        
    energy_loss = jnp.mean(jnp.linalg.norm((u_t-𝒢δℋ).reshape(batch_size, -1), 2, 1))
        
    loss = operator_loss + energy_loss*model.F.energy_penalty
    return loss


from functools import partial
import optimistix as optx
@partial(jax.jit, static_argnums=(0,))
def gauss_legendre_4(f, u0, t, rtol=1e-8, atol=1e-8, max_steps = 20):
    """
    Integrates the ODE system using the Gauss-Legendre method of order 4.
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
    c = jnp.array([0.5 - jnp.sqrt(3)/6, 0.5 + jnp.sqrt(3)/6])
    A = jnp.array([[0.25, 0.25 - jnp.sqrt(3)/6], 
                   [0.25 + jnp.sqrt(3)/6, 0.25]])
    d = jnp.array([-jnp.sqrt(3), jnp.sqrt(3)])
    
    dt = t[1] - t[0]
    
    def q(x, z_next):
        return z_next[0]*(x-c[1])/(c[0]-c[1])*x/c[0] + z_next[1]*(x-c[0])/(c[1]-c[0])*x/c[1]
    
    def step(carry, tn): 
        un, z_guess = carry
        
        def eq(z, args):
            f0 = f(tn + c[0]*dt)
            f1 = f(tn + c[1]*dt)
            z1 = dt*(A[0,0] * f0 + A[0,1]*f1)
            z2 = dt*(A[1,0] * f0 + A[1,1]*f1)
            return z - jnp.array([z1, z2])
        
        solver = optx.Chord(rtol, atol)
        z_next = optx.root_find(eq, solver, z_guess, None, throw=False, max_steps = max_steps).value
        u_next = un + jnp.dot(d, z_next)
        
        z_guess = q(1+c[:,None], z_next)+un-u_next
        return (u_next, z_guess), un

    z_guess = jnp.zeros((2, u0.shape[0]))
    _, u_arr = jax.lax.scan(step, (u0, z_guess), t)
    return u_arr