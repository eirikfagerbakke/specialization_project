from dataclasses import dataclass
import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap, jacrev, jacfwd
import equinox as eqx
from typing import Callable, Union
from jaxtyping import Array

from ._abstract_operator_net import AbstractOperatorNet, AbstractHparams
import sys
sys.path.append("..")
from utils.trainer import Trainer
from utils.model_utils import param_labels, conjugate_grads_transform
import optax
from optax.contrib import reduce_on_plateau

@dataclass(kw_only=True, frozen=True)
class Hparams(AbstractHparams):
    # network parameters
    n_blocks: int # number of FNO blocks
    hidden_dim: int # dimension of the hidden layers
    modes_max: int # maximum number of modes to keep in the spectral convolutions
    
class SpectralConv1d(eqx.Module):
    weights: jax.Array
    in_channels: int
    out_channels: int
    max_modes: int

    def __init__(self, in_channels, out_channels, max_modes, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes = max_modes

        init_std = jnp.sqrt(2.0 / (self.in_channels))
        self.weights = random.normal(key, (self.in_channels, self.out_channels, self.max_modes), dtype = jnp.complex64)*init_std


    def __call__(self,v):
        channels, spatial_points = v.shape

        # shape of v_hat is (in_channels, spatial_points//2+1)
        v_hat = jnp.fft.rfft(v) #rfft over the last axis
        
        # shape of v_hat_trunc is (in_channels, self.max_modes)
        v_hat_trunc = v_hat[:, :self.max_modes]
        
        # shape of out_hat is (out_channels, self.modes)
        out_hat = jnp.einsum("iM,ioM->oM", v_hat_trunc, self.weights) #i: in, o: out, M: modes

        # shape of out_hat is (out_channels, spatial_points//2+1)
        out_hat_padded = jnp.zeros((self.out_channels, v_hat.shape[-1]), dtype = v_hat.dtype)
        out_hat_padded = out_hat_padded.at[:, :self.max_modes].set(out_hat)

        out = jnp.fft.irfft(out_hat_padded, n=spatial_points)

        return out
    
    def Dx(self, v, dx):
        channels, Nx = v.shape

        # Compute 2D Fourier transform
        v_hat = jnp.fft.rfft(v)
        v_hat_trunc = v_hat[:, :self.max_modes]
        
        out_hat = jnp.einsum("iM,ioM->oM", v_hat_trunc, self.weights)
        out_hat_padded = jnp.zeros((self.out_channels, Nx//2 + 1), dtype = v_hat.dtype)
        out_hat_padded = out_hat_padded.at[:, :self.max_modes].set(out_hat)

        # Apply differentiation in Fourier space
        kx = jnp.fft.rfftfreq(Nx, dx) * 2 * jnp.pi
        dx_hat = 1j * kx * out_hat_padded

        # Transform back to physical space
        dx_val = jnp.fft.irfft(dx_hat, n=Nx)
        return dx_val
    
class Bypass(eqx.Module):
    weights: Array

    def __init__(self, in_channels, out_channels, key):
        init_std = jnp.sqrt(2.0 / in_channels)
        self.weights = random.normal(key, (out_channels, in_channels))*init_std

    def __call__(self, v):
        return jnp.tensordot(self.weights, v, axes=1)
    
class FNOBlock1d(eqx.Module):
    spectral_conv: SpectralConv1d
    bypass: Bypass
    activation: Callable
    bias: Array
    
    def __init__(self, in_channels, out_channels, max_modes, activation, key = random.PRNGKey(0)):
        spectral_conv_key, bypass_key = jax.random.split(key)
        self.spectral_conv = SpectralConv1d(
            in_channels,
            out_channels,
            max_modes,
            spectral_conv_key,
        )
        
        self.bypass = Bypass(in_channels, out_channels, bypass_key)
        self.activation = activation
        self.bias = jnp.zeros((out_channels, 1))

    def __call__(self, v):
        # shape of v is (in_channels, spatial_points)
        return self.activation(self.spectral_conv(v) + self.bypass(v) + self.bias)

class FNO1d(AbstractOperatorNet):
    """Maps a function on a spatial domain at time t to the solution on the same spatial domain at time t.
    
    Input: (a(x,t), x, t)
        Shapes:
            a(x,t): (n, )
            x: (n, )
            t: (1, )
    Outputs: u(x,t)
        Shapes:
            u(x,t): (n, )
    """
    lifting: eqx.nn.Conv1d
    dynamic_fno_blocks: FNOBlock1d
    static_fno_blocks: FNOBlock1d
    projection: eqx.nn.Conv1d
    multiplier: float
    activation: Callable
    last_spectral_conv: SpectralConv1d
    last_bias: Array

    def __init__(self, hparams: Union[Hparams, dict]):
        if isinstance(hparams, dict):
            hparams = Hparams(**hparams)
        super().__init__(hparams)
        
        key = random.key(hparams.seed)
        keys = random.split(key, hparams.n_blocks + 2)
        
        self.lifting = eqx.nn.Conv1d(in_channels = 4, out_channels = hparams.hidden_dim, kernel_size = 1, key=keys[0])

        fno_blocks = eqx.filter_vmap(lambda key: FNOBlock1d(hparams.hidden_dim,hparams.hidden_dim,hparams.modes_max,jax.nn.gelu,key=key))(keys[1:-2])
        self.dynamic_fno_blocks, self.static_fno_blocks = eqx.partition(fno_blocks, eqx.is_array)
        
        self.projection = eqx.nn.Conv1d(in_channels = hparams.hidden_dim, out_channels = 1, kernel_size = 1, key=keys[-2])
    
        self.multiplier = 2*jnp.pi/self.period
        
        self.activation = jax.nn.gelu
        
        self.last_spectral_conv = SpectralConv1d(hparams.hidden_dim, hparams.hidden_dim, hparams.modes_max, keys[-1])
        
        self.last_bias = jnp.zeros((hparams.hidden_dim, 1))
    
    def stack_input(self, a, x, t):
        t_rep = jnp.full_like(x, t)
        x_cos = jnp.cos(self.multiplier*x)
        x_sin = jnp.sin(self.multiplier*x)
        return jnp.stack([a, x_cos, x_sin, t_rep])

    @jax.custom_jvp
    def __call__(self,a,x,t):        
        v = self.stack_input(a, x, t)
        
        v = self.lifting(v)
        
        def f(v, dynamic_fno_block):
            fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
            return fno_block(v), None
        
        v, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
        
        v_Lm1, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
        v_L = self.last_spectral_conv(v_Lm1) + self.last_bias

        v = self.projection(self.activation(v_L))[0]
        return v
    
    @__call__.defjvp
    def __call__jvp(self, a, x, t):
        v = self.stack_input(a, x, t)
        
        v = self.lifting(v)
        
        def f(v, dynamic_fno_block):
            fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
            return fno_block(v), None
        
        v, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
        
        v_Lm1, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
        v_L = self.last_spectral_conv(v_Lm1) + self.last_bias

        v = self.projection(self.activation(v_L))[0]
        return v
    
    def predict_whole_grid(self, a, x, t):
        """Utility function for predicting the output over the whole grid.
        Since FNO1d takes a vector x and scalar t as input, and maps to
        u(x, t) at t=t, shape: (M+1,).

        Args:
            a (M+1,): input function
            x (M+1,): spatial grid
            t (N+1,): temporal grid

        Returns:
            pred (N+1, M+1): prediction over the whole grid given by x and t
        """
        
        # we vectorize over the time input
        return vmap(self, (None, None, 0))(a, x, t)
    
    def predict_whole_grid_batch(self, a, x, t):
        """To predict over the whole grid on a batch of initial conditions."""
        return vmap(self.predict_whole_grid, (0, None, None))(a, x, t)
    
    def Dx(self, a, x, t):
        v = self.stack_input(a, x, t)
        
        v = self.lifting(v)
        
        def f(v, dynamic_fno_block):
            fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
            return fno_block(v), None
        
        v_Lm1, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
        v_L = self.last_spectral_conv(v_Lm1) + self.last_bias
        
        # compute derivatives
        # all have shape (hidden_channels, t_points, x_points)
        d_Q_d_v_L = jax.grad(lambda v_L : jnp.sum(self.projection(self.activation(v_L))))(v_L)
        d_K_L_dx = self.last_spectral_conv.Dx(v_Lm1, x[1]-x[0])
        
        return (d_Q_d_v_L * d_K_L_dx).sum(axis=0)
    
    def Dt(self, a, x, t):
        # self(a,x,t) shape : (Nx,)
        # x shape : (Nx,)
        return jacfwd(self, 2)(a, x, t)
    
    def Dx_whole_grid(self, a, x, t):
        return vmap(self.Dx, (None, None, 0))(a, x, t)
    
    def Dt_whole_grid(self, a, x, t):
        return vmap(self.Dt, (None, None, 0))(a, x, t)
    
    def u_t(self, a, x, t):
        return self.Dt(a, x, t)*self.u_std/self.t_std
    
    def u_t_whole_grid(self, a, x, t):
        return self.Dt_whole_grid(a, x, t)*self.u_std/self.t_std
    
    def u_x(self, a, x, t):
        return self.Dx(a, x, t)*self.u_std/self.x_std
    
    def u_x_whole_grid(self, a, x, t):
        return self.Dx_whole_grid(a, x, t)*self.u_std/self.x_std
    
    
class HparamTuning:
    def __init__(self, train_loader, val_loader, z_score_data, hparams=None, **trainer_kwargs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.λ_shape = 64
        self.trainer_kwargs = trainer_kwargs
        self.z_score_data = z_score_data
        self.hparams = hparams

    def __call__(self, trial):
        # Define the hyperparameters to tune
        #n_blocks = trial.suggest_int("n_blocks", 3, 6) # number of FNO blocks
        #hidden_dim = trial.suggest_int("hidden_dim", 50, 150) # dimension of the hidden layers
        #modes_max = trial.suggest_int("modes_max", 8, 32)
        #learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1) 
        
        # Optimizer used for the "regular" network parameters
        PATIENCE = 5 # Number of epochs with no improvement after which learning rate will be reduced
        COOLDOWN = 0 # Number of epochs to wait before resuming normal operation after the learning rate reduction
        FACTOR = 0.5  # Factor by which to reduce the learning rate:
        RTOL = 1e-4  # Relative tolerance for measuring the new optimum:
        ACCUMULATION_SIZE = 200 # Number of iterations to accumulate an average value:
        θ_optimizer = optax.chain(
            conjugate_grads_transform(),
            optax.adam(self.hparams["learning_rate"]),
            reduce_on_plateau(
                patience=PATIENCE,
                cooldown=COOLDOWN,
                factor=FACTOR,
                rtol=RTOL,
                accumulation_size=ACCUMULATION_SIZE,
            ),
        )
        
        # Self-adaptive hyperparameters       
        #is_self_adaptive = trial.suggest_categorical("self_adaptive", [True, False])
        is_self_adaptive = True
        λ_smooth_or_sharp = None
        if is_self_adaptive:
            λ_learning_rate=trial.suggest_float("λ_learning_rate", 1e-4, 1e-1)
            λ_mask=trial.suggest_categorical("λ_mask", ["exponential", "polynomial", "logistic"])
            λ_learnable=trial.suggest_categorical("λ_learnable", [True, False])
            if not λ_learnable:
                λ_smooth_or_sharp = trial.suggest_categorical("λ_smooth_or_sharp", ["smooth", "sharp"])
            λ_optimizer = optax.chain(optax.adam(λ_learning_rate), optax.scale(-1.))
            opt = optax.multi_transform({'θ': θ_optimizer, 'λ': λ_optimizer}, param_labels=param_labels)
        else:
            λ_learnable = None
            λ_learning_rate = None 
            λ_mask = None
            opt = θ_optimizer
        
        # Initialize the model and the trainer
        hparams = Hparams(
            #n_blocks=n_blocks,
            #hidden_dim=hidden_dim,
            #modes_max=modes_max,
            #learning_rate=learning_rate,
            λ_learning_rate = λ_learning_rate,
            λ_shape = self.λ_shape,
            λ_smooth_or_sharp = λ_smooth_or_sharp,
            λ_learnable = λ_learnable,
            λ_mask = λ_mask,
            **self.hparams,
            **self.z_score_data
        )
        
        model = FNO1d(hparams)
        if replicated:=self.trainer_kwargs.get("replicated"):
            model = eqx.filter_shard(model, replicated)
        opt_state = opt.init(eqx.filter([model], eqx.is_array))

        trainer = Trainer(model, opt, opt_state, self.train_loader, self.val_loader, trial=trial, **self.trainer_kwargs)
        trainer()
        best_val_loss = trainer.best_val_loss
        
        del trainer, opt_state, opt
        return best_val_loss
    
    
def compute_loss(model, a, u, key):
    """Computes the MSE loss of the model, by randomly selecting a time index and evaluating the model at that time.

    Args:
        model (eqx.Model)
        a (batch, M+1)
        u (batch, N+1, M+1)
        key : jax PRNGKey

    Returns:
        loss
    """
    batch_size, Np1, Mp1 = u.shape
    
    # choose a single timepoint t to map for each sample, shape (batch_size,)
    t_idx = random.randint(key, (batch_size,), 0, Np1)
    
    # predict mapping from t=0 to t=t, shape (batch_size, Mp1)
    u_pred = vmap(model, (0, None, 0))(a, Trainer.x, Trainer.t[t_idx])
    # select the ground truth at the random time points, shape (batch_size, Mp1)
    u_ground_truth = u[jnp.arange(batch_size), t_idx,:]
        
    #compute the loss 
    u_norms = jnp.linalg.norm(u_ground_truth, 2, 1)
    if model.is_self_adaptive:
        λ = model.self_adaptive(t_idx)[:,None] # (batch_size, 1)
        diff_norms = jnp.sqrt(jnp.sum(λ * jnp.square(u_ground_truth - u_pred), axis=1))
    else:
        diff_norms = jnp.linalg.norm(u_ground_truth - u_pred, 2, 1)
        
    loss = jnp.mean(diff_norms/u_norms)
    
    a_pred = vmap(model, (0, None, None))(a, Trainer.x, Trainer.t[0])
    ic_loss = jnp.mean(jnp.linalg.norm(a - a_pred, 2, 1)/jnp.linalg.norm(a, 2, 1))
    return loss + ic_loss*0.5