from dataclasses import dataclass
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap, grad, value_and_grad
import equinox as eqx
from typing import Callable, Union
from jaxtyping import Array

from ._abstract_operator_net import AbstractOperatorNet, AbstractHparams
import sys
sys.path.append("..")
from utils.trainer import Trainer
from utils.model_utils import param_labels, conjugate_grads_transform
import optax
from networks._abstract_operator_net import AbstractOperatorNet, AbstractHparams
from optax.contrib import reduce_on_plateau
from interpax import interp1d

@dataclass(kw_only=True, frozen=True)
class Hparams(AbstractHparams):
    # network parameters
    n_blocks: int # number of FNO blocks
    hidden_dim: int # dimension of the hidden layers
    modes_max: int # maximum number of modes to keep in the spectral convolutions
    max_steps: int = 30 # maximum number of time steps to predict
    
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
    
    def Dxx(self, v, dx):
        channels, Nx = v.shape

        # Compute 2D Fourier transform
        v_hat = jnp.fft.rfft(v)
        v_hat_trunc = v_hat[:, :self.max_modes]
        
        out_hat = jnp.einsum("iM,ioM->oM", v_hat_trunc, self.weights)
        out_hat_padded = jnp.zeros((self.out_channels, Nx//2 + 1), dtype = v_hat.dtype)
        out_hat_padded = out_hat_padded.at[:, :self.max_modes].set(out_hat)

        # Apply differentiation in Fourier space
        kx = jnp.fft.rfftfreq(Nx, dx) * 2 * jnp.pi
        dx_hat = - kx**2 * out_hat_padded

        # Transform back to physical space
        dx_val = jnp.fft.irfft(dx_hat, n=Nx)
        return dx_val
    
    def Dxxx(self, v, dx):
        channels, Nx = v.shape

        # Compute 2D Fourier transform
        v_hat = jnp.fft.rfft(v)
        v_hat_trunc = v_hat[:, :self.max_modes]
        
        out_hat = jnp.einsum("iM,ioM->oM", v_hat_trunc, self.weights)
        out_hat_padded = jnp.zeros((self.out_channels, Nx//2 + 1), dtype = v_hat.dtype)
        out_hat_padded = out_hat_padded.at[:, :self.max_modes].set(out_hat)

        # Apply differentiation in Fourier space
        kx = jnp.fft.rfftfreq(Nx, dx) * 2 * jnp.pi
        dx_hat = -1j* kx**3 * out_hat_padded

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

class FNOTimeStepping(AbstractOperatorNet):
    """Maps a function on a spatial domain at time t to the solution on the same spatial domain at time t+dt.
    """
    lifting: eqx.nn.Conv1d
    projection: eqx.nn.Conv1d
    multiplier: float
    max_steps: int
    dynamic_fno_blocks: Array
    static_fno_blocks: Array
    last_spectral_conv: SpectralConv1d
    last_bias: Array
    activation: Callable 

    def __init__(self, hparams: Union[Hparams, dict]):
        if isinstance(hparams, dict):
            hparams = Hparams(**hparams)
        super().__init__(hparams)
        
        key = random.key(hparams.seed)
        keys = random.split(key, hparams.n_blocks + 2)
        
        self.lifting = eqx.nn.Conv1d(in_channels = 3, out_channels = hparams.hidden_dim, kernel_size = 1, key=keys[0])

        fno_blocks = eqx.filter_vmap(lambda key: FNOBlock1d(hparams.hidden_dim,hparams.hidden_dim,hparams.modes_max,jax.nn.gelu,key=key))(keys[1:-2])
        
        self.projection = eqx.nn.Conv1d(in_channels = hparams.hidden_dim, out_channels = 1, kernel_size = 1, key=keys[-2])
    
        self.multiplier = 2*jnp.pi/self.period
        
        self.max_steps = hparams.max_steps
        
        self.dynamic_fno_blocks, self.static_fno_blocks = eqx.partition(fno_blocks, eqx.is_array)
        
        self.last_spectral_conv = SpectralConv1d(hparams.hidden_dim, hparams.hidden_dim, hparams.modes_max, keys[-1])
        
        self.last_bias = jnp.zeros((hparams.hidden_dim, 1))
        
        self.activation = jax.nn.gelu
        
    def stack_input(self, a, x):
        x_cos = jnp.cos(self.multiplier*x)
        x_sin = jnp.sin(self.multiplier*x)
        return jnp.stack([a, x_cos, x_sin])
    
    def __call__(self,a,x,t):
        def step(v, _):
            # combine with spatial grid
            v_prev = v
            v = self.stack_input(v, x)
                        
            v = self.lifting(v)
            
            def f(v, dynamic_fno_block):
                fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
                return fno_block(v), None
            
            v, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
            
            v = self.activation(self.last_spectral_conv(v) + self.last_bias)
            v = self.projection(v)[0]
            
            return v, v_prev
            
        _, out_max = jax.lax.scan(step, a, length = self.max_steps)
    
        t_max = self.encode_t(jnp.linspace(0, 2, self.max_steps))
        out = interp1d(t, t_max, out_max, method="akima")

        return out
    
    def predict_whole_grid(self, a, x, t):
        """Utility function for predicting the output over the whole grid."""
        return self(a, x, t)
    
    def predict_whole_grid_batch(self, a, x, t):
        """To predict over the whole grid on a batch of initial conditions."""
        return vmap(self.predict_whole_grid, (0, None, None))(a, x, t)
    
    def Dt(self, a, x, t):
        def step(v, _):
            # combine with spatial grid
            v_prev = v
            v = self.stack_input(v, x)
                        
            v = self.lifting(v)
            
            def f(v, dynamic_fno_block):
                fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
                return fno_block(v), None
            
            v, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
            
            v = self.activation(self.last_spectral_conv(v) + self.last_bias)
            v = self.projection(v)[0]
            
            return v, v_prev
            
        _, out_max = jax.lax.scan(step, a, length = self.max_steps)
    
        t_max = self.encode_t(jnp.linspace(0, 2, self.max_steps))
        out = interp1d(t, t_max, out_max, method="akima", derivative = 1)
        return out
    
    def Dx(self, a, x, t):
        def step(v, _):
            # combine with spatial grid
            v = self.stack_input(v, x)
                        
            v = self.lifting(v)
            
            def f(v, dynamic_fno_block):
                fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
                return fno_block(v), None
            
            v_Lm1, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
            
            v_L = self.last_spectral_conv(v_Lm1) + self.last_bias
            
            d_Q_d_v_L = grad(lambda v_L : jnp.sum(self.projection(self.activation(v_L))))(v_L)
            d_K_L_dx = self.last_spectral_conv.Dx(v_Lm1, x[1]-x[0])
            
            v = self.projection(self.activation(v_L))[0]
            return v, (d_Q_d_v_L * d_K_L_dx).sum(axis=0)
            
        _, out_max = jax.lax.scan(step, a, length = self.max_steps)
    
        t_max = self.encode_t(jnp.linspace(0, 2, self.max_steps))
        out = interp1d(t, t_max, out_max, method="akima")
        
        return out
    
    def Dxx(self, a, x, t):
        def step(v, _):
            # combine with spatial grid
            v = self.stack_input(v, x)
                        
            v = self.lifting(v)
            
            def f(v, dynamic_fno_block):
                fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
                return fno_block(v), None
            
            v_Lm1, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
            
            v_L = self.last_spectral_conv(v_Lm1) + self.last_bias
            
            def sum_projection(v_L):
                return jnp.sum(self.projection(self.activation(v_L)))
            
            d_Q_dv_L = grad(lambda v_L : jnp.sum(self.projection(self.activation(v_L)))) # 1st derivative
            d2_Q_dv_L2_val = grad(lambda v_L : jnp.sum(d_Q_dv_L(v_L)))(v_L) # 2nd derivative
            d_Q_dv_L_val = d_Q_dv_L(v_L)
            
            dx = x[1]-x[0]
            d_K_L_dx = self.last_spectral_conv.Dx(v_Lm1, dx)
            d2_K_L_dx2 = self.last_spectral_conv.Dxx(v_Lm1, dx)
            
            v = self.projection(self.activation(v_L))[0]
            return v, (d2_Q_dv_L2_val * (d_K_L_dx)**2 + d2_K_L_dx2*d_Q_dv_L_val).sum(axis=0)
            
        _, out_max = jax.lax.scan(step, a, length = self.max_steps)
    
        t_max = self.encode_t(jnp.linspace(0, 2, self.max_steps))
        out = interp1d(t, t_max, out_max, method="akima")
        
        return out
    
    def Dxxx(self, a, x, t):
        def step(v, _):
            # combine with spatial grid
            v = self.stack_input(v, x)
                        
            v = self.lifting(v)
            
            def f(v, dynamic_fno_block):
                fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
                return fno_block(v), None
            
            v_Lm1, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
            
            v_L = self.last_spectral_conv(v_Lm1) + self.last_bias
            
            d_Q_dv_L = grad(lambda v_L : jnp.sum(self.projection(self.activation(v_L)))) # 1st derivative
            d2_Q_dv_L2 = grad(lambda v_L : jnp.sum(d_Q_dv_L(v_L))) # 2nd derivative
            d3_Q_dv_L3_val = grad(lambda v_L : jnp.sum(d2_Q_dv_L2(v_L)))(v_L) # 3rd derivative
            d_Q_dv_L_val = d_Q_dv_L(v_L)
            d2_Q_dv_L2_val = d2_Q_dv_L2(v_L)
            
            dx = x[1]-x[0]
            d_K_L_dx = self.last_spectral_conv.Dx(v_Lm1, dx)
            d2_K_L_dx2 = self.last_spectral_conv.Dxx(v_Lm1, dx)
            d3_K_L_dx3 = self.last_spectral_conv.Dxxx(v_Lm1, dx)
            
            v = self.projection(self.activation(v_L))[0]
            return v, (d3_Q_dv_L3_val*d_K_L_dx**3 + 3*d2_Q_dv_L2_val*d_K_L_dx*d2_K_L_dx2+ d3_K_L_dx3*d_Q_dv_L_val).sum(axis=0)
            
        _, out_max = jax.lax.scan(step, a, length = self.max_steps)
    
        t_max = self.encode_t(jnp.linspace(0, 2, self.max_steps))
        out = interp1d(t, t_max, out_max, method="akima")
        
        return out
    
    def Dx_whole_grid(self, a, x, t):
        return self.Dx(a, x, t)
    
    def Dt_whole_grid(self, a, x, t):
        return self.Dt(a, x, t)
    
    def u_t(self, a, x, t):
        return self.Dt(a, x, t)*self.u_std/self.t_std
    
    def u_t_whole_grid(self, a, x, t):
        return self.Dt_whole_grid(a, x, t)*self.u_std/self.t_std
    
    def u_x(self, a, x, t):
        return self.Dx(a, x, t)*self.u_std/self.x_std
    
    def u_xx(self, a, x, t):
        return self.Dxx(a, x, t)*self.u_std/self.x_std**2
    
    def u_xxx(self, a, x, t):
        return self.Dxxx(a, x, t)*self.u_std/self.x_std**3
    
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
        #max_steps = trial.suggest_int("max_steps", 5, 50)
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
        is_self_adaptive=True
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
            #max_steps=max_steps,
            #learning_rate=learning_rate,
            λ_learning_rate = λ_learning_rate,
            λ_shape = self.λ_shape,
            λ_smooth_or_sharp = λ_smooth_or_sharp,
            λ_learnable = λ_learnable,
            λ_mask = λ_mask,
            **self.hparams,
            **self.z_score_data
        )
        
        model = FNOTimeStepping(hparams)
        if replicated:=self.trainer_kwargs.get("replicated"):
            model = eqx.filter_shard(model, replicated)
        opt_state = opt.init(eqx.filter([model], eqx.is_array))

        trainer = Trainer(model, opt, opt_state, self.train_loader, self.val_loader, trial=trial, **self.trainer_kwargs)
        trainer()
        best_val_loss = trainer.best_val_loss
        
        del trainer, opt_state, opt
        return best_val_loss
    
    
def compute_loss(model, a, u, key):
    """Computes the relative L2 loss over the whole training grid.

    Args:
        model (eqx.Model)
        a (batch, M+1)
        u (batch, N+1, M+1)
        key : jax PRNGKey

    Returns:
        loss
    """
    batch_size = u.shape[0]
    u_pred = model.predict_whole_grid_batch(a, Trainer.x, Trainer.t)
    
    #compute the loss 
    u_norms = jnp.linalg.norm(u.reshape(batch_size,-1), 2, 1)
    if model.is_self_adaptive:
        λ = model.self_adaptive.all_with_mask()[None,:,None] #(1,64,1)
        diff_norms = jnp.sqrt(jnp.sum((λ * jnp.square(u - u_pred)).reshape(batch_size,-1), axis=1))
    else:
        diff_norms = jnp.linalg.norm((u - u_pred).reshape(batch_size,-1), 2, 1)
        
    loss = jnp.mean(diff_norms/u_norms)
    
    return loss