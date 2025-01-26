from dataclasses import dataclass
import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap, grad, value_and_grad
import equinox as eqx
from typing import Callable, Union
from ._abstract_operator_net import AbstractOperatorNet, AbstractHparams
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
    
class Bypass(eqx.Module):
    weights: Array

    def __init__(self, in_channels, out_channels, key):
        init_std = jnp.sqrt(2.0 / in_channels)
        self.weights = random.normal(key, (out_channels, in_channels))*init_std

    def __call__(self, v):
        return jnp.tensordot(self.weights, v, axes=1)

class SpectralConv2d(eqx.Module):
    weights1: Array
    weights2: Array
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int

    def __init__(self, in_channels, out_channels, modes1, modes2, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        subkey1, subkey2 = random.split(key, 2)
     
        init_std = jnp.sqrt(2.0 / (self.in_channels))
        
        self.weights1 = random.normal(subkey1, (self.in_channels, self.out_channels, self.modes1, self.modes2), dtype = jnp.complex64)*init_std
        self.weights2 = random.normal(subkey2, (self.in_channels, self.out_channels, self.modes1, self.modes2), dtype = jnp.complex64)*init_std

    def __call__(self, v):
        in_channels, points_x, points_y = v.shape

        # Fourier transform
        v_hat = jnp.fft.rfft2(v)
        
        # Multiply relevant Fourier modes
        out_hat_padded = jnp.zeros((self.out_channels, points_x, points_y//2 + 1), dtype = v_hat.dtype)
        out_hat_padded = out_hat_padded.at[:, :self.modes1 , :self.modes2].set(self.compl_mul2d(v_hat[:, :self.modes1 , :self.modes2], self.weights1))
        out_hat_padded = out_hat_padded.at[:, -self.modes1:, :self.modes2].set(self.compl_mul2d(v_hat[:, -self.modes1: , :self.modes2], self.weights2))

        # Return to physical space
        out = jnp.fft.irfft2(out_hat_padded, s=[points_x, points_y])

        return out
    
    def compl_mul2d(self, input, weights):
        # (in_channel, x, y), (in_channel, out_channel, x,y) -> (out_channel, x,y)
        return jnp.einsum("ixy,ioxy->oxy", input, weights)
    
    def spatial_derivatives(self, v, dx):
        channels, Nt, Nx = v.shape

        # Compute 2D Fourier transform
        v_hat = jnp.fft.rfft2(v)
        
        out_hat_padded = jnp.zeros((self.out_channels, Nt, Nx//2 + 1), dtype = v_hat.dtype)
        out_hat_padded = out_hat_padded.at[:, :self.modes1 , :self.modes2].set(self.compl_mul2d(v_hat[:, :self.modes1 , :self.modes2], self.weights1))
        out_hat_padded = out_hat_padded.at[:, -self.modes1:, :self.modes2].set(self.compl_mul2d(v_hat[:, -self.modes1: , :self.modes2], self.weights2))

        # Apply differentiation in Fourier space
        kx = jnp.fft.rfftfreq(Nx, dx) * 2 * jnp.pi
        dx_hat = 1j * kx[None, :] * out_hat_padded
        dxx_hat = (1j * kx[None, :])**2 * out_hat_padded
        dxxx_hat = (1j * kx[None, :])**3 * out_hat_padded

        # Transform back to physical space
        dx_val = jnp.fft.irfft2(dx_hat, s=(Nt, Nx))
        dxx_val = jnp.fft.irfft2(dxx_hat, s=(Nt, Nx))
        dxxx_val = jnp.fft.irfft2(dxxx_hat, s=(Nt, Nx))
        return dx_val, dxx_val, dxxx_val
    
    def Dx(self, v, dx):
        channels, Nt, Nx = v.shape

        # Compute 2D Fourier transform
        v_hat = jnp.fft.rfft2(v)
        
        out_hat_padded = jnp.zeros((self.out_channels, Nt, Nx//2 + 1), dtype = v_hat.dtype)
        out_hat_padded = out_hat_padded.at[:, :self.modes1 , :self.modes2].set(self.compl_mul2d(v_hat[:, :self.modes1 , :self.modes2], self.weights1))
        out_hat_padded = out_hat_padded.at[:, -self.modes1:, :self.modes2].set(self.compl_mul2d(v_hat[:, -self.modes1: , :self.modes2], self.weights2))

        # Apply differentiation in Fourier space
        kx = jnp.fft.rfftfreq(Nx, dx) * 2 * jnp.pi
        dx_hat = 1j * kx[None, :] * out_hat_padded

        # Transform back to physical space
        dx_val = jnp.fft.irfft2(dx_hat, s=(Nt, Nx))
        return dx_val
        
    def Dt(self, v, dt):
        channels, Nt, Nx = v.shape

        # Compute 2D Fourier transform
        v_hat = jnp.fft.rfft2(v)
        
        out_hat_padded = jnp.zeros((self.out_channels, Nt, Nx//2 + 1), dtype = v_hat.dtype)
        out_hat_padded = out_hat_padded.at[:, :self.modes1 , :self.modes2].set(self.compl_mul2d(v_hat[:, :self.modes1 , :self.modes2], self.weights1))
        out_hat_padded = out_hat_padded.at[:, -self.modes1:, :self.modes2].set(self.compl_mul2d(v_hat[:, -self.modes1: , :self.modes2], self.weights2))


        # Apply differentiation in Fourier space# Apply differentiation in Fourier space
        kt = jnp.fft.fftfreq(Nt, dt) * jnp.pi * 2 
        dt_hat = 1j * kt[:, None] * out_hat_padded

        # Transform back to physical space
        dt_val = jnp.fft.irfft2(dt_hat, s=(Nt, Nx))
        return dt_val
    
class FNOBlock2d(eqx.Module):
    spectral_conv: SpectralConv2d
    bypass: Bypass
    bias: Array
    activation: Callable
    
    def __init__(self, in_channels, out_channels, modes1, modes2, activation, key = random.key(0)):
        spectral_conv_key, bypass_key, bias_key = jax.random.split(key, 3)
        self.spectral_conv = SpectralConv2d(
            in_channels,
            out_channels,
            modes1,
            modes2,
            spectral_conv_key,
        )
        self.bypass = Bypass(in_channels, out_channels, bypass_key)
        self.activation = activation
        self.bias = jnp.zeros((out_channels, 1, 1))

    def __call__(self, v):
        return self.activation(self.spectral_conv(v) + self.bypass(v) + self.bias)

class FNO2d(AbstractOperatorNet):
    """Maps a function on a spatial-temporal domain to the solution on the same spatial-temporal domain.
    For the case where the input is an initial condition, the field a(x,t) is constant in time (=a(x)).
    
    Input: (a(x,t), x, t)
        Shapes:
            a(x,t): (m,)
            x: (m,)
            t: (n,)
    Outputs: u(x,t)
        Shapes:
            u(x,t): (n, m)
    """
    lifting: eqx.nn.Conv2d
    last_spectral_conv: SpectralConv2d
    activation: Callable
    projection: eqx.nn.Conv2d
    last_bias_coeffs: Array
    modes_max: int
    multiplier: float
    dynamic_fno_blocks: FNOBlock2d
    static_fno_blocks: FNOBlock2d

    def __init__(self, hparams : Union[Hparams, dict]):
        if isinstance(hparams, dict):
            hparams = Hparams(**hparams)
        super().__init__(hparams)
        
        keys = random.split(random.key(hparams.seed), hparams.n_blocks + 3)
        
        self.lifting = eqx.nn.Conv2d(in_channels = 4, out_channels = hparams.hidden_dim, kernel_size = 1, key=keys[0])

        fno_blocks = eqx.filter_vmap(lambda key: FNOBlock2d(hparams.hidden_dim, hparams.hidden_dim, hparams.modes_max, hparams.modes_max, jax.nn.gelu, key=key))(keys[1:-3])
        self.dynamic_fno_blocks, self.static_fno_blocks = eqx.partition(fno_blocks, eqx.is_array)
        
        self.last_spectral_conv = SpectralConv2d(
            hparams.hidden_dim,
            hparams.hidden_dim,
            hparams.modes_max,
            hparams.modes_max,
            keys[-3],
        )
        
        self.activation = jax.nn.gelu
        
        self.projection = eqx.nn.Conv2d(in_channels = hparams.hidden_dim, out_channels = 1, kernel_size = 1, key=keys[-2])
        
        self.modes_max = hparams.modes_max
        
        self.last_bias_coeffs = random.normal(keys[-1], (4, self.modes_max, self.modes_max))
        
        self.multiplier = 2*jnp.pi/self.period

    def __call__(self,a,x,t):
        v = self.stack_input(a, x, t)
        
        v = self.lifting(v)
        
        def f(v, dynamic_fno_block):
            fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
            return fno_block(v), None
        
        v, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
        
        v = self.activation(self.last_spectral_conv(v) + self.last_bias(x, t))
        v = self.projection(v)

        return v[0]

    
    def predict_whole_grid(self, a, x, t):
        """FNO2d predicts over the whole grid already."""
        return self(a,x,t)
    
    def predict_whole_grid_batch(self, a, x, t):
        """To predict over the whole grid on a batch of initial conditions."""
        return vmap(self.predict_whole_grid, (0, None, None))(a, x, t)
    
    def last_bias(self, x, t):
        x_cos = jnp.cos(self.multiplier * jnp.arange(self.modes_max)[:, None] * x)
        x_sin = jnp.sin(self.multiplier * jnp.arange(self.modes_max)[:, None] * x)
        t_cos = jnp.cos(2 * jnp.pi * jnp.arange(self.modes_max)[:, None] * t / 5)
        t_sin = jnp.sin(2 * jnp.pi * jnp.arange(self.modes_max)[:, None] * t / 5)
        bias = jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[0], t_cos, x_cos) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[1], t_cos, x_sin) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[2], t_sin, x_cos) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[3], t_sin, x_sin)
        return bias[None,...]
    
    def last_bias_dx(self, x, t):
        x_cos_dx = -self.multiplier*jnp.arange(self.modes_max)[:, None] * jnp.sin(self.multiplier* jnp.arange(self.modes_max)[:, None] * x)
        x_sin_dx = self.multiplier*jnp.arange(self.modes_max)[:, None] *jnp.cos(self.multiplier* jnp.arange(self.modes_max)[:, None] * x)
        t_cos = jnp.cos(2* jnp.pi * jnp.arange(self.modes_max)[:, None] * t / 5)
        t_sin = jnp.sin(2*jnp.pi * jnp.arange(self.modes_max)[:, None] * t / 5)
        bias_dx = jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[0], t_cos, x_cos_dx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[1], t_cos, x_sin_dx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[2], t_sin, x_cos_dx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[3], t_sin, x_sin_dx)

        return bias_dx[None,...]
    
    def last_bias_spatial_derivatives(self, x, t):
        x_cos_dx = -self.multiplier*jnp.arange(self.modes_max)[:, None] * jnp.sin(self.multiplier* jnp.arange(self.modes_max)[:, None] * x)
        x_sin_dx = self.multiplier*jnp.arange(self.modes_max)[:, None] *jnp.cos(self.multiplier* jnp.arange(self.modes_max)[:, None] * x)
        x_cos_dxx = -(self.multiplier*jnp.arange(self.modes_max)[:, None])**2 * jnp.cos(self.multiplier* jnp.arange(self.modes_max)[:, None] * x)
        x_sin_dxx = -(self.multiplier*jnp.arange(self.modes_max)[:, None])**2 *jnp.sin(self.multiplier* jnp.arange(self.modes_max)[:, None] * x)
        x_cos_dxxx = (self.multiplier*jnp.arange(self.modes_max)[:, None])**3 * jnp.sin(self.multiplier* jnp.arange(self.modes_max)[:, None] * x)
        x_sin_dxxx = -(self.multiplier*jnp.arange(self.modes_max)[:, None])**3 *jnp.cos(self.multiplier* jnp.arange(self.modes_max)[:, None] * x)
        t_cos = jnp.cos(2* jnp.pi * jnp.arange(self.modes_max)[:, None] * t / 5)
        t_sin = jnp.sin(2*jnp.pi * jnp.arange(self.modes_max)[:, None] * t / 5)
        bias_dx = jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[0], t_cos, x_cos_dx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[1], t_cos, x_sin_dx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[2], t_sin, x_cos_dx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[3], t_sin, x_sin_dx)
                
        bias_dxx = jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[0], t_cos, x_cos_dxx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[1], t_cos, x_sin_dxx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[2], t_sin, x_cos_dxx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[3], t_sin, x_sin_dxx)
            
        bias_dxxx = jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[0], t_cos, x_cos_dxxx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[1], t_cos, x_sin_dxxx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[2], t_sin, x_cos_dxxx) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[3], t_sin, x_sin_dxxx)

        return bias_dx[None,...], bias_dxx[None,...], bias_dxxx[None,...]
        
    def last_bias_dt(self, x, t):
        x_cos = jnp.cos(self.multiplier * jnp.arange(self.modes_max)[:, None] * x)
        x_sin = jnp.sin(self.multiplier * jnp.arange(self.modes_max)[:, None] * x)
        t_cos = -2 * jnp.pi/5*jnp.arange(self.modes_max)[:, None] * jnp.sin(2* jnp.pi * jnp.arange(self.modes_max)[:, None] * t / 5)
        t_sin = 2 * jnp.pi/5*jnp.arange(self.modes_max)[:, None] * jnp.cos(2*jnp.pi * jnp.arange(self.modes_max)[:, None] * t / 5)
        bias = jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[0], t_cos, x_cos) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[1], t_cos, x_sin) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[2], t_sin, x_cos) \
                + jnp.einsum('nm,nj,mi->ji', self.last_bias_coeffs[3], t_sin, x_sin)
        return bias[None,...]

    
    def stack_input(self, a, x, t):
        # start by zero-padding in time, to allow for extrapolation post training. 
        # this is because the FFT assumes a signal given on a periodic domain [0, T)
        #t = jnp.zeros((self.period_t / t[1]).astype(int)).at[:len(t)].set(t)
        # each row of "a(x,t)" is "a(x)"
        a_repeated = jnp.tile(a, (len(t), 1))
        # create meshgrid
        x_repeated, t_repeated = jnp.meshgrid(x, t)
        
        # fourier basis
        x_repeated_cos = jnp.cos(self.multiplier*x_repeated)
        x_repeated_sin = jnp.sin(self.multiplier*x_repeated)
        
        # stack the inputs, shape is (4, n, m)
        return jnp.stack([a_repeated, x_repeated_cos, x_repeated_sin, t_repeated], axis=0)
    
    def Dx(self, a, x, t):
        v = self.stack_input(a, x, t)
        
        v = self.lifting(v)
        
        def f(v, dynamic_fno_block):
            fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
            return fno_block(v), None
        
        v_Lm1, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
        v_L = self.last_spectral_conv(v_Lm1) + self.last_bias(x, t)
        
        # compute derivatives
        # all have shape (hidden_channels, t_points, x_points)
        d_Q_d_v_L = jax.grad(lambda v_L : jnp.sum(self.projection(self.activation(v_L))))(v_L)
        d_K_L_dx = self.last_spectral_conv.Dx(v_Lm1, x[1]-x[0]) 
        d_b_L_dx = self.last_bias_dx(x, t)
        
        return (d_Q_d_v_L * (d_K_L_dx + d_b_L_dx)).sum(axis=0) # (t_points, x_points), sum over hidden_channels
    
    def spatial_derivatives(self, a, x, t):
        """Computes u_x, u_xx, u_xxx in their original scale."""
        v = self.stack_input(a, x, t)
        
        v = self.lifting(v)
        
        def f(v, dynamic_fno_block):
            fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
            return fno_block(v), None
        
        v_Lm1, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
        v_L = self.last_spectral_conv(v_Lm1) + self.last_bias(x, t)
        
        # compute derivatives
        # all have shape (hidden_channels, t_points, x_points)
    
        d_Q_dv_L = grad(lambda v_L : jnp.sum(self.projection(self.activation(v_L)))) # 1st derivative
        d2_Q_dv_L2 = grad(lambda v_L : jnp.sum(d_Q_dv_L(v_L))) # 2nd derivative
        d3_Q_dv_L3_val = grad(lambda v_L : jnp.sum(d2_Q_dv_L2(v_L)))(v_L) # 3rd derivative
        d_Q_dv_L_val = d_Q_dv_L(v_L)
        d2_Q_dv_L2_val = d2_Q_dv_L2(v_L)
        
        d_b_L_dx, d2_b_L_dx2, d3_b_L_dx3 = self.last_bias_spatial_derivatives(x, t)
        
        dx = x[1]-x[0]
        d_K_L_dx, d2_K_L_dx2, d3_K_L_dx3 = self.last_spectral_conv.spatial_derivatives(v_Lm1, dx)
        d_v_L_dx = d_K_L_dx + d_b_L_dx
        d2_v_L_dx2 = d2_K_L_dx2 + d2_b_L_dx2
        d3_v_L_dx3 = d3_K_L_dx3 + d3_b_L_dx3
  
        u_x = (d_Q_dv_L_val * (d_v_L_dx + d_b_L_dx)).sum(axis=0)*self.u_std/self.x_std
        u_xx = (d2_Q_dv_L2_val * d_v_L_dx**2 + d2_v_L_dx2*d_Q_dv_L_val).sum(axis=0)*self.u_std/self.x_std**2
        u_xxx = (d3_Q_dv_L3_val*d_v_L_dx**3 + 3*d2_Q_dv_L2_val*d_v_L_dx*d2_v_L_dx2+ d3_v_L_dx3*d_Q_dv_L_val).sum(axis=0)*self.u_std/self.x_std**3
  
        return u_x, u_xx, u_xxx
    
    def Dt(self, a, x, t):
        v = self.stack_input(a, x, t)
        
        v = self.lifting(v)
        
        def f(v, dynamic_fno_block):
            fno_block = eqx.combine(dynamic_fno_block, self.static_fno_blocks)
            return fno_block(v), None
        
        v_Lm1, _ = jax.lax.scan(f, v, self.dynamic_fno_blocks)
        v_L = self.last_spectral_conv(v_Lm1) + self.last_bias(x, t)
        
        # compute derivatives
        # all have shape (hidden_channels, t_points, x_points)
        d_Q_d_v_L = jax.grad(lambda v_L : jnp.sum(self.projection(self.activation(v_L))))(v_L)
        d_K_L_dt = self.last_spectral_conv.Dt(v_Lm1, t[1]-t[0]) 
        d_b_L_dt = self.last_bias_dt(x, t)
        
        return (d_Q_d_v_L * (d_K_L_dt + d_b_L_dt)).sum(axis=0) # (t_points, x_points), sum over hidden_channels
    
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
    
    def u_x_whole_grid(self, a, x, t):
        return self.Dx_whole_grid(a, x, t)*self.u_std/self.x_std
    
class HparamTuning:
    def __init__(self, train_loader, val_loader, z_score_data, hparams=None, **trainer_kwargs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.λ_shape = 64
        self.z_score_data = z_score_data
        self.trainer_kwargs = trainer_kwargs
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
        
        model = FNO2d(hparams)
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
    batch_size = u.shape[0]
    u_pred = vmap(model, (0, None, None))(a, Trainer.x, Trainer.t)
    
    #compute the loss 
    u_norms = jnp.linalg.norm(u.reshape(batch_size,-1), 2, 1)
    if model.is_self_adaptive:
        λ = model.self_adaptive.all_with_mask()[None,:,None]
        diff_norms = jnp.sqrt(jnp.sum((λ * jnp.square(u - u_pred)).reshape(batch_size,-1), axis=1))
    else:
        diff_norms = jnp.linalg.norm((u - u_pred).reshape(batch_size,-1), 2, 1)
        
    loss = jnp.mean(diff_norms/u_norms)
    
    a_pred = u_pred[:,0]
    ic_loss = jnp.mean(jnp.linalg.norm(a - a_pred, 2, 1)/jnp.linalg.norm(a, 2, 1))
    return loss + ic_loss*0.5