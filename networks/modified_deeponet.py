from dataclasses import dataclass
import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap, grad
import equinox as eqx
import optax
from typing import Callable, Optional, Union
from jaxtyping import Array
from ._abstract_operator_net import AbstractOperatorNet, AbstractHparams
import jax_dataloader as jdl
import sys
sys.path.append("..")
from utils.trainer import Trainer
from utils.model_utils import param_labels, init_he
from optax.contrib import reduce_on_plateau

@dataclass(kw_only=True, frozen=True)
class Hparams(AbstractHparams):
    """Stores the hyperparameters for the ModifiedDeepONet network"""
    number_of_sensors: int
    width: int
    depth: int
    interact_size: int
    num_query_points: int

class ModifiedDeepONet(AbstractOperatorNet):
    encoder_branch: eqx.nn.Linear
    encoder_trunk: eqx.nn.Linear
    
    first_branch_layer: eqx.nn.Linear
    first_trunk_layer: eqx.nn.Linear
    intermediate_branch_layers: eqx.nn.Linear
    intermediate_trunk_layers: eqx.nn.Linear
    last_branch_layer: eqx.nn.Linear
    last_trunk_layer: eqx.nn.Linear
    
    multiplier: float
    activation: Callable
    
    number_of_sensors: int
    num_query_points: int
    
    bias: Array

    def __init__(self, hparams : Union[Hparams, dict]):
        if isinstance(hparams, dict):
            hparams = Hparams(**hparams)
        super().__init__(hparams)
        
        key = random.key(hparams.seed)
        self.number_of_sensors = hparams.number_of_sensors
               
        generate_branch_keys, generate_trunk_keys = random.split(key, 2)
        branch_keys = random.split(generate_branch_keys, hparams.depth+1) #+1 to account for the encoder
        trunk_keys = random.split(generate_trunk_keys, hparams.depth+1)
        
        self.encoder_branch = init_he(eqx.nn.Linear(self.number_of_sensors, hparams.width, key=branch_keys[0]),branch_keys[0])
        self.encoder_trunk = init_he(eqx.nn.Linear(3, hparams.width, key=trunk_keys[0]), trunk_keys[0])
        
        self.first_branch_layer = init_he(eqx.nn.Linear(self.number_of_sensors, hparams.width, key=branch_keys[1]), branch_keys[1])
        self.first_trunk_layer = init_he(eqx.nn.Linear(3, hparams.width, key=trunk_keys[1]), trunk_keys[1])
        
        self.intermediate_branch_layers = eqx.filter_vmap(lambda key: init_he(eqx.nn.Linear(hparams.width, hparams.width, key=key), key))(branch_keys[2:-1])
        self.intermediate_trunk_layers = eqx.filter_vmap(lambda key: init_he(eqx.nn.Linear(hparams.width, hparams.width, key=key), key))(trunk_keys[2:-1])
        
        self.last_branch_layer = init_he(eqx.nn.Linear(hparams.width, hparams.interact_size, key=branch_keys[-1]), branch_keys[-1])
        self.last_trunk_layer = init_he(eqx.nn.Linear(hparams.width, hparams.interact_size, key=trunk_keys[-1]), trunk_keys[-1])
        
        self.activation = jax.nn.gelu
        self.multiplier = float(2*jnp.pi/self.period)

        self.num_query_points = hparams.num_query_points
        
        self.bias = jnp.zeros((1,))

    def __call__(self, a, x, t):
        # Encode in Fourier basis for periodicity
        y = jnp.array([jnp.cos(self.multiplier*x), jnp.sin(self.multiplier*x), t])
        a = self.encoder(a)
        
        # Compute the encoder outputs
        B_E = self.activation(self.encoder_branch(a))
        T_E = self.activation(self.encoder_trunk(y))

        # Compute the first layer outputs
        B = self.activation(self.first_branch_layer(a))
        T = self.activation(self.first_trunk_layer(y))

        a = jnp.multiply(B, B_E) + jnp.multiply(1 - B, T_E) 
        y = jnp.multiply(T, B_E) + jnp.multiply(1 - T, T_E)
        
        # Compute the intermediate layer outputs
        def f(values, layers):
            a, y = values
            branch_layer, trunk_layer = layers
            
            B = self.activation(branch_layer(a))
            T = self.activation(trunk_layer(y))
            
            a = jnp.multiply(B, B_E) + jnp.multiply(1 - B, T_E) 
            y = jnp.multiply(T, B_E) + jnp.multiply(1 - T, T_E)
            return (a, y), None
        
        (a, y), _ = jax.lax.scan(f, (a, y), (self.intermediate_branch_layers, self.intermediate_trunk_layers))

        # Compute the last layer outputs
        B = self.last_branch_layer(a) # last layer of the branch network is not activated
        T = self.activation(self.last_trunk_layer(y)) # last layer of the trunk network is activated, following the original implementation
        
        outputs = jnp.dot(B, T) + self.bias[0] # the outputs are combined by a dot product
        return outputs

    def encoder(self, a):
        """
        Downsamples the input function a to the size of the branch network input size.
        Assumes that the input function is equispaced, and that the grid points are a multiple of "number_of_sensors".
        """
        return a[::len(a)//self.number_of_sensors]
    
    def multiple_query_points_one_a(self, a, x, t):
        return vmap(self, (None, 0, 0))(a, x, t)
    
    def predict_whole_grid(self, a, x, t):
        y = jnp.stack(jnp.meshgrid(x, t), axis=-1).reshape(-1, 2)
        return jax.lax.map(lambda y: self(a, y[0], y[1]), y, batch_size=1000).reshape(len(t), len(x))
    
    def predict_whole_grid_batch(self, a, x, t):
        return vmap(self.predict_whole_grid, (0, None, None))(a, x, t)
    
    def Dx(self, a, x, t):
        return grad(self, 1)(a, x, t)
    
    def Dt(self, a, x, t):
        return grad(self, 2)(a, x, t)
    
    def Dx_whole_grid(self, a, x, t):
        xx, tt = jnp.meshgrid(x, t)
        return vmap(self.Dx, (None, 0, 0))(a, xx.ravel(), tt.ravel()).reshape(xx.shape)
    
    def Dt_whole_grid(self, a, x, t):
        xx, tt = jnp.meshgrid(x, t)
        return vmap(self.Dt, (None, 0, 0))(a, xx.ravel(), tt.ravel()).reshape(xx.shape)
    
    def u_t(self, a, x, t):
        return self.Dt(a, x, t)*self.u_std/self.t_std
    
    def u_t_whole_grid(self, a, x, t):
        return self.Dt_whole_grid(a, x, t)*self.u_std/self.t_std
    
    def u_x(self, a, x, t):
        return self.Dx(a, x, t)*self.u_std/self.x_std
    
    def u_x_whole_grid(self, a, x, t):
        return self.Dx_whole_grid(a, x, t)*self.u_std/self.x_std
    
def compute_loss(model, a, u, key):
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
    t_idx = random.randint(t_key, (batch_size, model.num_query_points), 0, Np1)
    x_idx = random.randint(x_key, (batch_size, model.num_query_points), 0, Mp1)

    # Select the ground truth data at the query points
    # Has shape (batch_size, num_query_points)
    u_at_query_points = u[jnp.arange(batch_size)[:, None], t_idx, x_idx]
    
    # For each input function, compute the prediction of the model at the query points. (inner vmap)
    # Do this for each sample in the batch. (outer vmap)
    # Has shape (batch_size, num_query_points)   
    u_pred = vmap(vmap(model, (None, 0, 0)))(a, Trainer.x[x_idx], Trainer.t[t_idx])
    
    #compute the loss 
    u_norms = jnp.linalg.norm(u_at_query_points, 2, 1)
    if model.is_self_adaptive:
        λ = model.self_adaptive(t_idx) 
        diff_norms = jnp.sqrt(jnp.sum(λ * jnp.square(u_at_query_points - u_pred), axis=1))
    else:
        diff_norms = jnp.linalg.norm(u_at_query_points - u_pred, 2, 1)
    
    loss = jnp.mean(diff_norms/u_norms)
    
    a_pred = vmap(vmap(model, (None, 0, None)), (0, None, None))(a, Trainer.x, Trainer.t[0])
    ic_loss = jnp.mean(jnp.linalg.norm(a - a_pred, 2, 1)/jnp.linalg.norm(a, 2, 1))
    return loss + ic_loss*0.5
    
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
        #width=trial.suggest_int("width", 50, 150)
        #depth=trial.suggest_int("depth", 3, 10)
        #interact_size=trial.suggest_int("interact_size", 5, 25)
        #learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1)
        #number_of_sensors=trial.suggest_categorical("number_of_sensors", [16, 32, 64])
        #num_query_points=trial.suggest_int("num_query_points", 50, 2000)
        
        # Optimizer used for the "regular" network parameters
        PATIENCE = 5 # Number of epochs with no improvement after which learning rate will be reduced
        COOLDOWN = 0 # Number of epochs to wait before resuming normal operation after the learning rate reduction
        FACTOR = 0.5  # Factor by which to reduce the learning rate:
        RTOL = 1e-4  # Relative tolerance for measuring the new optimum:
        ACCUMULATION_SIZE = 200 # Number of iterations to accumulate an average value:
        θ_optimizer = optax.chain(
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
            #width=width,
            #depth=depth,
            #interact_size=interact_size,
            #learning_rate=learning_rate,
            #number_of_sensors=number_of_sensors,
            #num_query_points=num_query_points,
            λ_learning_rate = λ_learning_rate,
            λ_shape = self.λ_shape,
            λ_smooth_or_sharp = λ_smooth_or_sharp,
            λ_learnable = λ_learnable,
            λ_mask = λ_mask,
            **self.z_score_data,
            **self.hparams
        )
        
        model = ModifiedDeepONet(hparams)
        if replicated:=self.trainer_kwargs.get("replicated"):
            model = eqx.filter_shard(model, replicated)
        opt_state = opt.init(eqx.filter([model], eqx.is_array))

        trainer = Trainer(model, opt, opt_state, self.train_loader, self.val_loader, trial=trial, **self.trainer_kwargs)
        trainer()
        best_val_loss = trainer.best_val_loss
        
        del trainer, opt_state, opt
        return best_val_loss