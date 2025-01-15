from dataclasses import dataclass
import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap
import equinox as eqx
import optax
from typing import Callable, Optional, Union
from jaxtyping import Array
from ._abstract_operator_net import AbstractOperatorNet, AbstractHparams
import jax_dataloader as jdl
import sys
sys.path.append("..")
from utils.trainer import Trainer
from utils.model_utils import param_labels

@dataclass(kw_only=True, frozen=True)
class Hparams(AbstractHparams):
    """Stores the hyperparameters for the ModifiedDeepONet network"""
    number_of_sensors: int
    width: int
    depth: int
    interact_size: int
    period: float

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

    def __init__(self, hparams : Union[Hparams, dict]):
        if isinstance(hparams, dict):
            hparams = Hparams(**hparams)
        super().__init__(hparams)
        
        key = random.key(hparams.seed)
        number_of_sensors = hparams.number_of_sensors
               
        generate_branch_keys, generate_trunk_keys = random.split(key, 2)
        branch_keys = random.split(generate_branch_keys, hparams.depth+1) #+1 to account for the encoder
        trunk_keys = random.split(generate_trunk_keys, hparams.depth+1)
        
        self.encoder_branch = eqx.nn.Linear(number_of_sensors, hparams.width, key=branch_keys[0])
        self.encoder_trunk = eqx.nn.Linear(2, hparams.width, key=trunk_keys[0])
        
        self.first_branch_layer = eqx.nn.Linear(number_of_sensors, hparams.width, key=branch_keys[1])
        self.first_trunk_layer = eqx.nn.Linear(2, hparams.width, key=trunk_keys[1])
        
        self.intermediate_branch_layers = eqx.filter_vmap(lambda key: eqx.nn.Linear(hparams.width, hparams.width, key=key))(branch_keys[2:-1])
        self.intermediate_trunk_layers = eqx.filter_vmap(lambda key: eqx.nn.Linear(hparams.width, hparams.width, key=key))(trunk_keys[2:-1])
        
        self.last_branch_layer = eqx.nn.Linear(hparams.width, hparams.interact_size, key=branch_keys[-1])
        self.last_trunk_layer = eqx.nn.Linear(hparams.width, hparams.interact_size, key=trunk_keys[-1])
        
        self.activation = jax.nn.gelu
        self.multiplier = float(2*jnp.pi/hparams.period)
        
        self.step_size = 0.1

    def __call__(self, a, x, t):
        num_steps = t // self.step_size
        
        # Encode in Fourier basis for periodicity
        y = jnp.array([jnp.cos(self.multiplier*x), jnp.sin(self.multiplier*x)])
        
        # Compute the trunk encoder and first layer outputs, as they are used in every step
        T_E = self.activation(self.encoder_trunk(y))
        T = self.activation(self.first_trunk_layer(y))
        def step(v):
            # Compute the encoder outputs
            B_E = self.activation(self.encoder_branch(v))

            # Compute the first layer outputs
            B = self.activation(self.first_branch_layer(v))

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
            T = self.activation(self.last_trunk_layer(y)) # last layer of the trunk network is activated, following the original theorem
            
            v_next = jnp.dot(B, T) # the outputs are combined by a dot product
            return v_next, None
        
        v_prev, _ = jax.lax.scan(step, a, length = num_steps)
        
        # do one additional step
        v_next, _ = step(v)
        
        # linearly interpolate between v and v_next
        alpha = (t % self.step_size) / self.step_size
        v = (1 - alpha) * v_prev + alpha * v_next
        return v
    
    def predict_whole_grid(self, a, x, t):
        """As the DeepONet gives the prediction at scalar x and t,
        this is a utility function for predicting over the whole grid, with x and t being vectors.

        Args:
            a (M+1,): input function
            x (M+1,): spatial grid
            t (N+1,): temporal grid
        Returns:
            pred (N+1, M+1): the prediction for u(x,t) for all grid points given by x and t
        """
        pred = vmap(vmap(self, (None, 0, None)), (None, None, 0))(a, x, t)
        return pred
    
    def get_self_adaptive(self):
        return self.self_adaptive
    
def compute_loss(model, a, u, key, num_query_points=100):
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
    u_at_query_points = u[jnp.arange(batch_size)[:, None], t_idx, x_idx]
    
    # For each input function, compute the prediction of the model at the query points. (inner vmap)
    # Do this for each sample in the batch. (outer vmap)
    # Has shape (batch_size, num_query_points)   
    u_pred = vmap(vmap(model, (None, 0, 0)))(a, Trainer.x[x_idx], Trainer.t[t_idx])
    
    #compute the loss 
    u_norms = jnp.linalg.norm(u_at_query_points.reshape(batch_size,-1), 2, 1)
    if model.is_self_adaptive:
        λ = model.self_adaptive(t_idx) 
        diff_norms = jnp.sqrt(jnp.sum(λ * jnp.square(u_at_query_points - u_pred), axis=1))
    else:
        diff_norms = jnp.linalg.norm((u_at_query_points - u_pred).reshape(batch_size,-1), 2, 1)
    
    loss = jnp.mean(diff_norms/u_norms)
    return loss

    
class HparamTuning:
    def __init__(self, train_loader, val_loader, number_of_sensors, period, N, **kwargs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.number_of_sensors = number_of_sensors
        self.period = period
        self.N = N
        self.kwargs = kwargs

    def __call__(self, trial):
    
        is_self_adaptive = trial.suggest_categorical("self_adaptive", [True, False])
        if is_self_adaptive:
            hparams = Hparams(
                width=trial.suggest_int("width", 50, 150),
                depth=trial.suggest_int("depth", 3, 10),
                interact_size=trial.suggest_int("interact_size", 5, 25),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1),
                number_of_sensors=self.number_of_sensors,
                period=self.period,
                λ_learning_rate=trial.suggest_float("λ_learning_rate", 1e-4, 1e-1),
                λ_shape=self.N+1,
                λ_mask=trial.suggest_categorical("λ_mask", ["soft_relu", "quadratic", "sigmoid"]),
            )
            
            θ_optimizer = optax.adam(hparams.learning_rate)
            λ_optimizer = optax.chain(optax.adam(hparams.λ_learning_rate), optax.scale(-1.))
            opt = optax.multi_transform({'θ': θ_optimizer, 'λ': λ_optimizer}, param_labels=param_labels)
        else:
            hparams = Hparams(
                width=trial.suggest_int("width", 50, 150),
                depth=trial.suggest_int("depth", 3, 10),
                interact_size=trial.suggest_int("interact_size", 5, 25),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2),
                number_of_sensors=self.number_of_sensors,
                period=self.period,
            )

            opt = optax.adam(hparams.learning_rate)    
        
        model = ModifiedDeepONet(hparams)
        model = eqx.filter_shard(model, Trainer.replicated)
        opt_state = opt.init(eqx.filter([model], eqx.is_array))

        trainer = Trainer(model, opt, opt_state, self.train_loader, self.val_loader, trial=trial, **self.kwargs)
        trainer()
        last_val_loss = trainer.current_val_loss
        
        return last_val_loss