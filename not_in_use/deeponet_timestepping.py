from dataclasses import dataclass
import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap
import equinox as eqx
from typing import Callable, Union
from ._abstract_operator_net import AbstractOperatorNet, AbstractHparams
import sys
sys.path.append("..")
from utils.trainer import Trainer
from utils.model_utils import param_labels
import optax

@dataclass(kw_only=True, frozen=True)
class Hparams(AbstractHparams):
    trunk_width: int
    branch_width: int
    trunk_depth: int
    branch_depth: int
    interact_size: int
    period: float
    number_of_sensors: int
    
class DeepONetTimeStepper(AbstractOperatorNet):
    branch_net: eqx.nn.MLP
    trunk_net: eqx.nn.MLP
    multiplier: float
    
    def __init__(self, hparams : Union[Hparams, dict]):
        if isinstance(hparams, dict):
            hparams = Hparams(**hparams)
        super().__init__(hparams)
        
        key = random.key(hparams.seed)
        
        number_of_sensors = hparams.number_of_sensors
        
        key, b_key, t_key = random.split(key, 3)
        activation = jax.nn.gelu
        
        self.branch_net = eqx.nn.MLP(
            in_size = number_of_sensors,
            out_size = hparams.interact_size,
            width_size = hparams.branch_width,
            depth = hparams.branch_depth,
            activation = activation,
            key = b_key,
        )
        
        self.trunk_net = eqx.nn.MLP(
            in_size = 2, # input is [cos(2πx/P), sin(2πx/P)]
            out_size = hparams.interact_size,
            width_size = hparams.trunk_width,
            depth = hparams.trunk_depth,
            activation = activation,
            final_activation = activation,
            key = t_key,
        )
        
        self.multiplier = 2*jnp.pi/hparams.period
        
        self.step_size = 0.1
        
    def __call__(self, a, x, t):
        """

        Args:
            a (M+1,): input function
            x scalar: spatial query point
            t scalar: temporal query point

        Returns:
            pred scalar: prediction for u(x,t) at x=x and t=t.
        """
        
        num_steps = t // self.step_size
        # the trunk output is the same for all t, so we compute it once
        trunk_out = self.trunk_net(jnp.array([jnp.cos(self.multiplier*x), jnp.sin(self.multiplier*x)]))
        
        def step(v):
            branch_out = self.branch_net(v)
            v_next = jnp.dot(branch_out,trunk_out)
            return v_next, None
        
        v_prev, _ = jax.lax.scan(step, a, length = num_steps)
        v_next, _ = step(v_prev)
        
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
        return vmap(vmap(self, (None, 0, None)), (None, None, 0))(a, x, t)
    
    def get_self_adaptive(self):
        return self.self_adaptive
    
   
class HparamTuning:
    def __init__(self, train_loader, val_loader, number_of_sensors, period, N, **kwargs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.number_of_sensors = number_of_sensors
        self.N = N
        self.period = period
        self.kwargs = kwargs

    def __call__(self, trial):
        
        is_self_adaptive = trial.suggest_categorical("self_adaptive", [True, False])
        if is_self_adaptive:
            hparams = Hparams(
                trunk_width=trial.suggest_int("trunk_width", 50, 150),
                branch_width=trial.suggest_int("branch_width", 50, 150),
                trunk_depth=trial.suggest_int("trunk_depth", 3, 10),
                branch_depth=trial.suggest_int("branch_depth", 3, 10),
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
                trunk_width=trial.suggest_int("width", 50, 150),
                branch_width=trial.suggest_int("width", 50, 150),
                trunk_depth=trial.suggest_int("depth", 3, 10),
                branch_depth=trial.suggest_int("depth", 3, 10),
                interact_size=trial.suggest_int("interact_size", 5, 25),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2),
                number_of_sensors=self.number_of_sensors,
                period=self.period,
            )

            opt = optax.adam(hparams.learning_rate)    
        
        model = DeepONet(hparams)
        model = eqx.filter_shard(model, Trainer.replicated)
        opt_state = opt.init(eqx.filter([model], eqx.is_array))

        trainer = Trainer(model, opt, opt_state, self.train_loader, self.val_loader, trial=trial, **self.kwargs)
        trainer()
        last_val_loss = trainer.current_val_loss
        
        """
        # compute final loss over whole validation set, over the whole grid
        for (a, u) in self.val_loader:
            u_pred = vmap(model.predict_whole_grid, (0, None, None))(a, Trainer.x, Trainer.t)
            last_val_loss += jnp.mean(jnp.linalg.norm(u - u_pred, 2, 1)/jnp.linalg.norm(u, 2, 1))
        """
        
        return last_val_loss
    
    
def compute_loss(model, a, u, key, num_query_points=100):
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
    u_norms = jnp.linalg.norm(u_at_query_points, 2, 1)
    if model.is_self_adaptive:
        λ = model.self_adaptive(t_idx) 
        diff_norms = jnp.sqrt(jnp.sum(λ * jnp.square(u_at_query_points - u_pred), axis=1))
    else:
        diff_norms = jnp.linalg.norm((u_at_query_points - u_pred), 2, 1)
        
    loss = jnp.mean(diff_norms/u_norms)
    return loss