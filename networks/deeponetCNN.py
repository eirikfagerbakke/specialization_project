from dataclasses import dataclass
import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, vmap, grad
import equinox as eqx
from typing import Callable, Union
from ._abstract_operator_net import AbstractOperatorNet, AbstractHparams
import sys
sys.path.append("..")
from utils.trainer import Trainer
from utils.model_utils import param_labels, init_he
import optax
from jaxtyping import Array
from optax.contrib import reduce_on_plateau

@dataclass(kw_only=True, frozen=True)
class Hparams(AbstractHparams):
    trunk_width: int
    branch_width: int
    trunk_depth: int
    branch_depth: int
    interact_size: int
    number_of_sensors: int
    num_query_points: int
    
class DeepONetCNN(AbstractOperatorNet):
    branch_net: eqx.nn.MLP
    trunk_net: eqx.nn.MLP
    multiplier: float
    number_of_sensors: int
    num_query_points: int
    bias: Array
    
    def __init__(self, hparams : Union[Hparams, dict]):
        if isinstance(hparams, dict):
            hparams = Hparams(**hparams)
        super().__init__(hparams)
        
        key = random.key(hparams.seed)
        
        self.number_of_sensors = hparams.number_of_sensors
        
        key, b_key, t_key = random.split(key, 3)
        activation = jax.nn.gelu
        
        b_keys = random.split(b_key, 7)
        self.branch_net = init_he(eqx.nn.Sequential([
            eqx.nn.Conv1d(in_channels = 1,out_channels=10,kernel_size=7, key=b_keys[0]), # 64 -> 58
            eqx.nn.MaxPool1d(kernel_size=2, stride=2), # 58 -> 29
            eqx.nn.Lambda(jax.nn.gelu),
            eqx.nn.Conv1d(in_channels = 10,out_channels=50,kernel_size=5, key=b_keys[1]), # 29 -> 25
            eqx.nn.MaxPool1d(kernel_size=2, stride=2), # 25 -> 12 
            eqx.nn.Lambda(jax.nn.gelu),
            eqx.nn.Conv1d(in_channels = 50,out_channels=100,kernel_size=3, key=b_keys[2]), # 12 -> 10
            eqx.nn.MaxPool1d(kernel_size=2, stride=2), # 10 -> 5
            eqx.nn.Lambda(jax.nn.gelu),
            eqx.nn.Lambda(jnp.ravel),
            eqx.nn.Linear(500, 50, key=b_keys[3]),
            eqx.nn.Lambda(jax.nn.gelu),
            eqx.nn.Linear(50, hparams.interact_size, key=b_keys[5]),]
        ), b_keys[6])
        
        self.trunk_net = init_he(eqx.nn.MLP(
            in_size = 3, # input is [cos(2πx/P), sin(2πx/P), t]
            out_size = hparams.interact_size,
            width_size = hparams.trunk_width,
            depth = hparams.trunk_depth,
            activation = activation,
            final_activation = activation,
            key = t_key,
        ), t_key)
        
        self.multiplier = 2*jnp.pi/self.period
        self.num_query_points = hparams.num_query_points
        self.bias = jnp.zeros((1,))
        
    def eval_branch(self, a):
        return self.branch_net(self.encoder(a))
    
    def eval_trunk(self, x, t):
        return self.trunk_net(jnp.array([jnp.cos(self.multiplier*x), jnp.sin(self.multiplier*x), t]))
        
    def __call__(self, a, x, t):
        """
        Args:
            a (M+1,): input function
            x scalar: spatial query point
            t scalar: temporal query point

        Returns:
            pred scalar: prediction for u(x,t) at x=x and t=t.
        """
        branch_out = self.eval_branch(a)
        trunk_out = self.eval_trunk(x,t)
        
        return jnp.dot(branch_out,trunk_out) + self.bias[0]
    
    def multiple_query_points_one_a(self, a, x, t):
        branch_out = self.eval_branch(a) # (p,)
        trunk_out = vmap(self.eval_trunk)(x,t) # (num_points, p)
        
        return trunk_out @ branch_out + self.bias[0] # (num_points,)
    
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
        #return vmap(vmap(self, (None, 0, None)), (None, None, 0))(a, x, t)
        xx, tt = jnp.meshgrid(x, t)
        return self.multiple_query_points_one_a(a, xx.ravel(), tt.ravel()).reshape(len(t), len(x))
    
    def predict_whole_grid_batch(self, a, x, t):
        """As the DeepONet gives the prediction at scalar x and t,
        this is a utility function for predicting over the whole grid, with x and t being vectors.

        Args:
            a (batch, M+1): input function
            x (M+1,): spatial grid
            t (N+1,): temporal grid

        Returns:
            pred (batch, N+1, M+1): the prediction for u(x,t) for all grid points given by x and t
        """
        xx, tt = jnp.meshgrid(x, t)
        branch_out = vmap(self.eval_branch)(a) # (batch, p)
        trunk_out = vmap(self.eval_trunk, out_axes=1)(xx.ravel(),tt.ravel()) # (p, num_points)
        
        # (batch, p) @ (p, num_points) -> (batch, num_points)
        return (branch_out @ trunk_out).reshape(a.shape[0], len(t), len(x)) + self.bias[0]
    
    def get_self_adaptive(self):
        return self.self_adaptive
    
    def Dx(self, a, x, t):
        return grad(self, 1)(a, x, t)
    
    def Dt(self, a, x, t):
        return grad(self, 2)(a, x, t)
    
    def u_t(self, a, x, t):
        return self.Dt(a, x, t)*(self.u_std+self.eps)/(self.t_std + self.eps)
    
    def u_x(self, a, x, t):
        return self.Dx(a, x, t)*(self.u_std+self.eps)/(self.x_std + self.eps)
    
    def encoder(self, a):
        """
        Downsamples the input function a to the size of the branch network input size.
        Assumes that the input function is equispaced, and that the grid points are a multiple of "number_of_sensors".
        """
        return a[::len(a)//self.number_of_sensors]
   
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
        #trunk_width=trial.suggest_int("trunk_width", 50, 150)
        #branch_width=trial.suggest_int("branch_width", 50, 150)
        #trunk_depth=trial.suggest_int("trunk_depth", 3, 10)
        #branch_depth=trial.suggest_int("branch_depth", 3, 10)
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
        λ_smooth_or_sharp = None
        is_self_adaptive = True
        if is_self_adaptive:
            λ_learning_rate=trial.suggest_float("λ_learning_rate", 1e-4, 1e-1)
            λ_mask=trial.suggest_categorical("λ_mask", ["exponential", "polynomial", "logistic"])
            λ_learnable=trial.suggest_categorical("λ_learnable", [True, False])
            if not λ_learnable:
                λ_smooth_or_sharp = trial.suggest_categorical("λ_smooth_or_sharp", ["smooth", "sharp"])
            λ_optimizer = optax.chain(optax.adam(λ_learning_rate), optax.scale(-1.))
            opt = optax.multi_transform({'θ': θ_optimizer, 'λ': λ_optimizer}, param_labels=param_labels)
        #else:
        #    λ_learnable = None
        #    λ_learning_rate = None 
        #    λ_mask = None
        #    opt = θ_optimizer
        
        # Initialize the model and the trainer
        hparams = Hparams(
            #trunk_width=trunk_width,
            #branch_width=branch_width,
            #trunk_depth=trunk_depth,
            #branch_depth=branch_depth,
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
        
        model = DeepONet(hparams)
        if replicated:=self.trainer_kwargs.get("replicated"):
            model = eqx.filter_shard(model, replicated)
        opt_state = opt.init(eqx.filter([model], eqx.is_array))

        trainer = Trainer(model, opt, opt_state, self.train_loader, self.val_loader, trial=trial, **self.trainer_kwargs)
        trainer()
        best_val_loss = trainer.best_val_loss
        
        del trainer, opt_state, opt
        return best_val_loss
    
def compute_loss(model, a, u, key):
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
    t_idx = random.randint(t_key, (batch_size, model.num_query_points), 0, Np1)
    x_idx = random.randint(x_key, (batch_size, model.num_query_points), 0, Mp1)

    # Select the ground truth data at the query points
    # Has shape (batch_size, num_query_points)
    u_at_query_points = u[jnp.arange(batch_size)[:, None], t_idx, x_idx]
    
    # For each input function, compute the prediction of the model at the query points. (inner vmap)
    # Do this for each sample in the batch. (outer vmap)
    # Has shape (batch_size, num_query_points)
    u_pred = vmap(model.multiple_query_points_one_a)(a, Trainer.x[x_idx], Trainer.t[t_idx])
    
    #compute the loss 
    u_norms = jnp.linalg.norm(u_at_query_points, 2, 1)
    if model.is_self_adaptive:
        λ = model.self_adaptive(t_idx) 
        diff_norms = jnp.sqrt(jnp.sum(λ * jnp.square(u_at_query_points - u_pred), axis=1))
    else:
        diff_norms = jnp.linalg.norm((u_at_query_points - u_pred), 2, 1)
        
    loss = jnp.mean(diff_norms/u_norms)
    return loss