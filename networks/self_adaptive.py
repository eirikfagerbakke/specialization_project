import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable, Optional
from jaxtyping import Array
from dataclasses import dataclass

@dataclass(kw_only=True, frozen=True)
class SAHparams:
    """Stores the hyperparameters for the self adaptive weights λ"""
    λ_learning_rate: float = None # learning rate for the self adaptive weights λ
    λ_optimizer: str = "adam" # optimizer to use for the self adaptive weights λ
    λ_mask: str = "polynomial" # mask to apply to the λ-weights, before multiplying them with the loss
    λ_learnable: bool = False # whether the mask should be learnable
    λ_shape: int = 64
    λ_smooth_or_sharp: str = "smooth" # parameter for the masks, if they are fixed

class SelfAdaptive(eqx.Module):
    """Module initializes and stores self adaptive weights for the network."""
    λ: Array #the actual self adaptive weights
    λ_mask: Callable #mask to apply to the λ-weights, before multiplying them with the loss
    λ_mask_inv: Callable #inverse of the mask
    λ_shape: int #shape of the λ-weights
    λ_inv_min: float #minimum value for the λ-weights
    λ_inv_max: float #maximum value for the λ-weights
    a: Array | float #parameter for the masks, if initialized as array it is learnable
    
    def __init__(self, hparams : SAHparams):
        self.λ_shape = hparams.λ_shape
        self.λ = jnp.ones(self.λ_shape)
        if hparams.λ_learnable:
            self.a = jnp.ones(1)  
        else:
            if hparams.λ_mask == "logistic":
                if hparams.λ_smooth_or_sharp == "smooth":
                    self.a = 5.
                else:
                    self.a = 50.
            elif hparams.λ_mask == "polynomial" or hparams.λ_mask == "exponential":
                if hparams.λ_smooth_or_sharp == "smooth":
                    self.a = 1.
                else:
                    self.a = 3.               
        
        AVAILABLE_MASKS = {
            "exponential" : lambda λ: jnp.exp(jnp.abs(self.a)*(λ-1)),
            "polynomial" : lambda λ: jnp.where(λ >= 1, λ**jnp.abs(self.a), jnp.exp(jnp.abs(self.a)*(λ-1))),
            "logistic" : lambda λ: 2/(1+jnp.exp(jnp.abs(self.a)*(-λ+1))),
        }

        INVERSE_MASKS = {
            "exponential" : lambda λ: (jnp.log(λ)+jnp.abs(self.a))/jnp.abs(self.a),
            "polynomial" : lambda λ: jnp.where(λ >= 1, λ**(1/jnp.abs(self.a)), (jnp.log(λ) + jnp.abs(self.a))/jnp.abs(self.a)),
            "logistic" : lambda λ : 1/jnp.abs(self.a) *(jnp.abs(self.a) -jnp.log((-λ + 2) / λ)),
        }
        
        RANGE = {
            "exponential" : (0.25, None),
            "polynomial" : (0.25, None),
            "logistic" : (0.5, 1.5),
        }
        
        λ_range = RANGE[hparams.λ_mask]
        self.λ_inv_min = λ_range[0]
        self.λ_inv_max = λ_range[1]
        self.λ_mask = AVAILABLE_MASKS[hparams.λ_mask]
        self.λ_mask_inv = INVERSE_MASKS[hparams.λ_mask]
            
    def __call__(self, t_idx):        
        # Apply the operation based on the adjusted `t_idx`
        return self.λ_mask(jnp.take(self.λ, t_idx))
    
    def all_with_mask(self):
        return self.λ_mask(self.λ)
    
    def normalize(self):
        return self.λ_mask_inv(jnp.clip(self.λ/jnp.mean(self.λ), self.λ_inv_min, self.λ_inv_max))
    

def normalize_self_adaptive(self_adaptive):
    self_adaptive = eqx.tree_at(lambda x: x.λ, 
                                self_adaptive,
                                self_adaptive.λ_mask_inv(jnp.clip(self_adaptive.λ/jnp.mean(self_adaptive.λ), 
                                                                  self_adaptive.λ_inv_min, 
                                                                  self_adaptive.λ_inv_max)))
    return self_adaptive

def get_self_adaptive(model):
    """Retrieves a list of all the self-adaptive weights instances in the model."""
    is_self_adaptive = lambda x: isinstance(x, SelfAdaptive)
    return [x for x in jax.tree_util.tree_leaves(model, is_leaf=is_self_adaptive) if is_self_adaptive(x)]
