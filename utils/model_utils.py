import json
import equinox as eqx
from dataclasses import asdict
from typing import Type, Any
from jaxtyping import PyTree
import jax
jax.config.update("jax_enable_x64", True)
import equinox as eqx
from jax import random
import jax.numpy as jnp
import optax
import sys
sys.path.append("..")
from networks.self_adaptive import get_self_adaptive

def save_model(model: eqx.Module, hparams: Any, filename: str) -> None:
    """Saves an Equinox model and its hyperparameters to a file.

    Args:
        model (eqx.Module): The model instance to save.
        hparams (Any): An instance of a dataclass containing the hyperparameters.
        filename (str): The filename to save the model to.
    """
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(asdict(hparams))
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load_model(Network: Type[eqx.Module], Hparams: Type[Any], filename: str) -> eqx.Module:
    """Loads a saved model and its hyperparameters from a file.

    Args:
        Network (Type[eqx.Module]): The model class (not an instance).
        Hparams (Type[Any]): The dataclass type for the hyperparameters.
        filename (str): The filename to load the model from.

    Returns:
        eqx.Module: The deserialized model instance.
    """
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = Network(Hparams(**hyperparams))
        return eqx.tree_deserialise_leaves(f, model)
    
    
def init_F(model, key, scale=1e-2):
    def init_fn(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        out, in_ = weight.shape
        return random.normal(key, (out, in_))*scale
    
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    weights = get_weights(model)
    new_weights = [init_fn(weight, subkey)
                    for weight, subkey in zip(weights, random.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model

def init_he(model, key):
    def init_fn(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        out, in_ = weight.shape
        init_std = jnp.sqrt(2.0 / in_)
        return random.normal(key, (out, in_))*init_std
    
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    weights = get_weights(model)
    new_weights = [init_fn(weight, subkey)
                    for weight, subkey in zip(weights, random.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model

def param_labels(listed_model : list[eqx.Module]) -> list[PyTree]:
    model = listed_model[0]
    labels = jax.tree.map(lambda _: 'θ', model) # the regular weights are labeled 'θ' 
    if hasattr(model, 'self_adaptive'):
        labels = eqx.tree_at(lambda m: m.self_adaptive, labels, 'λ') # the self-adaptive weights of u are labeled 'λ_u'
        
    if hasattr(model, 'F'):
        labels = eqx.tree_at(lambda m: m.F, labels, 'φ') # the energy net weights are labeled 'φ'
        if hasattr(model, 'F.self_adaptive'):
            labels = eqx.tree_at(lambda m: m.F.self_adaptive, labels, 'λ_F') # the self-adaptive weights of F are labeled 'λ_F'
        if hasattr(model, 'u.self_adaptive'):
            labels = eqx.tree_at(lambda m: m.u.self_adaptive, labels, 'λ_u') # the self-adaptive weights of u are labeled 'λ_u'
    return [labels]

def param_labels_hno_self_adaptive(listed_model : list[eqx.Module]) -> list[PyTree]:
    """Assigns labels to the weights of the model. Specifically, the regular weights are labeled 'θ', and the self-adaptive weights are labeled 'λ'.

    Args:
        listed_model (list[eqx.Module]): The model to assign labels to. Has to be listed, to make it un-callable. 

    Returns:
        list[PyTree]: The (listed) model with assigned labels.
    """
    model = listed_model[0]
    labels = jax.tree.map(lambda _: 'θ', model) # the regular weights are labeled 'θ'
    labels = eqx.tree_at(lambda m: m.F, labels, 'φ') # the energy net weights are labeled 'φ'
    #labels = eqx.tree_at(lambda m: m.F.self_adaptive, labels, 'λ_F') # the self-adaptive weights of F are labeled 'λ_F'
    labels = eqx.tree_at(lambda m: m.u.self_adaptive, labels, 'λ_u') # the self-adaptive weights of u are labeled 'λ_u'
    return [labels]

def param_labels_hno(listed_model : list[eqx.Module]) -> list[PyTree]:
    """Assigns labels to the weights of the model. Specifically, the regular weights are labeled 'θ', and the self-adaptive weights are labeled 'λ'.

    Args:
        listed_model (list[eqx.Module]): The model to assign labels to. Has to be listed, to make it un-callable. 

    Returns:
        list[PyTree]: The (listed) model with assigned labels.
    """
    model = listed_model[0]
    labels = jax.tree.map(lambda _: 'θ', model) # the regular weights are labeled 'θ'
    labels = eqx.tree_at(lambda m: m.F, labels, 'φ')
    return [labels]
    
def param_count(model: eqx.Module) -> int:
    """Counts the number of parameters in a model.

    Args:
        model (eqx.Module): The model to count the parameters of.

    Returns:
        int: The number of parameters in the model.
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))

def conjugate_grads_transform():
    def init_fn(params):
        # Returns an empty state
        return None

    def update_fn(updates, state, params=None):
        # Conjugate the gradients if they are complex
        updates = jax.tree_util.tree_map(
            lambda g: jnp.conj(g) if jnp.iscomplexobj(g) else g, updates
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)
