import jax
import equinox as eqx
from jax import random, grad, value_and_grad, vmap
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type
from ._abstract_operator_net import AbstractOperatorNet, AbstractHparams
from ._self_adaptive import SelfAdaptive
import sys
sys.path.append("..")
from utils.trainer import Trainer
from jax.scipy.integrate import trapezoid

from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt

@dataclass(frozen=True, kw_only=True)
class EnergyNetHparams:
    depth: int
    width: int
    learning_rate: float

@dataclass(frozen=True, kw_only=True)
class Hparams:
    energy_net: EnergyNetHparams
    operator_net: AbstractHparams
    
class EnergyNet(eqx.Module):
    mlp : eqx.nn.MLP

    def __init__(self, hparams):
        activation = jax.nn.gelu
        key = random.key(0)

        self.mlp = eqx.nn.MLP(
            in_size=2, #takes u and u_x as inputs
            out_size='scalar',
            width_size=hparams.width,
            depth=hparams.depth,
            activation=activation,
            key=key,
        )
        
    def __call__(self, u, u_x):
        return self.mlp(jnp.array([u,u_x]))

class HamiltonianInformedNeuralOperator(eqx.Module):
    """A Hamiltonian PDE can be written on the form u_t = 𝒢 δℋ/δu.
    The operator network predicts the function u(x,t). From this function, we compute
    u_t and 𝒢 δℋ/δu using automatic differentiation.
    These terms should be equal, and can be added as a penalty term to the loss function.
    
    This gives us a network that is "informed" by the Hamiltonian structure of the PDE.
    """
    F : EnergyNet
    u : AbstractOperatorNet
    self_adaptive : SelfAdaptive
    
    def __init__(self, energy_net : eqx.Module, operator_net: AbstractOperatorNet):
        self.F = energy_net
        self.u = operator_net
        self.self_adaptive = self.u.self_adaptive

    def __call__(self, a, x, t):
        """The input and output of the HINO is the same as the input of the OperatorNet.
        In all cases, a is an array of point-evaluations of the function a.
        
        DeepONets: 
            Input:
                x: scalar
                t: scalar
            Output:
                u_t(x,t): scalar, at x=x and t=t
        FNO1d:
            Input:
                x: (M+1,)
                t: scalar
            Output:
                u_t(x,t): (M+1,) at t=t
        FNO2d:
            Input:
                x: (M+1,)
                t: (N+1,)
            Output:
                u_t(x,t): (N+1,M+1)
        """
        u_eval, u_t = value_and_grad(self.u.predict_whole_grid, 2)(a,x,t) # u and u_t, values
        
        u = lambda x : self.u(a, x, t) # u(x), function in x only (for convenience)
        u_x = grad(u) # u_x(x), function
        
        dFdu = lambda x : grad(self.F, 0)(u(x), u_x(x)) # ∂F/∂u(x), function
        dFdu_x = lambda x : grad(self.F, 1)(u(x), u_x(x)) # ∂F/∂u_x(x), function
        δℋ = lambda x : dFdu(x) - grad(dFdu_x)(x) # δℋ/δu(x), function
        
        𝒢δℋ =  grad(δℋ)(x) # 𝒢 δℋ/δu , value
        return u_eval, u_t, 𝒢δℋ