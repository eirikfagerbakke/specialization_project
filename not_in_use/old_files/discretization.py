import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
import jax
from typing import Callable

# Represents the interval [x0, x_final] discretized into n equally-spaced points.
class SpatialDiscretization(eqx.Module):
    x0: float = eqx.field(static=True)
    x_final: float = eqx.field(static=True)
    vals: Float[Array, "M"]

    @classmethod
    def discretize_fn(cls, x0: float, x_final: float, M: int, fn: Callable):
        if M < 2:
            raise ValueError("Must discretize [x0, x_final] into at least two points")
        vals = jax.vmap(fn)(jnp.linspace(x0, x_final, M))
        return cls(x0, x_final, vals)
    
    @property
    def M(self):
        return len(self.vals) - 1

    @property
    def δx(self):
        return (self.x_final - self.x0) / self.M
    
    @property
    def x(self):
        return jnp.linspace(self.x0, self.x_final, len(self.vals))

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretization):
            if self.x0 != other.x0 or self.x_final != other.x_final:
                raise ValueError("Mismatched spatial discretizations")
            other = other.vals
        return SpatialDiscretization(self.x0, self.x_final, fn(self.vals, other))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)
    
    def __getitem__(self, idx):
        return self.vals[idx]

@jax.jit
def central_difference_1(y: SpatialDiscretization) -> SpatialDiscretization:
    y_next = jnp.roll(y.vals, shift=1)
    y_prev = jnp.roll(y.vals, shift=-1)
    
    return SpatialDiscretization(y.x0, y.x_final, (y_next - y_prev) / (2 * y.δx))

@jax.jit
def central_difference_2(y: SpatialDiscretization) -> SpatialDiscretization:
    y_next = jnp.roll(y.vals, shift=1)
    y_prev = jnp.roll(y.vals, shift=-1)
    
    return SpatialDiscretization(y.x0, y.x_final, (y_next - 2 * y.vals + y_prev) / y.δx**2)

class SpatioTemporalDiscretization(eqx.Module):
    x0: float = eqx.field(static=True)
    x_final: float = eqx.field(static=True)
    t0: float = eqx.field(static=True)
    t_final: float = eqx.field(static=True)
    vals: Float[Array, "N+1 M"]
    
    def from_spatial_discretization(u: SpatialDiscretization, t0: float, t_final: float):
        return SpatioTemporalDiscretization(u.x0, u.x_final, t0, t_final, u.vals[jnp.newaxis,:])
    
    @property
    def M(self):
        return len(self.vals[0])
    
    @property
    def N(self):
        return len(self.vals) - 1
    
    @property
    def δx(self):
        return (self.x_final - self.x0) / (self.M-1)
    
    @property
    def δt(self):
        return (self.t_final - self.t0) / self.N
    
    @property
    def x(self):
        return jnp.linspace(self.x0, self.x_final, self.M)
    
    @property
    def t(self):
        return jnp.linspace(self.t0, self.t_final, self.N + 1)