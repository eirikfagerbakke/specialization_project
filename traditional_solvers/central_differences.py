import jax 
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def Dx(y, dx, order = 6, axis=0):
    """Assumes periodic boundary conditions"""
    y_p_1 = jnp.roll(y, shift=-1, axis=axis)
    y_m_1 = jnp.roll(y, shift=1, axis=axis)
    if order == 2:
        return (y_p_1 - y_m_1) / (2 * dx)
    elif order == 4:
        y_p_2 = jnp.roll(y, shift=-2, axis=axis)
        y_m_2 = jnp.roll(y, shift=2, axis=axis)
        return (-y_p_2 + 8*y_p_1 - 8*y_m_1 + y_m_2)/(12*dx)
    elif order == 6:
        y_p_2 = jnp.roll(y, shift=-2, axis=axis)
        y_m_2 = jnp.roll(y, shift=2, axis=axis)
        y_p_3 = jnp.roll(y, shift=-3, axis=axis)
        y_m_3 = jnp.roll(y, shift=3, axis=axis)
        return (y_p_3 - 9*y_p_2 + 45*y_p_1 - 45*y_m_1 + 9*y_m_2 - y_m_3)/(60*dx)
    else:
        raise ValueError("Only 2nd, 4th and 6th order accurate first derivatives are implemented")
        

def Dxx(y, dx, order = 6, axis=0):
    """Assumes periodic boundary conditions"""
    y_p_1 = jnp.roll(y, shift=-1, axis=axis)
    y_m_1 = jnp.roll(y, shift=1, axis=axis)
    
    if order == 2:
        return (y_p_1 - 2 * y + y_m_1) / dx**2
    elif order == 4:
        y_p_2 = jnp.roll(y, shift=-2, axis=axis)
        y_m_2 = jnp.roll(y, shift=2, axis=axis)
        return (-y_p_2+16*y_p_1-30*y+16*y_m_1-y_m_2)/(12*dx**2)
    elif order == 6:
        y_p_2 = jnp.roll(y, shift=-2, axis=axis)
        y_m_2 = jnp.roll(y, shift=2, axis=axis)
        y_p_3 = jnp.roll(y, shift=-3, axis=axis)
        y_m_3 = jnp.roll(y, shift=3, axis=axis)
        return (270*y_m_1 - 27*y_m_2 + 2*y_m_3 + 270*y_p_1 - 27*y_p_2 + 2*y_p_3 - 490*y) / (180*dx**2)
    else:
        raise ValueError("Only 2nd, 4th and 6th order accurate second derivatives are implemented")