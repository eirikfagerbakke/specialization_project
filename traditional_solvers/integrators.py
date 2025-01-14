import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optimistix as optx
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def implicit_midpoint(f, u0, dt, t, args, rtol, atol, max_steps = 20):
    def step(carry, tn):
        un, dt = carry

        fn = f(tn, un, args)

        # The update should satisfy y1 = eq(y1), i.e. y1 is a fixed point of fn
        def eq(u, args):
            return un + dt * f(tn+0.5*dt, 0.5*(un+u), args)

        u_next_euler = un + dt * fn # Euler step as guess

        solver = optx.Chord(rtol, atol)
        u_next = optx.fixed_point(eq, solver, u_next_euler, args, max_steps = max_steps).value  # satisfies y1 == fn(y1)
        return (u_next, dt), un
    
    _, u_arr = jax.lax.scan(step, (u0, dt), t)
    return u_arr

@partial(jax.jit, static_argnums=(0,))
def gauss_legendre_4(f, u0, dt, t, args, rtol, atol, max_steps = 20):
    """
    Integrates the ODE system using the Gauss-Legendre method of order 4.
    Implementation follows "IV.8 Implementation of Implicit Runge-Kutta Methods" in 
    "Solving Ordinary Differential Equations II" by Hairer and Wanner

    Args:
      f: The right-hand side function of the ODE system.
      u0: Initial condition.
      dt: Time step.
      t: Array of time points.
      args: Additional arguments to pass to f.
      rtol: Relative tolerance for the nonlinear solver.
      atol: Absolute tolerance for the nonlinear solver.

    Returns:
      An array of solution values at the given time points.
    """
    c = jnp.array([0.5 - jnp.sqrt(3)/6, 0.5 + jnp.sqrt(3)/6])
    A = jnp.array([[0.25, 0.25 - jnp.sqrt(3)/6], 
                   [0.25 + jnp.sqrt(3)/6, 0.25]])
    d = jnp.array([-jnp.sqrt(3), jnp.sqrt(3)])
    
    def q(x, z_next):
        return z_next[0]*(x-c[1])/(c[0]-c[1])*x/c[0] + z_next[1]*(x-c[0])/(c[1]-c[0])*x/c[1]
    
    def step(carry, tn): 
        un, z_guess = carry
        
        def eq(z, args):
            z1 = dt*(A[0,0] * f(tn + c[0]*dt, un + z[0], args) + A[0,1]*f(tn + c[1]*dt, un + z[1], args))
            z2 = dt*(A[1,0] * f(tn + c[0]*dt, un + z1, args) + A[1,1]*f(tn + c[1]*dt, un + z[1], args))
            return z - jnp.array([z1, z2])
        
        solver = optx.Chord(rtol, atol)
        z_next = optx.root_find(eq, solver, z_guess, args, throw=False, max_steps = max_steps).value
        u_next = un + jnp.dot(d, z_next)
        
        z_guess = q(1+c[:,None], z_next)+un-u_next
        return (u_next, z_guess), un

    z_guess = jnp.zeros((2, u0.shape[0]))
    _, u_arr = jax.lax.scan(step, (u0, z_guess), t)
    return u_arr

@partial(jax.jit, static_argnums=(0,))
def gauss_legendre_6(f, u0, dt, t, args, rtol, atol, max_steps=20):
    """
    Integrates the ODE system using the Gauss-Legendre method of order 6.
    Implementation follows "IV.8 Implementation of Implicit Runge-Kutta Methods" in 
    "Solving Ordinary Differential Equations II" by Hairer and Wanner

    Args:
      f: The right-hand side function of the ODE system.
      u0: Initial condition.
      dt: Time step.
      t: Array of time points.
      args: Additional arguments to pass to f.
      rtol: Relative tolerance for the nonlinear solver.
      atol: Absolute tolerance for the nonlinear solver.
      

    Returns:
      An array of solution values at the given time points.
    """
    c = jnp.array([0.5 - jnp.sqrt(15)/10, 0.5, 0.5 + jnp.sqrt(15)/10])
    A = jnp.array([[5/36, 2/9-jnp.sqrt(15)/15, 5/36-jnp.sqrt(15)/30],
                    [5/36+jnp.sqrt(15)/24, 2/9, 5/36-jnp.sqrt(15)/24],
                    [5/36+jnp.sqrt(15)/30, 2/9+jnp.sqrt(15)/15, 5/36]])
    d = jnp.array([5/3, -4/3, 5/3])
    
    def q(x, z_next):
        z_guess = z_next[0]*(x-c[1])/(c[0]-c[1])*x/c[0]*(x-c[2])/(c[0]-c[2])
        z_guess += z_next[1]*(x-c[0])/(c[1]-c[0])*x/c[1]*(x-c[2])/(c[1]-c[2])
        z_guess += z_next[2]*(x-c[0])/(c[2]-c[0])*(x-c[1])/(c[2]-c[1])*x/c[2]
        return z_guess
    
    @jax.jit
    def step(carry, tn): 
        un, z_guess = carry
        
        def eq(z, args):
            z0 = dt*(A[0,0]*f(tn + c[0]*dt, un + z[0], args) + A[0,1]*f(tn + c[1]*dt, un + z[1], args) + A[0,2]*f(tn + c[2]*dt, un + z[2], args))
            z1 = dt*(A[1,0]*f(tn + c[0]*dt, un + z0, args) + A[1,1]*f(tn + c[1]*dt, un + z[1], args) + A[1,2]*f(tn + c[2]*dt, un + z[2], args))
            z2 = dt*(A[2,0]*f(tn + c[0]*dt, un + z0, args) + A[2,1]*f(tn + c[1]*dt, un + z1, args) + A[2,2]*f(tn + c[2]*dt, un + z[2], args))
            return z - jnp.array([z0, z1, z2])
        
        solver = optx.Chord(rtol, atol)
        z_next = optx.root_find(eq, solver, z_guess, args, throw=False, max_steps=max_steps).value
        u_next = un + jnp.dot(d, z_next)
            
        z_guess = q(1+c[:,None], z_next)+un-u_next
        return (u_next, z_guess), un

    z_guess = jnp.zeros((3, u0.shape[0]))
    _, u_arr = jax.lax.scan(step, (u0, z_guess), t)
    return u_arr