'''
Basic GNN using JAX

Based on:
 - https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
'''

import jax
import jax.numpy as jnp

##########################################
def main():
    print("Using jax", jax.__version__)
    print(f'Devices: {jax.devices()}')

    # jax.numpy provides a numpy-like interface to tensors on backend device (DeviceArray objects)
    a = jnp.zeros((2, 3), dtype=jnp.float32)
    #print(a.__class__)
    #print(f'a = {a}')

    # JAX arrays are immutable, so we must return a new array if we want to modify an existing one
    a_new = a.at[0, 0].set(1.)
    #print(f'a_new = {a_new}')

    # We can autodiff functions using jax.grad
    print()
    print("Let's do a simple test computing a gradient of a function...")
    print(f'y_i = <(x_i + 2)^2 + 3>')
    x = a_new
    y, dydx = jax.value_and_grad(simple_graph)(x)
    analytical_dydx = (1/a.size)*(2*x+4)
    print(f'x = {x}')
    print(f'y = {y}')
    print(f'dy/dx = {dydx}')
    print(f'analytical dy/dx: {analytical_dydx}')
    print()

#----------------------------------------
# y = <(x_i + 2)^2 + 3>
def simple_graph(x):
    x = x + 2
    x = x ** 2
    x = x + 3
    y = x.mean()
    return y

#----------------------------------------
#----------------------------------------
#----------------------------------------
if __name__ == '__main__':
    main()