"""

while many of the functions used in this project are created from scratch,
they are compared and testsed agaisnt open source equivalent builds using 
pytorch, JAX, and other libraries

"""

import jax.numpy as jnp 
import numpy as np
import matplotlib.pyplot as plt

def test():
    x_jnp = jnp.linspace(0, 10, 100)
    y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
    plt.plot(x_jnp, y_jnp)
    plt.show()
    
if __name__ == "__main__":
    test()