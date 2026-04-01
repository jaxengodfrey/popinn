import jax.numpy as jnp

# def phi_equilibrium(x: jnp.ndarray, gamma: float, theta: float = 1.0, nu: float = 1.0):
#     """computes the steady state frequency spectrum
#     $\phi(x) = \frac{\theta \nu}{x (1 - x)} \frac{1 - e^{-2\gamma(1 - x)}}{1 - e^{-2\gamma}}$
#     Assumes no under/overdominance, i.e. h = 0.5

#     Args:
#         x (jnp.ndarray): frequency
#         gamma (float): scaled selection coefficient, equal to 2*Nref*s, where s is the selective advantage and Nref is the reference population size, typically the ancestral size.
#         theta (float, optional): Population scaled mutation rate, equal to 4*Nref*u, where u is the mutation event rate per generation and Nref is the reference population size, typically the ancestral size. Defaults to 1.0.
#         nu (float, optional): Population size relative to the reference population size Nref, i.e. nu = N/Nref. Defaults to 1.0.
#     """
#     phi_neutral = 1. / x
#     phi_selected = 1. / (x * (1. - x)) * jnp.expm1(-2. * gamma * (1. - x)) / jnp.expm1(-2. * gamma)
#     return theta * nu * jnp.where(jnp.less(jnp.abs(gamma), 1e-8), phi_neutral, phi_selected)


def g_equilibrium(x: jnp.ndarray, gamma: float,
                         theta: float = 1.0, nu: float = 1.0) -> jnp.ndarray:
    """computes the steady state frequency spectrum $g(x)$, a reparametrization that does not diverge at x = 0.
    $g(x) = x(1-x)phi(x)$
    Assumes no under/overdominance, i.e. h = 0.5

    Args:
        x (jnp.ndarray): frequency
        gamma (float): scaled selection coefficient, equal to 2*Nref*s, where s is the      selective advantage and Nref is the reference population size, typically the ancestral size.
        theta (float, optional): Population scaled mutation rate, equal to 4*Nref*u, where u is the mutation event rate per generation and Nref is the reference population size, typically the ancestral size. Defaults to 1.0.
        nu (float, optional): Population size relative to the reference population size Nref, i.e. nu = N/Nref. Defaults to 1.0.

    Returns:
        g (jnp.ndarray): a new scaled phi array
    """

    # g_neutral = 1. - x
    # g_selected = jnp.expm1(-2. * gamma * (1. - x)) / jnp.expm1(-2. * gamma)

    # return theta * nu * jnp.where(jnp.less(jnp.abs(gamma), 1e-8), g_neutral, g_selected)


    return 1. - x


