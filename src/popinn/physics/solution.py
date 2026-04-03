import dadi
import numpy as np
import jax.numpy as jnp

def calc_dadi(gamma_evol, gamma_init, N_func, tf = 1., ns = 200, pts = 300):
    xx = dadi.Numerics.default_grid(pts = pts)
    phi0 = dadi.PhiManip.phi_1D(xx, gamma = gamma_init)
    phif = dadi.Integration.one_pop(phi0, xx, tf, N_func, gamma = gamma_evol)
    return xx, phif

def get_final_sols(gamma_evol, gamma_init, tf = 1., ns = 200, pts = 300):

    sols = np.zeros((gamma_evol.shape[0], pts))

    for idx, gam in enumerate(gamma_evol):
        x, f = calc_dadi(gam, gamma_init, lambda t: 1., tf = tf, ns = ns, pts = 300)
        g = f * x * (1. - x) 
        sols[idx] = g

    return jnp.asarray(x[1:]), jnp.asarray(sols[:,1:])