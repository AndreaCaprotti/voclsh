import numpy as np

import scipy as sc
from scipy import io
from scipy import optimize
import states_functions as sf

# Optimal coefficients functions for variance optimisation

def variance_pc (probs, coefs):
    # variance from choice of coefficients and probability ditribution
    c2 = coefs**2
    return probs@c2 -(probs@coefs)**2

                ##-----##

def nnz_probs(povm_mat, flat_rho):
    # finds probabilities of a given (flattened) state by projecting on the POVM effects. 
    # Additional check to guarantee that no probability is actually zero (would correspond to a noisy measurement) to avoid issues later
    probs = np.round(povm_mat@flat_rho,10)

    n = len(probs) # number of effects
    # check for zero probabilities 
    nz = len(np.where(probs == 0))

    if nz > 0:
        eps = (10e-9)*min(probs[np.where(probs>0)])/n**2 # makes this small so not to interfere too much on actual probabilities
        for i in range(n):
            if probs[i] == 0:
                probs[i] = eps                           # fixes minimal value to zero probabilities
            else:
                probs[i] = probs[i] - eps/nz             # subtracts contribution equally from all other probabilities
    # corresponds to presence of noise in the measurement process
    return probs

                ##-----##

def opt_invmat_state ( povm_mat, probs):
    # optimal estimator matrix for a given fixed state, given its projection on the POVM effects
    # represents a variation of usual Moore-Penrose pseudo-inverse
    dmat = np.diag(np.reciprocal(probs))
    
    return dmat@povm_mat@sc.linalg.inv(povm_mat.T @ dmat @ povm_mat)

                ##-----##

def opt_coef_state (povm_mat, flat_obs, flat_rho):
    # optimal coefficients for the variance (and shadow norm) for a set of probabilities (fixed state)
    probs = nnz_probs(povm_mat, flat_rho)
    lmat = opt_invmat_state(povm_mat, probs)
    
    return lmat@flat_obs

                ##-----##

def opt_sn_state (povm_mat, flat_obs, flat_rho):
    # optimal shadow norm (first term in variance) given an observable and a particular state
    # uses previous functions for calls

    probs = nnz_probs(povm_mat, flat_rho)
    coefs = opt_coef_state(povm_mat, flat_obs, flat_rho)
    dmat = np.diag(probs)
    
    return np.real(np.conj(coefs)@dmat@coefs) # this should always be positive, reality imposed to ignore null imaginary part in function handling

                ##-----##

def var_state_optimisation (x, povm_mat, basis_mat, flat_obs):
    # actual optimisation target function, constructing the state from the free parameters and estimates the variance 
    # using the optimal coefficients
    rho_flat = sf.flat_state_from_mat (x, basis_mat) # adapts free variables into a density matrix (in proper subspace)
    
    return - opt_sn_state(povm_mat, flat_obs, rho_flat) + np.real((rho_flat@flat_obs)**2)

                ##-----##

def variance_optimisation(povm_cm, basis_m ,observable):
    # the real heart of the module, where the magic actually happens
    # takes as input - the POVM coefficient matrix
    #                - the proper basis matrix
    #                - the target observable
    
    d = len(observable)  # this is a matrix, so this is dimension of *vector* space
    D = d**2
    flatobs = sf.flatten_in_basis(observable, basis_m)
    
    x0 = np.ones(D)/(D) # "maximally mixed" initial condition

    res = sc.optimize.minimize(var_state_optimisation, 
                                x0, 
                                args=(povm_cm, basis_m, flatobs),
                                method='powell',
                                tol = 10e-5
                            )
    
    xs = res.x
    flat_rhos = sf.flat_state_from_mat(xs, basis_m) # optimal state
    ocs  = opt_coef_state(povm_cm, flatobs, flat_rhos)
    rhos = np.reshape(basis_m@flat_rhos,(d,d))
    var  = -res.fun
    
    return res