# Optimal coefficients functions for variance optimisation

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


def opt_coef_state (povm_mat, flat_obs, flat_rho):
    # optimal coefficients for the variance (and shadow norm) for a set of probabilities (fixed state)
    probs = nnz_probs(povm_mat, flat_rho)
    lmat = opt_inv_state(povm_mat, probs)
    
    return lmat@flat_obs


def opt_sn_state (povm_mat, flat_obs, flat_rho):
    # optimal shadow norm (first term in variance) given an observable and a particular state
    # uses previous functions for calls

    probs = nnz_probs(povm_mat, flat_rho)
    coefs = opt_coef_state(povm_mat, flat_obs, flat_rho)
    dmat = np.diag(probs)
    
    return np.real(np.conj(coefs)@dmat@coefs) # this should always be positive, reality imposed to ignore null imaginary part in function handling

