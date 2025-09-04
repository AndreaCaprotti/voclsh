# additional useful functions

def qubit (theta, phi):
    # single qubit projector definition from angle parameters
    vec = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
    return np.round(np.outer(vec,np.conj(vec)),10)
                   
def flat_state_from_mat (x, basis_mat):
    # generates a (flattened) quantum states from an array of free parameters interpreted as the
    # real and imaginary part of a trinagular matrix, used to generate the state
    
    d = int(max(basis_mat.shape)) # dimension of vector, corresponds to vector space dimension
    dd = int(min(basis_mat.shape)) # dimension of subspace

    # check of consistent dimensions
    if np.sqrt(len(x)) != d:
        raise ValueError('dimension of input variables not consistent with basis expressed')
    
    # to easily build the triangular matrix, inputs are reshaped in a square matrix
    xmat = np.reshape(x,(d,d)) 
    re = np.tril(xmat) 
    im = np.tril(xmat.T,k=-1) # this selects all elements in *upper* triangular matrix of xmat
    T = re+1j*im              # triangular matrix
    H = T@np.conj(T.T)        # positive semidefinite version by def.

    if dd < d:              # dimension of subspace smaller than HS space
        # to ensure the matrix is in the relevant subspace, it is projected onto the basis and
        # then expanded back into the full HS space for correct normalisation
        H = np.reshape(np.conj(basis_mat.T)@H.flatten(),(d,d))
    
    rho = H / np.trace(H)                        # normalisation
    return np.conj(basis_mat.T)@(rho.flatten())  # final projection into proper subspace and flattening
