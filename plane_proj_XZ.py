# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Plane projectors optimal variance
# Let us consider one of the simplest cases: a (tensor) projector on a reduced subspace

# +
import pickle
import numpy as np

import scipy as sc
from scipy import io
from scipy import optimize

import time
import os

# Graphics
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
#rcParams['pcolor.shading']= 'auto'
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

cm = 1.0/2.54  # centimeters in inches
px = 1.0/plt.rcParams['figure.dpi']  # pixel in inches

goldenratio=1.618
imagewidth = 7 # 
# -

# homebrewed modules for the actual variance optimisation and handling of different functions
import opt_variance_coef as ocf  # variance optimisation related functions
import states_functions as sf    # state and observables related functions
import povm_functions as pf      # POVM generation and handling related functions

# +
# global variables
single_povm = pf.pauli_povm_single('X','Z') # single qubit plane POVM

density = 30  # number of different projectors considered

N_min = 1     # minimal dimension considered
N_max = 2     # maximal dimension considered

# saving directory (created if non-existing)
directory= 'plane_proj_XZ'
os.system(f'mkdir {directory}')
# -

# # Cycle over different dimensions

for N in range(1,N_max+1):
    filename = directory+f'_{N}' # just to be clear, file name is the same as directory
    # tensor POVM definition
    povm = pf.tensor_same(single_povm, N)
    [pcm,bm] = pf.povm_coef_matrix(povm)
    can_em = sc.linalg.pinv(pcm) # returns the Moore-Penrose pseudo-inverse (canonical inverse)

    # canonical inversion
    single_obs = sf.qubit(0,0)          # we've seen that it's the same for all projectors of the same size (for the plane case at least)
    obs = pf.tensor_same(single_obs,N)
    cc = can_em.T@(sf.flatten_in_basis(obs,bm)) # canonical coefficients
    [var_can, rho_can] = ocf.fix_coef_var_optimisation(pcm, bm, cc) # canonical maximal variance
    np.save(f'{directory}/{filename}_can',var_can)

    # optimal upper bound on variance
    var_vec = []
    for j in range(density):
        theta = j*np.pi/(2*density) # angle runs between 0 and \pi
        phi = 0                     # we consider projectors on the prime meridian
        
        singleobs = sf.qubit(theta, phi)
        obs = pf.tensor_same(singleobs, N) # N-tensor of same observable
        # optimal variance
        [var, rho, oc] = ocf.variance_optimisation(pcm, bm ,obs) # actual variance optimisation

        var_vec.append(var)
    np.save(f'{directory}/{filename}_opt',var_vec)


