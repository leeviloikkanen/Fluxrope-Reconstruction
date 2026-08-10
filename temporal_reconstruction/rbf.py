"""

File to create RBF reconstructed field
Ease of modifying/changing technique in the future 

"""
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from sklearn.neighbors import NearestNeighbors
import scipy


#LOOCV METHOD
def E_func(eps, centers, values, kernel, interpolator):
    #O(N³) so scales poorly with number of points
    N_pts = np.shape(centers)[0]
    L= np.shape(centers)[1]
    E = np.zeros([N_pts,L])
    eps = abs(eps)
    for i in range(N_pts):
        r_used = np.vstack((centers[:i,:],centers[i+1:,:]))
        b_used = np.vstack((values[:i,:],values[i+1:,:]))
        rbf_trial = interpolator(r_used,b_used, kernel=kernel,
                    epsilon=eps,
                    smoothing=0.0
                    )
        
        B_recon_rbf = rbf_trial(centers[i][None, :])[0]
       
        B_true = values[i,:]
       
        E[i,:] = B_true - B_recon_rbf
    
    return scipy.linalg.norm(E)

#Slow own minimizatin function. Probably better to try use something like 
#scipy.optimization.minimize. Values very small tho
def find_eps(centers, values, kernel, interpolator, style = "log", start = -20, end = 20, Num = 200):
    #Simple function to loop through epsilon values to find best one
    if style == "log":
        slots = np.logspace(start, end, Num)
    elif style == "linear":
        slots = np.linspace(start,end,Num)
    else:
        raise "Invalid style: either linear or log"
    
    min_eps = 1
    min = 1
    for i in slots:
        
        res = E_func(i,centers, values, kernel, interpolator)
        
        #print(res)
        if res< min:
            min = res
            min_eps = i
    return min_eps, min

def find_eps_LOSO(centers, values, groups, kernel, interpolator = RBFInterpolator, eps_start = -9, eps_end = -5, eps_num = 100):
    centers = np.asarray(centers)
    values = np.asarray(values)
    groups = np.asarray(groups)
    folds = [(np.nonzero(groups == g)[0], np.nonzero(groups !=g)[0]) for g in np.unique(groups)]

    eps_grid = np.logspace(eps_start,eps_end, eps_num)
    best = None
    for eps in eps_grid:
        total_sq = 0.0
        for hold_idx, train_idx in folds:
            interp = interpolator(centers[train_idx], values[train_idx], epsilon=eps, kernel = kernel)
            pred = interp(centers[hold_idx])
            total_sq += np.sum((pred-values[hold_idx])**2)

        if best is None or total_sq < best[1]:
            best = (eps, total_sq)
    eps, total_sq = best
    return eps, total_sq

#MAIN RBF reconstruction function 
def RBF_missing_data(df, sc_names, missing_sc = None, eps_method = "neighbour", kernel = "multiquadric", interpolator = RBFInterpolator):
    #Modify to select only sc that aren't in missing_sc then just same things as below: 

    if missing_sc is None:
        included_sc = sc_names
    else:
        included_sc = [sc for sc in sc_names if sc not in missing_sc]

    included_pos_cols = sum([[f"{sc}_pos_x", f"{sc}_pos_y", f"{sc}_pos_z"] for sc in included_sc], [])
    included_B_cols = sum([[f"{sc}_vg_B_x", f"{sc}_vg_B_y", f"{sc}_vg_B_z"] for sc in included_sc], [])

    centers_inc = df[included_pos_cols].to_numpy().reshape(-1, 3)
    values_inc = df[included_B_cols].to_numpy().reshape(-1, 3)
    #pick epsilon
    "https://www.math.iit.edu/~fass/Dolomites.pdf?" #nearest neighbor method mentioned
    nbrs = NearestNeighbors(n_neighbors=2).fit(centers_inc)
    dists, _ = nbrs.kneighbors(centers_inc)

    if eps_method == "neighbour":
        epsilon = np.median(dists[:, 1])
        epsilon = 1/epsilon
    elif eps_method == "LOOCV":
        #This is very slow and seemingly choise of epsilon >1e-3 makes little difference 
        #run once and the manually set found epsilon.
        epsilon, _ = find_eps(centers_inc,values_inc, kernel, interpolator)
    elif eps_method == "LOSO":
        """
        Leave one spacecraft out cross validation. remove a entire spacecraft
        dataset to properly make it effect the reconstruction quality
        """
        groups = np.tile(np.arange(len(included_sc)), len(df))
        epsilon, sq = find_eps_LOSO(centers_inc, values_inc, groups, kernel, interpolator)

    else:
        raise ValueError("No valid epsilon method set!")
    print(f"RBF epsilon (missing {missing_sc}) = {epsilon:.3g}")    
    
    #RBF interpolation
    rbf = interpolator(
        centers_inc, values_inc,
        epsilon=epsilon,
        kernel = kernel,
        smoothing=0.0
    )

    return rbf, included_pos_cols, included_B_cols, included_sc

