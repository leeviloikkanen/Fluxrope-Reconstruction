#This script was created by Leevi Loikkanen to do error analysis between Vlasiator simulation files 
#and reconstructed magnetic fields from virtual spacecraft data using Radial Basis Functions  
#by means of point-wise error and Wasserstein distances 
#RBF method and error analysis methods used here outlined in paper: 
#https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2023EA003369
#Specifically timeseries reconstruction of FHA run between 1340-1372s
#Determined Bulk velocity from 4 outside spacecraft average to avoid bias of inner tetrahedron. 
#Timeseries start when spacecrafts come in contact with flux rope structure and ends when last spacecraft loses contact
#Taylor's hypothesis is assumed to hold (at least for the fluxrope)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
import matplotlib as mpl
import analysator as pt

"""
SC1–4 overall means (from 1353 onwards):
  vg_v_x = -884632.1570458194
  vg_v_y = -332263.0899893005
  vg_v_z = 165187.1091635656
"""
output_dir ="/home/leeviloi/fluxrope_thesis/timeseries_tail/"
vg_v_x = -739256.9
vg_v_y = -268152.8
vg_v_z = 147101.5

vel_bulk = -1*np.array([vg_v_x,vg_v_y,vg_v_z])

#TIP for future Leevi KEEP EVERYTHING IN METERS
#well done 
#Time series reconstruction code structure
"""
Only main differences to static case: 
    -No constant vlasiator time to compare to (still have to figure this out in terms of coordinates and stuff)
    -Instead of having the centers ready from the file using the bulk velocity calculating the
     centers for each timestep.
    -other than that everything prettymuch the same. 
    IDEA: for first problem on list: select constant Vlasiator timestep where fluxrope is the best. have that basis for coordinates
    TODO see IDEA
"""
#COPIED FROM RBF_error.py

#Shared info
df = pd.read_csv("/home/leeviloi/plas_obs_vg_b_timeseries_tail_right_z=0.5.csv")

R_e = 6371000   

#STARTING SC locations 
sc_init = {
    "sc1": np.array([-27.0, 3.0, 0.5]) * R_e,
    "sc2": np.array([-26.0, 3.0, 1.5]) * R_e,
    "sc3": np.array([-26.0, 3.86602540, 0.0]) * R_e,
    "sc4": np.array([-26.0, 2.13397460, 0.0]) * R_e,
    "sc5": np.array([-26.85714286, 3.0, 0.64285714]) * R_e,
    "sc6": np.array([-26.85714286, 3.12371791, 0.42857143]) * R_e,
    "sc7": np.array([-26.85714286, 2.87628209, 0.42857143]) * R_e,
}

times = df["Timeframe"].to_numpy() 
T = len(times)
sc_names  = [f"sc{i}" for i in range(1, 8)]
#lets make a df[pos_columns]

df["dt"] = df["Timeframe"] - df["Timeframe"].iloc[0]

for sc, init_pos in sc_init.items():
    df[f"{sc}_pos_x"] = init_pos[0] + vel_bulk[0] * df["dt"]
    df[f"{sc}_pos_y"] = init_pos[1] + vel_bulk[1] * df["dt"]
    df[f"{sc}_pos_z"] = init_pos[2] + vel_bulk[2] * df["dt"]


pos_cols = sum([[f"{sc}_pos_x", f"{sc}_pos_y", f"{sc}_pos_z"]
                for sc in sc_init.keys()], [])
B_cols   = sum([[f"{sc}_vg_B_x", f"{sc}_vg_B_y", f"{sc}_vg_B_z"]
                for sc in sc_init.keys()], [])

#######################
#Radial Basis Function#
#######################


def RBF_missing_data(missing_sc = None):
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
    epsilon = np.median(dists[:, 1])
    print(f"RBF epsilon (missing {missing_sc}) = {epsilon/1000:.3g} km")
    
    #RBF interpolation
    rbf = RBFInterpolator(
        centers_inc, values_inc,
        kernel="multiquadric",
        epsilon=epsilon,
        smoothing=0.0
    )
    """
    Outline: 
        -be able to select which spacecraft to drop
        then select the remaining SC from the data frame and calculate new RBF 
        NOTE: With new RBF have to be very careful how it effects the logic of all other
        fuctions, so implement with care!
        One way would be maybe that the function is always called but the data missing = None?
    """

    return rbf, included_pos_cols, included_B_cols

rbf, included_pos_cols, included_B_cols =  RBF_missing_data()

def sample_slice(coord1, coord2, const_coord, plane, nx, ny):
    if plane == "xy":
        X, Y = np.meshgrid(coord1, coord2)
        pts  = np.column_stack([X.ravel(), Y.ravel(),
                                np.full(X.size, const_coord)])
        Bxyz = rbf(pts)
        Bx, By, Bz = [Bxyz[:,i].reshape(ny,nx) for i in range(3)]
        return X, Y, Bx, By, Bz

    elif plane == "xz":
        X, Z = np.meshgrid(coord1, coord2)
        pts  = np.column_stack([X.ravel(),
                                np.full(X.size, const_coord),
                                Z.ravel()])
        Bxyz = rbf(pts)
        Bx, By, Bz = [Bxyz[:,i].reshape(ny,nx) for i in range(3)]
        return X, Z, Bx, Bz, By

    elif plane == "yz":
        Y, Z = np.meshgrid(coord1, coord2)
        pts  = np.column_stack([np.full(Y.size, const_coord),
                                Y.ravel(), Z.ravel()])
        Bxyz = rbf(pts)
        Bx, By, Bz = [Bxyz[:,i].reshape(ny,nx) for i in range(3)]
        return Y, Z, By, Bz, Bx


def plot_rbf_slices(time, nx=200, ny=200, L_Re=1.2,output_dir = None, output_file = None):
    """
    Plots stuff obviously duhhh
    maybe add more comments 
    """
    row = df[df["Timeframe"] == time].iloc[0]
    cluster = row[pos_cols].to_numpy().reshape(-1,3)
    bary    = cluster.mean(axis=0)

    L_m = L_Re * R_e
    xs = np.linspace(bary[0]-L_m, bary[0]+L_m, nx)
    ys = np.linspace(bary[1]-L_m, bary[1]+L_m, ny)
    zs = np.linspace(bary[2]-L_m, bary[2]+L_m, ny)

    XY = sample_slice(xs, ys, bary[2], "xy", nx, ny)
    XZ = sample_slice(xs, zs, bary[1], "xz", nx, ny)
    YZ = sample_slice(ys, zs, bary[0], "yz", nx, ny)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)
    for ax, (data, title) in zip(axs, zip([XY,XZ,YZ], ["X-Y","X-Z","Y-Z"])):
        C1, C2, U, V, W = data
        
        mag = np.hypot(U, V)
        cf  = ax.contourf(C1, C2, W, 30, cmap="coolwarm")
        ax.streamplot(C1, C2, U, V,
                      color=mag, cmap="magma", density=1.5, linewidth=0.5)
        if title == "X-Y":
            sc_u = cluster[:,0]; sc_v = cluster[:,1]
            out_comp = "Z"
        elif title == "X-Z":
            sc_u = cluster[:,0]; sc_v = cluster[:,2]
            out_comp = "Y"
        else:  
            sc_u = cluster[:,1]; sc_v = cluster[:,2]
            out_comp = "X"

        ax.scatter(sc_u, sc_v, c="k", s=20, label="SC")
        ax.legend(loc="upper right", fontsize="small")
        ax.margins(0)
        ax.set_aspect("equal")
        ax.set_title(f"{title}")
        ax.set_xlabel(f"{title[0]} (m)")
        ax.set_ylabel(f"{title[-1]} (m)")

        fig.colorbar(cf, ax=ax, orientation="vertical",
                     label=fr"$B_{{{out_comp}}}$")
    fig.suptitle(f"RBF reconstruction at Time = {row['Timeframe']:.1f} s")

    if output_dir == None:
        output_dir = "~/"

    if output_file == None:
        output_file = f"RBF_timeseries_reconstruction_{time}s.png"
    
    output_file = output_dir+output_file
            
    plt.savefig(output_file)
    plt.close()
    return


def Wasserstein_Hull(time, type = "filled", save = True, buffer = 0, error_cutoff = 20, info = True, output_dir =None, output_file =None):
    """
    Changes to be made: with time determine RBF sc locations, but
    have vlasiator stay in place and just change file time
    """
    from scipy.spatial import ConvexHull, Delaunay
    file = f"/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.000{time}.vlsv"
    if info:
        print(file)
    vlsvfile = pt.vlsvfile.VlsvReader(file)
    
    #Vlasitor convex hull from initial spacecraft locations
    init_pts = np.vstack(list(sc_init.values()))
    hull_init = ConvexHull(init_pts)
    dela_init = Delaunay(init_pts[hull_init.vertices])

    buf = buffer * R_e
    mins_i = init_pts.min(axis=0) - buf
    maxs_i = init_pts.max(axis=0) + buf
    nx = ny = nz = 60
    xs_i = np.linspace(mins_i[0], maxs_i[0], nx)
    ys_i = np.linspace(mins_i[1], maxs_i[1], ny)
    zs_i = np.linspace(mins_i[2], maxs_i[2], nz)
    X_i, Y_i, Z_i = np.meshgrid(xs_i, ys_i, zs_i, indexing="ij")

    pts_i = np.column_stack([X_i.ravel(), Y_i.ravel(), Z_i.ravel()])
    mask_i = dela_init.find_simplex(pts_i) >= 0
    B_vlas = vlsvfile.read_interpolated_variable("vg_b_vol", pts_i[mask_i])

    #RBF convex hull points from "moved" location
    row = df[df["Timeframe"] == time].iloc[0]
    dyn_pts = row[pos_cols].to_numpy().reshape(-1, 3)
    hull_dyn = ConvexHull(dyn_pts)
    dela_dyn = Delaunay(dyn_pts[hull_dyn.vertices])

    mins_d = dyn_pts.min(axis=0) - buf
    maxs_d = dyn_pts.max(axis=0) + buf
    xs_d = np.linspace(mins_d[0], maxs_d[0], nx)
    ys_d = np.linspace(mins_d[1], maxs_d[1], ny)
    zs_d = np.linspace(mins_d[2], maxs_d[2], nz)
    X_d, Y_d, Z_d = np.meshgrid(xs_d, ys_d, zs_d, indexing="ij")

    pts_d = np.column_stack([X_d.ravel(), Y_d.ravel(), Z_d.ravel()])
    mask_d = dela_dyn.find_simplex(pts_d) >= 0
    B_rbf = rbf(pts_d[mask_d])

    #Collect W_rels into array for plotting
    W_rels = []
    for comp in range(3):
        comp_vlas = B_vlas[:, comp]
        comp_rbf = B_rbf[:, comp]
        W1 = wasserstein_distance(comp_rbf, comp_vlas)
        med = np.median(comp_vlas)
        Wden = wasserstein_distance(comp_vlas, np.full_like(comp_vlas, med))
        W_rels.append(float(round(W1/Wden, 4)))

    if info:
        dB = np.linalg.norm(B_rbf-B_vlas,axis=1)
        B_vlas_mag = np.linalg.norm(B_vlas,axis=1)
        valid = np.isfinite(dB) & np.isfinite(B_vlas_mag) & (B_vlas_mag > 0)

        error_per = np.full_like(dB, np.nan)
        error_per[valid] = 100 * dB[valid] / B_vlas_mag[valid]

        fraction = np.count_nonzero(error_per[valid] < error_cutoff) / np.count_nonzero(valid)
        print(f"Fraction of points with <{error_cutoff}%: {fraction:.3f}")
        
    if save:
        labels = [r"$B_x$", r"$B_y$", r"$B_z$"]
        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        for ax, lbl, vlas_comp, rbf_comp, W in zip(axes, labels, 
                                        B_vlas.T, B_rbf.T, W_rels):
            lo, hi = np.percentile(np.concatenate((vlas_comp, rbf_comp)), [0.5, 99.5])
            bins = np.linspace(lo, hi, 41)
            if type == "filled":
                ax.hist(vlas_comp, bins=bins, alpha=0.5, label="Vlasiator")
                ax.hist(rbf_comp, bins=bins, alpha=0.5, label="RBF")
            else:
                ax.hist(vlas_comp, bins=bins, histtype="step", label="Vlasiator")
                ax.hist(rbf_comp, bins=bins, histtype="step", label="RBF")
            ax.axvline(np.median(vlas_comp), ls="--", color="k")
            ax.set_title(f"$W_{{rel}}$={W}")
            ax.set_xlabel(f"{lbl}")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.suptitle(f"Component distributions  t={time}s")


        if output_dir == None:
            output_dir = "~/"

        if output_file == None:
            output_file = f"Wassertein_hull_{type}_{time}s.png"
        output_file = output_dir+output_file
        
        
        plt.savefig(output_file)
        plt.close(fig)

    return W_rels, round(fraction,3)


def plot_Wass_time(save =True, error_cutoff = 20, output_dir = None, output_file = None):

    """
    Convex hull at time index and calculate Wasserstein distance from that. 
    Calculate Wasserstein distance from hull also in Vlasiator timestep 
    Plot W_rel/time
    for time in times:
        read Vlasitor
        get points in hull for vlasitor and RBF
        Calculate W_rels
    Plot
    """
    times = df["Timeframe"]
    W_x,W_y,W_z,error = [], [], [], []
    for t in times:
        data, error_frac  = Wasserstein_Hull(t, save = False,error_cutoff=error_cutoff)
        W_x.append(data[0])
        W_y.append(data[1])
        W_z.append(data[2])
        error.append(error_frac)
    if save:
        fig, ax = plt.subplots(1, 2, figsize=(12,5))
    
        ax[0].plot(times, W_x, label=r'$W_x$')
        ax[0].plot(times, W_y, label=r'$W_y$')
        ax[0].plot(times, W_z, label=r'$W_z$')

        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel(r"Relative Wasserstein Distance $W_{\mathrm{rel}}$")
        ax[0].set_title("Wasserstein Distance vs Time")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()
        
        ax[1].plot(times, error)
        
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel(r"Point-wise error")
        ax[1].set_title(f"Fraction of points with error <{error_cutoff}%")
        ax[1].grid(True, alpha=0.3)
        fig.suptitle(f"Bulk velocity: ({vg_v_x},{vg_v_y},{vg_v_z}) m/s")
        fig.tight_layout()

        if output_dir == None:
            output_dir = "~/"

        if output_file == None:
            output_file = f"Wasserstein_vs_Time+error_abs_bulk={np.sqrt(vg_v_x**2+vg_v_y**2+vg_v_z**2)}.png"
        
        output_file = output_dir+output_file
        
        plt.savefig(output_file)
    return 


#RUN
#for i in range(T):
#    plot_rbf_slices(t_idx= i)
#Wasserstein_Hull(time = 1340, save = False)
plot_Wass_time(output_dir=output_dir,output_file="Wasserstein_vs_Time.png")