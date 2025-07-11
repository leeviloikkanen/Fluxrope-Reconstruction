#This script was created by Leevi Loikkanen to do error analysis between Vlasiator simulation files 
#and reconstructed magnetic fields from virtual spacecraft data using Radial Basis Functions  
#by means of point-wise error and Wasserstein distances 
#RBF method and error analysis methods used here outlined in paper: 
#https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2023EA003369

#NOTE: SOME FUNCTION HAVE NOT PROPERLY BEEN UPDATED SO THERE MIGHT BE SOME ERRORS (namely plot_hist_component_comparison() needs index map)
#TODO: modify functions to have directly some folder_dir+identifier to automatically change file names and locations at start 

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
import pytools as pt
import matplotlib as mpl

#Shared info
df = pd.read_csv("/home/leeviloi/plas_obs_vg_b_full_1352_tail_through+pos_z=-0.5_inner_scale=0.14.csv")
#CHECK TIME
t = 1352
#files 
file = f"/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.000{t}.vlsv"
vlsvfile = pt.vlsvfile.VlsvReader(file)

R_e = 6371000   
R_e_km  = 6371.0
#Size of area to consider
Lsize = 1.2
L_vlas = Lsize * R_e    
L_rbf = Lsize * R_e_km  
#position of examination and resolution
pos_idx = 90        
nx, ny  = 200, 200    

times = df["Position_Index"].to_numpy() 
T = len(times)
sc_names  = [f"sc{i}" for i in range(1, 8)]
pos_cols  = sum([[f"{sc}_pos_x", f"{sc}_pos_y", f"{sc}_pos_z"] for sc in sc_names], [])
B_cols    = sum([[f"{sc}_vg_B_x", f"{sc}_vg_B_y", f"{sc}_vg_B_z"] for sc in sc_names], [])

all_points = df[pos_cols].to_numpy().reshape(T * 7, 3) 

start_idx = 0
end_idx   = 100 
selected_points = all_points[start_idx*7:end_idx*7]

#######################
#Radial Basis Function#
#######################

centers = (df[pos_cols].to_numpy().reshape(T * 7, 3) / 1000.0)   # km
values  =  df[B_cols].to_numpy().reshape(T * 7, 3)   

def RBF_missing_data(missing_sc = None):
    #Modify to select only sc that aren't in missing_sc then just same things as below: 

    if missing_sc is None:
        included_sc = sc_names
    else:
        included_sc = [sc for sc in sc_names if sc not in missing_sc]

    included_pos_cols = sum([[f"{sc}_pos_x", f"{sc}_pos_y", f"{sc}_pos_z"] for sc in included_sc], [])
    included_B_cols = sum([[f"{sc}_vg_B_x", f"{sc}_vg_B_y", f"{sc}_vg_B_z"] for sc in included_sc], [])

    centers_inc = df[included_pos_cols].to_numpy().reshape(-1, 3) / 1000.0  
    values_inc = df[included_B_cols].to_numpy().reshape(-1, 3)
    #pick epsilon
    "https://www.math.iit.edu/~fass/Dolomites.pdf?" #nearest neighbor method mentioned
    nbrs = NearestNeighbors(n_neighbors=2).fit(centers_inc)
    dists, _ = nbrs.kneighbors(centers_inc)
    epsilon = np.median(dists[:, 1])
    print(f"RBF epsilon (missing {missing_sc}) = {epsilon:.3g} km")
    
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

#outer tetra = ["sc2","sc3","sc4"]
#inner tetra = ["sc5","sc6","sc7"]
missing_sc =None
rbf, included_pos_cols, included_B_cols = RBF_missing_data(missing_sc)
"""
#pick epsilon
"https://www.math.iit.edu/~fass/Dolomites.pdf?" #nearest neighbor method mentioned
nbrs = NearestNeighbors(n_neighbors=2).fit(centers)
dists, _ = nbrs.kneighbors(centers)
epsilon = np.median(dists[:,1])

print(f"RBF epsilon = {epsilon:.3g} km")

#RBF interpolation
rbf = RBFInterpolator(
    centers, values,
    kernel="multiquadric",
    epsilon=epsilon,
    smoothing=0.0
)
"""

row = df.loc[df["Position_Index"] == pos_idx].iloc[0]
cluster_now = row[pos_cols].to_numpy().reshape(7, 3) / 1000.0     
bary_rbf = cluster_now.mean(axis=0)     

def sample_slice(coord1, coord2, const_coord, plane):
    """
    main thing to note about this function is that the output
    order of coordinates is dependant on chosen plane
    ex. yz plane will output coordinates as Y, Z, By, Bz, Bx
    Out of plane component will always be last
    """
    if plane == "xy":
        X, Y = np.meshgrid(coord1, coord2)                           
        pts  = np.column_stack([X.ravel(), Y.ravel(),
                                np.full(X.size, const_coord)])
        Bxyz = rbf(pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return X, Y, Bx, By, Bz                                    
    elif plane == "xz":
        X, Z = np.meshgrid(coord1, coord2)                           
        pts  = np.column_stack([X.ravel(),
                                np.full(X.size, const_coord),
                                Z.ravel()])
        Bxyz = rbf(pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return X, Z, Bx, Bz, By                                    
    elif plane == "yz":
        Y, Z = np.meshgrid(coord1, coord2)                           
        pts  = np.column_stack([np.full(Y.size, const_coord),
                                Y.ravel(), Z.ravel()])
        Bxyz = rbf(pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return Y, Z, By, Bz, Bx                                   
    

x = np.linspace(bary_rbf[0]-L_rbf, bary_rbf[0]+L_rbf, nx)   
y = np.linspace(bary_rbf[1]-L_rbf, bary_rbf[1]+L_rbf, ny)
z = np.linspace(bary_rbf[2]-L_rbf, bary_rbf[2]+L_rbf, ny)
#This is in Kilometers
XY_RBF = sample_slice(x, y, bary_rbf[2], "xy")       
XZ_RBF = sample_slice(x, z, bary_rbf[1], "xz")       
YZ_RBF = sample_slice(y, z, bary_rbf[0], "yz")   

#Vlasiator

#location of spacecrafts at desired index
sc_names = [f"sc{i}" for i in range(1, 8)]
pos_cols = sum([[f"{sc}_pos_x", f"{sc}_pos_y", f"{sc}_pos_z"] for sc in sc_names], [])
row     = df.loc[df["Position_Index"] == pos_idx].iloc[0]
points  = row[pos_cols].to_numpy().reshape(7, 3)
points_incl = row[included_pos_cols].to_numpy().reshape(int(len(included_pos_cols)/3), 3) 
bary_vlas    = points.mean(axis=0)                              

                             
x = np.linspace(bary_vlas[0]-L_vlas, bary_vlas[0]+L_vlas, nx)
y = np.linspace(bary_vlas[1]-L_vlas, bary_vlas[1]+L_vlas, ny)
z = np.linspace(bary_vlas[2]-L_vlas, bary_vlas[2]+L_vlas, ny)


#function to extract meshgrids of simulation data in different planes for reference 
def sample_slice_vlas(coord1, coord2, const_coord, plane):
    """
    main thing to note about this function is that the output
    order of coordinates is dependant on chosen plane
    ex. yz plane will output coordinates as Y, Z, By, Bz, Bx
    Out of plane component will always be last
    """
    if plane == "xy":
        X, Y = np.meshgrid(coord1, coord2)                 
        pts  = np.column_stack([X.ravel(), Y.ravel(),
                                np.full(X.size, const_coord)])
        Bxyz = vlsvfile.read_interpolated_variable("vg_b_vol", pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return X, Y, Bx, By, Bz                            
    elif plane == "xz":
        X, Z = np.meshgrid(coord1, coord2)                 
        pts  = np.column_stack([X.ravel(),
                                np.full(X.size, const_coord),
                                Z.ravel()])
        Bxyz = vlsvfile.read_interpolated_variable("vg_b_vol", pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return X, Z, Bx, Bz, By                            
    elif plane == "yz":
        Y, Z = np.meshgrid(coord1, coord2)                
        pts  = np.column_stack([np.full(Y.size, const_coord),
                                Y.ravel(), Z.ravel()])
        Bxyz = vlsvfile.read_interpolated_variable("vg_b_vol", pts)
        Bx, By, Bz = (Bxyz[:, i].reshape(nx, ny) for i in range(3))
        return Y, Z, By, Bz, Bx                           
    

#This is in meters
XY_Vlas = sample_slice_vlas(x, y, bary_vlas[2], "xy")     
XZ_Vlas = sample_slice_vlas(x, z, bary_vlas[1], "xz")     
YZ_Vlas = sample_slice_vlas(y, z, bary_vlas[0], "yz")

######################
#STATISTICAL ANALYSIS# 
######################

#Wasserstein Distance per component for selected plane 

def plane_wasserstein(rbf_plane, vlas_plane, plane_name):
    """
    Calculates the 1D Wasserstein distance between
    RBF and Vlasiator B components in a given plane
    """
    idx_map = {
    "xy": (2, 3, 4),  
    "xz": (2, 4, 3),   
    "yz": (4, 2, 3),  
    }
    id_x, id_y, id_z = idx_map[plane_name]

    # reorder each into (Bx, By, Bz)
    Br = [rbf_plane[idx] for idx in (id_x, id_y, id_z)]
    Bs = [vlas_plane[idx] for idx in (id_x, id_y, id_z)]
    rel_wass = {}
    for comp, B_RBF, B_vlas in zip("xyz", Br, Bs):

        B_RBF_flat, B_vlas_flat = B_RBF.ravel(), B_vlas.ravel()

        W1   = wasserstein_distance(B_RBF_flat, B_vlas_flat)

        med = np.median(B_vlas_flat)
        #normalizing wasserstein distance by comparison to single valued 
        W_den = wasserstein_distance(B_vlas_flat, np.full_like(B_vlas_flat,med))

        relW = W1 /W_den                             
        rel_wass[f"W1_rel_{comp}"] = relW

    rel_wass["plane"] = plane_name
    return rel_wass

#This could/should be moved into one of the plotting functions. No need to calculate it everytime 
results = []
results.append(plane_wasserstein(XY_RBF, XY_Vlas, "xy"))
results.append(plane_wasserstein(XZ_RBF, XZ_Vlas, "xz"))
results.append(plane_wasserstein(YZ_RBF, YZ_Vlas, "yz"))

df_W = pd.DataFrame(results).set_index("plane")
"""
print(f"Wasserstein values at position {pos_idx}")
print(df_W)
"""

vlas_planes = [XY_Vlas,XZ_Vlas,YZ_Vlas]
RBF_planes = [XY_RBF,XZ_RBF,YZ_RBF]  

def error_perscentage(B_plane_RBF, B_plane_vlas,rel_error =True):
    """
    The logic/naming here is wrong since depending on the plane the 
    order is not necessarily x,y,z but since the components match each other 
    and the absolute is only thing that matters, it works
    """
    Pr, Qr, Bxr, Byr, Bzr = B_plane_RBF
    Pv, Qv, Bxv, Byv, Bzv = B_plane_vlas     
    if not np.allclose(Pr,Pv/1e3) and np.allclose(Qr,Qv/1e3):
        raise ValueError("fields dont match")
    dBx = Bxr - Bxv
    dBy = Byr - Byv
    dBz = Bzr - Bzv

    dB_mag = np.sqrt(dBx**2 + dBy**2 + dBz**2)   
    Bv_mag = np.sqrt(Bxv**2+Byv**2+Bzv**2)
    if rel_error:
        error_per = 100*dB_mag/Bv_mag
    else: 
        error_per = dB_mag
    return Pr, Qr, error_per #, Bv_mag
"""
TEST PLOT to compare how the absolute error is distributed in the planes
Xr, Yr, error_per, Bv_mag = error_perscentage(XY_RBF, XY_Vlas, rel_error=False)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

contour = axs[0].contourf(Xr / 1e3, Yr / 1e3, error_per, 40, cmap="viridis")
contour1 = axs[1].contourf(Xr / 1e3, Yr / 1e3, Bv_mag, 40, cmap="viridis")

fig.colorbar(contour, ax=axs[0], label="ABS error")
fig.colorbar(contour1, ax=axs[1], label="ABS B vlas")

for ax in axs:
    ax.set_aspect("equal")
    ax.set_xlabel("X [10³ km]")
    ax.set_ylabel("Y [10³ km]")
    ax.margins(0)

plt.tight_layout()
plt.savefig("/home/leeviloi/scripts_useful/error_test_xy_abs_L=1.2_pos=20_2.png")
plt.show()
"""

##########
#PLOTTING#
##########
def plot_point_wise_error(save = True, points = points):
    """
    Plots all 3 planes of point-wise errors between vlasiator and RBF 
    at a simulation position.
    TODO: have input be the planes instead of using specificied planes already
    NOTE full_rbf_vlas_comp more up-to-date use of same function
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    err_xy =  error_perscentage(XY_RBF,XY_Vlas)
    err_xz = error_perscentage(XZ_RBF,XZ_Vlas)
    err_yz =error_perscentage(YZ_RBF,YZ_Vlas)
    panels = [
        ("X-Y  (z = {:.0f} km)".format(bary_vlas[2]/1e3), err_xy , ("X","Y")),
        ("X-Z  (y = {:.0f} km)".format(bary_vlas[1]/1e3), err_xz, ("X","Z")),
        ("Y-Z  (x = {:.0f} km)".format(bary_vlas[0]/1e3), err_yz, ("Y","Z")),
    ]
    #err_min = min(data[-1].min() for data in (err_xy, err_xz, err_yz))
    #err_max = max(data[-1].max() for data in (err_xy, err_xz, err_yz))

    err_vmin, err_vmax = 0.0, 50.0                     
    levels   = np.linspace(err_vmin, err_vmax, 31)     
    norm     = mpl.colors.Normalize(vmin=err_vmin, vmax=err_vmax)

    for ax, (title, (P,Q,error), (lab1,lab2)) in zip(axes, panels):
        

        cf = ax.contourf(P/1e3, Q/1e3, error, levels = levels, cmap="viridis", norm = norm, extend = "max")
        #cf = ax.contourf(P/1e3, Q/1e3, error, 30, cmap="viridis")
        if lab1 == "X":             u = points[:,0]/1e6
        elif lab1 == "Y":           u = points[:,1]/1e6
        else:                       u = points[:,2]/1e6   

        if lab2 == "Y":             v = points[:,1]/1e6
        elif lab2 == "Z":           v = points[:,2]/1e6
        else:                       v = points[:,0]/1e6   
        ax.margins(0) 
        ax.scatter(u, v, c="k", s=40, label="spacecraft")
        ax.set_xlabel(f"{lab1}  (10³ km)")
        ax.set_ylabel(f"{lab2}  (10³ km)")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.legend(loc="upper right",fontsize="small")

    sm = mpl.cm.ScalarMappable(cmap="viridis",
                            norm=norm)

    #sm = mpl.cm.ScalarMappable(cmap="viridis",
    #                           norm=mpl.colors.Normalize(vmin=err_min, vmax=err_max))
    fig.colorbar(sm, ax=axes.ravel().tolist(),
                orientation="vertical", label="Error %")
    fig.suptitle(f"Relative error at Position = {pos_idx} at {t}s", fontsize=16)

    if save: 
        plt.savefig(f"/home/leeviloi/fluxrope_thesis/error_threeslice_L=1.2_pos={pos_idx}_1432s.png")
    return

def plot_hist_component_comparison(plane_RBF, plane_Vlas, plane, type = "filled"):
    """
    Function plotting histograms of components of RBF and Vlasiator on a give plane
    and also the plane's Wasserstein distance for each component
    TODO Fix indexing issue between plane output and component decomposition 
    """
    n_bins = 40
    _, _, Bx_RBF, By_RBF, Bz_RBF = plane_RBF
    _, _, Bx_vlas,By_vlas,Bz_vlas = plane_Vlas
    B_RBF = [Bx_RBF.ravel(), By_RBF.ravel(),Bz_RBF.ravel()]
    B_vlas =[Bx_vlas.ravel(),By_vlas.ravel(),Bz_vlas.ravel()]
    
    fig, axes = plt.subplots(1,3, figsize=(12, 3.5))
    labels = ["$B_x$","$B_y$","$B_z$"]
    axis = ["x","y","z"]
    for ax, B_comp_RBF, B_comp_vlas, label,axe  in zip(axes,B_RBF,B_vlas, labels, axis):
        #print(B_comp_vlas.min())
        lo, hi = np.percentile(np.concatenate([B_comp_RBF, B_comp_vlas]), [0.5, 99.5])
        bins   = np.linspace(lo, hi, n_bins+1)
        
        #Density = true means that the histogram is normalize in to a probability function --> integral = 1 
        if type == "filled":
            ax.hist(B_comp_vlas, bins=bins, histtype="stepfilled",
                alpha = 0.5, label="Vlasiator")
            ax.hist(B_comp_RBF, bins=bins, histtype="stepfilled",
                alpha = 0.5, label="RBF",)
        else: 
            ax.hist(B_comp_vlas, bins=bins, histtype="step",
                lw=1.8, label="Vlasiator")
            ax.hist(B_comp_RBF, bins=bins, histtype="step",
                lw=1.2, label="RBF",)
        comp_med = np.median(B_comp_vlas)

        ax.axvline(comp_med, ls = "--", label = r"$\tilde{B}_{\text{vlas}}$")
        ax.set_xlabel(f"{label}")
        W_val = np.round(df_W.loc[plane, f"W1_rel_{axe}"], 3)
        ax.set_title(rf"$W_{{\mathrm{{rel}}}} = {W_val}$")
        ax.grid(alpha=0.3)
        axes[0].legend(loc="upper right")
        fig.tight_layout()
        fig.suptitle(f"Histograms of component counts at Pos={pos_idx}, plane = {plane}")
        plt.savefig(f"/home/leeviloi/fluxrope_thesis/histogram_comparision_comp_counts_{plane}_pos_{pos_idx}_no_den_type={type}.png")
    return


def plot_any_plane(norm_vec):
    """
    outline:
      -gather planes along normal to provided vector
      -calculate error using error percentage function (any plane should work)
      -plot (1,3) of streamlines and error
    """
    n = np.asarray(norm_vec)
    n = n/np.linalg.norm(n)
    #Logic here is to choose a arbitrary helper vector to calculate crossproduct with
    #then calculate crossproduct with new vector (u) and norm to get another vector (v) also
    #perpendicular to n so u and v form the plane perpendicular to n 
    helper = np.array([1,0,0] if abs(n[0]<0.9) else np.array([0,1,0]))
    u = np.cross(n,helper)
    u = u/np.linalg.norm(u)

    return

def plot_vlas_RBF_error(vlas_planes, rbf_planes, save = True, rel_error = True, points = points):
    """
    Creates a 3x3 plot of countours  (First row Vlasiator xy, xz and yz planes with streamlines,
    Second row RBF xy, xz, yz planes with streamliens, Third row point-wise error comparison of 
    the magnetic field strenght of the first two rows)  
    """
    err_xy =  error_perscentage(rbf_planes[0],vlas_planes[0], rel_error=rel_error)
    err_xz = error_perscentage(rbf_planes[1],vlas_planes[1],rel_error=rel_error)
    err_yz =error_perscentage(rbf_planes[2],vlas_planes[2], rel_error=rel_error)
    fig, axes = plt.subplots(3,3,figsize = (13,11), constrained_layout=True)
    fig.dpi = 500
    panels = [
    ("X-Y  (z = {:.0f} km)".format(bary_vlas[2]/1e3),("X","Y","z"),err_xy[-1]),
    ("X-Z  (y = {:.0f} km)".format(bary_vlas[1]/1e3),("X","Z","y"),err_xz[-1]),
    ("Y-Z  (x = {:.0f} km)".format(bary_vlas[0]/1e3),("Y","Z","x"),err_yz[-1]),
    ]
    if rel_error:
        err_vmin, err_vmax = 0.0, 50.0                     
        levels   = np.linspace(err_vmin, err_vmax, 31)     
        norm     = mpl.colors.Normalize(vmin=err_vmin, vmax=err_vmax)
        error_lbl = "Error (%)"
        error_title = "point-wise error (%)"
    else:

        err_vmin, err_vmax = 0.0, 1.5e-8
        levels   = np.linspace(err_vmin, err_vmax, 31)
        norm     = mpl.colors.Normalize(vmin=err_vmin, vmax=err_vmax)
        error_lbl = "|ΔB|"  
        error_title = "Absolute point-wise error"          

    clus_size = 20

    for i, (vlas_plane,rbf_plane, panel) in enumerate(zip(vlas_planes,rbf_planes,panels)):

        title, (lab1, lab2, lab3), error = panel
        Pr, Qr = rbf_plane[0]/1e3, rbf_plane[1]/1e3
    
        #
        #Vlasiator Plotting
        #
        cont_0 = axes[0,i].contourf(Pr,Qr,vlas_plane[-1], 30, cmap="coolwarm")
        speed = np.hypot(vlas_plane[2], vlas_plane[3])
        axes[0,i].streamplot(Pr, Qr, vlas_plane[2], vlas_plane[3],
                    color=speed, cmap="magma", density=2, linewidth = 0.4)
        
        if lab1 == "X":             u = points[:,0]/1e6
        elif lab1 == "Y":           u = points[:,1]/1e6
        else:                       u = points[:,2]/1e6   

        if lab2 == "Y":             v = points[:,1]/1e6
        elif lab2 == "Z":           v = points[:,2]/1e6
        else:                       v = points[:,0]/1e6   
        
        cbar = fig.colorbar(cont_0, ax=axes[0,i], orientation="vertical", shrink = 0.8)
        cbar.set_label(f"$B_{lab3}$")
        axes[0,i].scatter(u, v, c="k", s=clus_size, label="spacecraft")
        axes[0,i].margins(0)
        #axes[0,i].set_xlabel(f"{lab1}  (10³ km)")
        axes[0,i].set_ylabel(f"{lab2}  (10³ km)")
        axes[0,i].set_aspect("equal")
        axes[0,0].legend(loc="upper right",fontsize="small")
        axes[0,i].set_title(title)
        #axes[0,i].legend(loc= "upper right",fontsize="small")
        #
        #RBF plotting
        #
        cont_1 = axes[1,i].contourf(Pr,Qr,rbf_plane[-1], 30, cmap="coolwarm")
        speed = np.hypot(rbf_plane[2], rbf_plane[3])
        axes[1,i].streamplot(Pr, Qr, rbf_plane[2], rbf_plane[3],
                    color=speed, cmap="magma", density=2, linewidth = 0.4)
        
        cbar = fig.colorbar(cont_1, ax=axes[1,i], orientation="vertical", shrink = 0.8)
        cbar.set_label(f"$B_{lab3}$")
        axes[1,i].scatter(u, v, c="k", s=clus_size, label="spacecraft")
        axes[1,i].margins(0)
        #axes[1,i].set_xlabel(f"{lab1}  (10³ km)")
        axes[1,i].set_ylabel(f"{lab2}  (10³ km)")
        axes[1,i].set_aspect("equal")
        axes[1,0].legend(loc="upper right",fontsize="small")
        #axes[1,i].set_title(title)
        #
        #POINT-WISE ERROR
        #
        cf = axes[2,i].contourf(Pr, Qr, error, levels = levels, cmap="viridis", norm = norm, extend = "max")
      
        axes[2,i].margins(0) 
        axes[2,i].scatter(u, v, c="k", s=clus_size, label="spacecraft")
        axes[2,i].set_xlabel(f"{lab1}  (10³ km)")
        axes[2,i].set_ylabel(f"{lab2}  (10³ km)")
        axes[2,i].set_aspect("equal")
        #axes[2,i].set_title(title)
        axes[2,0].legend(loc="upper right",fontsize="small")
    
    sm = mpl.cm.ScalarMappable(cmap="viridis",
                            norm=norm)

    fig.colorbar(sm, ax=axes[2,:].ravel().tolist(),
                orientation="vertical", label=error_lbl, shrink = 0.8)
    #Label each row
    row_y = [0.96, 0.64, 0.32]  

    for y, txt in zip(row_y,
                    ["Vlasiator",
                    "RBF",
                    error_title]):
        fig.text(0.5, y, txt, ha="center", va="center", fontsize=20)

    #fig.tight_layout()
    fig.suptitle(f"Comparison of Vlasiator and RBF reconstruction at Pos={pos_idx}, time = {t}", fontsize = 20)
    if save:
        plt.savefig(f"/home/leeviloi/fluxrope_thesis/fly_through_tail/full_vlas_rbf_comp_pos_{pos_idx}_time={t}_L={Lsize}_GOOD.png")
    return


def full_Wasser_hist(vlas_planes, rbf_planes, type = "filled", save = True):
    """
    For a collections of xy, xz, yz planes from vlasiator and the RBF reconstruction
    at a certain position index, this function creates a 3x3 plot of histograms showing
    the comparisons of component counts in said planes. 
    """
    fig, axes = plt.subplots(3,3, figsize = (12,10), constrained_layout = True)
    n_bins = 40
    fig.dpi = 150
    #index map since sample_plane and sample_vlas_plane output form
    idx_map = {"xy": (2, 3, 4),
            "xz": (2, 4, 3),
            "yz": (4, 2, 3)}

    planes = ["xy","xz","yz"]
    labels = ["$B_x$","$B_y$","$B_z$"]
    axis = ["x","y","z"]
    for i, (plane_vlas, plane_rbf, plane) in enumerate(zip(vlas_planes,rbf_planes, planes)):
        sel = idx_map[plane]               
        B_RBF = [plane_rbf[idx].ravel()  for idx in sel]  
        B_vlas = [plane_vlas[idx].ravel() for idx in sel]  

        for j, (B_comp_RBF, B_comp_vlas, label,axe)  in enumerate(zip(B_RBF,B_vlas, labels, axis)):
            #print(B_comp_vlas.min())
            ax = axes[i,j]
            lo, hi = np.percentile(np.concatenate([B_comp_RBF, B_comp_vlas]), [0.5, 99.5])
            bins   = np.linspace(lo, hi, n_bins+1)
            
            #Density = true means that the histogram is normalize in to a probability function --> integral = 1 
            if type == "filled":
                ax.hist(B_comp_vlas, bins=bins, histtype="stepfilled",
                    alpha = 0.5, label="Vlasiator")
                ax.hist(B_comp_RBF, bins=bins, histtype="stepfilled",
                    alpha = 0.5, label="RBF",)
            else: 
                ax.hist(B_comp_vlas, bins=bins, histtype="step",
                    lw=1.8, label="Vlasiator")
                ax.hist(B_comp_RBF, bins=bins, histtype="step",
                    lw=1.2, label="RBF",)
            comp_med = np.median(B_comp_RBF)

            ax.axvline(comp_med, ls = "--", label = r"$\tilde{B}_{\text{vlas}}$")
            if i == 2:
                ax.set_xlabel(f"{label}")
            W_val = np.round(df_W.loc[plane, f"W1_rel_{axe}"], 3)
            ax.set_title(rf"$W_{{\mathrm{{rel}}}} = {W_val}$")

            if j == 0:
                ax.set_ylabel(f"${plane}$ plane")

            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize="small")
    fig.suptitle(f"Histograms of component comparisons at pos {pos_idx}, time = 1432s, L={Lsize}")
    if save:
        plt.savefig(f"/home/leeviloi/fluxrope_thesis/histogram_3x3_L={Lsize}_time=1432s_pos={pos_idx}_CORRECT.png")
    return


def Wasser_3D_hist(sc_points, type = "filled", save = True, path=None, pos_idx = pos_idx, buffer = 0, error_cutoff = 20, info = True):
    """
    This function creates a three histogram plots of the values of the components in 
    RBF reconstruction and Vlasiator data inside a convex hull made of the virtual 
    spacecraft constellation. The function also prints the fraction of points that are
    below a relative error cutoff.

    Return: (Wasser_x, Wasser_y, Wasser_z) 
    """
    if path == None:
        path = f"/home/leeviloi/fluxrope_thesis/fly_through_z=-1_inner=0.14/histogram_comparison_comp_counts_type={type}_3D_pos_{pos_idx}.png"
    nx, ny, nz = 60, 60, 60
    lil = buffer*R_e
    x = np.linspace(sc_points[:,0].min()-lil,sc_points[:,0].max()+lil,nx)
    y = np.linspace(sc_points[:,1].min()-lil,sc_points[:,1].max()+lil,ny)
    z = np.linspace(sc_points[:,2].min()-lil,sc_points[:,2].max()+lil,nz)
    W_rels = []
    X, Y, Z = np.meshgrid(x,y,z, indexing="ij")
    
    pts = np.column_stack([X.ravel(),Y.ravel(),Z.ravel()])

    from scipy.spatial import ConvexHull, Delaunay
    hull = ConvexHull(sc_points)
    trig = Delaunay(sc_points[hull.vertices])
    inside = trig.find_simplex(pts)>=0
    hull_idx = inside.reshape(nx,ny,nz)

    pts = pts[inside]
    #Keep the index for later
  
    Bxyz_vlas = vlsvfile.read_interpolated_variable("vg_b_vol", pts)
    
    Bx_vlas = np.zeros((nx, ny, nz)) * np.nan
    By_vlas = np.full_like(Bx_vlas, np.nan)
    Bz_vlas = np.full_like(Bx_vlas, np.nan)

    #Later
    Bx_vlas[hull_idx] = Bxyz_vlas[:, 0]
    By_vlas[hull_idx] = Bxyz_vlas[:, 1]
    Bz_vlas[hull_idx] = Bxyz_vlas[:, 2]
    Bxyz_RBF = rbf(pts/1000)
    Bx_RBF = np.zeros((nx, ny, nz)) * np.nan
    By_RBF  = np.full_like(Bx_vlas, np.nan)
    Bz_RBF  = np.full_like(Bx_vlas, np.nan)
    #Later
    Bx_RBF[hull_idx] = Bxyz_RBF[:, 0]
    By_RBF[hull_idx] = Bxyz_RBF[:, 1]
    Bz_RBF[hull_idx] = Bxyz_RBF[:, 2]
    
    B_RBF = [Bx_RBF.ravel(), By_RBF.ravel(),Bz_RBF.ravel()]
    B_vlas =[Bx_vlas.ravel(),By_vlas.ravel(),Bz_vlas.ravel()]
    
    #Point-wise error inside the Convex Hull  
    #USED also to check validity of extrapolation_limit() SDF
    dBx = B_RBF[0] - B_vlas[0]
    dBy = B_RBF[1] - B_vlas[1]
    dBz = B_RBF[2] - B_vlas[2]

    dB_mag = np.sqrt(dBx**2 + dBy**2 + dBz**2)   
    Bv_mag = np.sqrt(B_vlas[0]**2+B_vlas[1]**2+B_vlas[2]**2)
    #Get rid of Nans
    valid_mask = np.isfinite(dB_mag) & np.isfinite(Bv_mag) & (Bv_mag > 0)
    
    error_per = np.full_like(Bv_mag, np.nan)
 
    error_per[valid_mask] = 100 * dB_mag[valid_mask] / Bv_mag[valid_mask]
    good_mask = error_per < error_cutoff

    error_num = error_per[good_mask]

    fraction = len(error_num) / np.count_nonzero(valid_mask)
    if info: 
        print(f"Fraction of points with error <{error_cutoff}%: {fraction:.3f}")
    

    #Plotting
    fig, axes = plt.subplots(1,3, figsize=(12, 3.5))
    labels = ["$B_x$","$B_y$","$B_z$"]
    axis = ["x","y","z"]
    n_bins = 40
    for ax, B_comp_RBF, B_comp_vlas, label,axe  in zip(axes,B_RBF,B_vlas, labels, axis):
        mask = np.isfinite(B_comp_RBF) & np.isfinite(B_comp_vlas)
        B_comp_RBF = B_comp_RBF[mask]
        B_comp_vlas = B_comp_vlas[mask]
        #print(B_comp_vlas.min())
        lo, hi = np.percentile(np.concatenate([B_comp_RBF, B_comp_vlas]), [0.5, 99.5])
        bins   = np.linspace(lo, hi, n_bins+1)
        
        B_RBF_flat, B_vlas_flat = B_comp_RBF.ravel(), B_comp_vlas.ravel()

        W1   = wasserstein_distance(B_RBF_flat, B_vlas_flat)

        comp_med = np.median(B_vlas_flat)
        #normalizing wasserstein distance by comparison to single valued 
        W_den = wasserstein_distance(B_vlas_flat, np.full_like(B_vlas_flat,comp_med))

        relW = np.round(W1 /W_den,4)       
        W_rels.append(relW)
        #Density = true means that the histogram is normalize in to a probability function --> integral = 1 
        if type == "filled":
            ax.hist(B_comp_vlas, bins=bins, histtype="stepfilled",
                alpha = 0.5, label="Vlasiator")
            ax.hist(B_comp_RBF, bins=bins, histtype="stepfilled",
                alpha = 0.5, label="RBF",)
        else: 
            ax.hist(B_comp_vlas, bins=bins, histtype="step",
                lw=1.8, label="Vlasiator")
            ax.hist(B_comp_RBF, bins=bins, histtype="step",
                lw=1.2, label="RBF",)
        

        ax.axvline(comp_med, ls = "--", label = r"$\tilde{B}_{\text{vlas}}$")
        ax.set_xlabel(f"{label}")
        ax.set_title(rf"$W_{{\mathrm{{rel}}}} = {relW}$")
        ax.grid(alpha=0.3)
        axes[0].legend(loc="upper left")
    fig.tight_layout(rect=[0, 0, 1, 0.93]) 
    fig.suptitle(f"Histograms of component counts at Pos={pos_idx}, Convex Hull")

    if save:
        plt.savefig(path)
    plt.close()
    return tuple(W_rels)


def extrapolation_limit(sc_points,Dis_min = 0, Dis_max = 0.5, inner = False, error_cutoff = 50):
    import trimesh
    from scipy.spatial import ConvexHull, Delaunay
    """
    Outline 
       -method to calculate Wasserstein distance + average pointwise error for given distance from the convex hull
           -use idea from Wasser_3D_hist for 3D extraction of data
           -one method for point-wise error could be % of points >20% error
           -distance from convex hull? 
           -signed distance function could be helpful
           -scale error significance with B strenght? since close by largest error due to magnetopause location 

       -vary distance starting from larger then expected then use that dataset to narrow down
        the distance till it fits the set requirements
    Return fraction,[W_rel_x, W_rel_y, W_rel_z]

    Dis_min/inner
    if Dis_min set to any value <0 the points inside the constellation are considered as well
    if inner is set then only points inside the constellation are considered
    by default the Dis_min starts from the surface of the convex hull and considers only points outside of it
    """
    D_max = Dis_max*R_e
    D_min = Dis_min*R_e
    if D_min > D_max:
        raise ValueError(f"Dis_Min is greater than Dis_max")
    nx, ny, nz = 60, 60, 60
    #Wraps thighly around constellation so max number of points are used
    if inner:
        lil = 0 
    else:
        lil = 1.5*D_max

    x = np.linspace(sc_points[:, 0].min()-lil, sc_points[:, 0].max()+lil, nx)
    y = np.linspace(sc_points[:, 1].min()-lil, sc_points[:, 1].max()+lil, ny)
    z = np.linspace(sc_points[:, 2].min()-lil, sc_points[:, 2].max()+lil, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    hull = ConvexHull(sc_points)
    faces = hull.simplices
    vertices = sc_points
    #Signed Distance Function
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    closest_pts, unsigned_dist, _ = mesh.nearest.on_surface(pts)

    inside = mesh.contains(pts)
    signed_dist = unsigned_dist * np.where(inside, -1.0, 1.0)
    sdf_grid = signed_dist.reshape((nx, ny, nz))
    
    if inner: 
        shell_mask = (signed_dist <= 0)
    elif D_min<0:
        shell_mask = (signed_dist < D_max)
    else: 
        shell_mask = (signed_dist > D_min) & (signed_dist < D_max)
    pts_in_shell = pts[shell_mask]

    Bxyz_vlas = vlsvfile.read_interpolated_variable("vg_b_vol", pts_in_shell)
    Bxyz_RBF = rbf(pts_in_shell / 1000.0)  

    #Gathering data and calculating error
    Bx_vlas = np.full((nx, ny, nz), np.nan)
    By_vlas = np.full((nx, ny, nz), np.nan)
    Bz_vlas = np.full((nx, ny, nz), np.nan)

    Bx_RBF = np.full((nx, ny, nz), np.nan)
    By_RBF = np.full((nx, ny, nz), np.nan)
    Bz_RBF = np.full((nx, ny, nz), np.nan)

    shell_idx = shell_mask.reshape((nx, ny, nz))

    Bx_vlas[shell_idx] = Bxyz_vlas[:, 0]
    By_vlas[shell_idx] = Bxyz_vlas[:, 1]
    Bz_vlas[shell_idx] = Bxyz_vlas[:, 2]

    Bx_RBF[shell_idx] = Bxyz_RBF[:, 0]
    By_RBF[shell_idx] = Bxyz_RBF[:, 1]
    Bz_RBF[shell_idx] = Bxyz_RBF[:, 2]

    B_RBF = [Bx_RBF.ravel(), By_RBF.ravel(),Bz_RBF.ravel()]
    B_vlas =[Bx_vlas.ravel(),By_vlas.ravel(),Bz_vlas.ravel()]
    W_rels = []
    for B_comp_RBF, B_comp_Vlas in zip(B_RBF,B_vlas):

        valid_mask = np.isfinite(B_comp_RBF) & np.isfinite(B_comp_Vlas)
        B_comp_RBF = B_comp_RBF[valid_mask]
        B_comp_Vlas = B_comp_Vlas[valid_mask]

        W1   = wasserstein_distance(B_comp_RBF, B_comp_Vlas)
        comp_med = np.median(B_comp_Vlas)

        #normalizing wasserstein distance by comparison to single valued 
        W_den = wasserstein_distance(B_comp_Vlas, np.full_like(B_comp_Vlas,comp_med))
        relW = np.round(W1 /W_den,4)       
        W_rels.append(relW)
        
    #error 
    dBx = B_RBF[0] - B_vlas[0]
    dBy = B_RBF[1] - B_vlas[1]
    dBz = B_RBF[2] - B_vlas[2]

    dB_mag = np.sqrt(dBx**2 + dBy**2 + dBz**2)   
    Bv_mag = np.sqrt(B_vlas[0]**2+B_vlas[1]**2+B_vlas[2]**2)
    #Get rid of Nans
    valid_mask = np.isfinite(dB_mag) & np.isfinite(Bv_mag) & (Bv_mag > 0)

    error_per = np.full_like(Bv_mag, np.nan)
    error_per[valid_mask] = 100 * dB_mag[valid_mask] / Bv_mag[valid_mask]
    
    good_mask = error_per < error_cutoff
    error_num = error_per[good_mask]

    fraction = len(error_num) / np.count_nonzero(valid_mask)
    if inner: 
        print(f"Fraction of points with error <{error_cutoff}%: {fraction:.3f} inside constellation")
    elif Dis_min == 0:
        print(f"Fraction of points with error <{error_cutoff}%: {fraction:.3f} at distance {D_max/1000:.1f} km from constellation")
    elif D_min<0: 
        print(f"Cumulative fraction of points with error <{error_cutoff}%: {fraction:.3f} at distance {D_max/1000:.1f} km from constellation")
    else: 
        print(f"Fraction of points with error <{error_cutoff}%: {fraction:.3f} between {D_min/1000:.1f} km to {D_max/1000:.1f} km from constellation")
    
    return fraction, W_rels
def limit_plot(error_cut= 50, min_dist= 0.01, max_dist=1, steps = 15, shells = True, save = True, threshold = 0.5, pos = pos_idx):
    
    """
    instead of just making the shell larger shells give the ability to 
    see the fraction of values inside the increased distance shell instead 
    the of whole shell. 
    Ex.
    | shell1 | shell2 | shell3 | 
       0.823    0.654    0.43
    instead of 
    |         shell3           | 
               0.730  
    This is now implemented if shells is set to True

    """
    row     = df.loc[df["Position_Index"] == pos].iloc[0]
    points  = row[pos_cols].to_numpy().reshape(7, 3) 

    Dis_edges = np.linspace(min_dist, max_dist, steps + 1)
    Dis_mids = 0.5 * (Dis_edges[:-1] + Dis_edges[1:])  

    fractions = []
    W_x, W_y, W_z = [], [], []

    fraction_inner, _ = extrapolation_limit(points, inner=True, error_cutoff=error_cut)
   
    outer_edge = Dis_edges[1:]
    if shells:
        inner_edge = Dis_edges[:-1]
    else:
        inner_edge = np.full_like(outer_edge,-1)
    for D_min, D_max in zip(inner_edge, outer_edge):
        fraction, W_rels = extrapolation_limit(points, Dis_min=D_min, Dis_max=D_max, inner=False, error_cutoff=error_cut)
        fractions.append(fraction)
        W_x.append(W_rels[0])
        W_y.append(W_rels[1])
        W_z.append(W_rels[2])
        

    cutoff_value = threshold*fraction_inner
    drop_idx = next((i for i, f in enumerate(fractions) if f < cutoff_value), None)
    drop_dist = Dis_mids[drop_idx] if drop_idx is not None else None

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    axs[0].plot(Dis_mids, fractions, marker="o", label="Shell Accuracy")
    axs[0].scatter(0, fraction_inner, c="red", label="Inside Constellation")
    if drop_dist:
        axs[0].axvline(drop_dist, color="black", linestyle="--", label=rf"{threshold*100:.0f}% drop at {drop_dist:.2f} $R_e$")
    axs[0].set_xlabel(r"Distance from Convex Hull $R_e$")
    axs[0].set_ylabel(f"Fraction with error < {error_cut}%")
    axs[0].set_title("Extrapolation Accuracy")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(Dis_mids, W_x, label=r"$W_x$")
    axs[1].plot(Dis_mids, W_y, label=r"$W_y$")
    axs[1].plot(Dis_mids, W_z, label=r"$W_z$")

    axs[1].set_xlabel(r"Distance from Convex Hull $R_e$")
    axs[1].set_ylabel("Relative Wasserstein Distance")
    axs[1].set_title("Wasserstein Distance by Component")
    axs[1].grid(True)
    axs[1].legend()

    plt.suptitle(f"Position: {pos}")
    plt.tight_layout()
    if save:
        if shells:
            plt.suptitle(f"Error by Shell, Position: {pos}")
            plt.savefig(f"/home/leeviloi/fluxrope_thesis/Accuracy_and_Wasserstein_vs_Distance_Pos={pos}_{error_cut}%_shells.png")
        else: 
            plt.suptitle(f"Cumulative error, Position: {pos}")
            plt.savefig(f"/home/leeviloi/fluxrope_thesis/fly_through_high_res_z=-2_inner=0.14/Accuracy_and_Wasserstein_vs_Distance_Pos={pos}_{error_cut}%_Cumulative.png")

def W_rel_stats(save = True, anim = False):
    """
    This function loops through all the position indices and creates three histogram plots 
    containing all the 1D Wasserstein  
    """
    wx = []
    wy = []
    wz = []
    for pos in range(T):
        #location of spacecrafts at desired index
        if anim:
            anim_path= f"/home/leeviloi/fluxrope_thesis/fly_through_tail/hist_3D_anim/histogram_comparison_comp_counts_3D_pos_{pos}.png"
        else:
            anim_path = None
        row     = df.loc[df["Position_Index"] == pos].iloc[0]
        points  = row[pos_cols].to_numpy().reshape(7, 3) 
    
        w_rel_x, w_rel_y, w_rel_z = Wasser_3D_hist(points, save=anim, path=anim_path, pos_idx=pos)
        #Not relavant for code to work
        #Just wanted to see where outliers were
        """
        if w_rel_x>0.55:
            print(f"W_x = {w_rel_x} at {pos}")

        if w_rel_y>0.55:
            print(f"W_y = {w_rel_y} at {pos}")

        if w_rel_z>0.55:
            print(f"W_z = {w_rel_z} at {pos}")
        """
        wx.append(w_rel_x)
        wy.append(w_rel_y)
        wz.append(w_rel_z)
    fig, axes = plt.subplots(1, 3, figsize=(12,3.5), constrained_layout=True)
    series  = (wx, wy, wz)
    labels  = ("$W_{\\mathrm{rel},x}$", "$W_{\\mathrm{rel},y}$", "$W_{\\mathrm{rel},z}$")

    for ax, w, lab in zip(axes, series, labels):
        w = np.asarray(w)
        p90 = np.percentile(w, 90)          
        p10 = np.percentile(w,10)

        ax.axvline(p90, color="crimson", ls="--", lw=1.8,
               label=f"90%: {np.round(p90,2)}")
        bin_edges = np.linspace(0, 1.6, 51)
        ax.hist(w, bins=bin_edges, color="steelblue", alpha=0.7)
        ax.set_xlim([0,1.6])
        ax.set_xlabel(lab)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        ax.legend(loc= "upper right")
    fig.suptitle("Convex Hull Distribution of $W_{rel}$ errors",
                fontsize=14)    
    if save: 
        plt.savefig(f"/home/leeviloi/fluxrope_thesis/fly_through_tail/W_rel_stats_3D_bins=50.png")
    return

def keep_only_curved(mesh, thresh_rad=np.deg2rad(5), radius = 60):
    
    def max_angle(pts):
        #need 3 points to calculate curvature
        if pts.shape[0] < 3:
            return 0.0
        v = np.diff(pts, axis=0)
        #Calculates unit vector and calculates dot product
        #with next vector
        v_unit = v/np.linalg.norm(v, axis=1, keepdims=True)
        dots   =  (v_unit[:-1] * v_unit[1:]).sum(axis=1)
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        return angles.max()
    
    def as_blocks(mesh):
        blocks = []
        for cid in range(mesh.n_cells):
            ug = mesh.extract_cells(cid)
            pd = ug.extract_surface()
            blocks.append(pd)
        return blocks   
    
    curved = [blk.tube(radius=radius) for blk in as_blocks(mesh)
              if max_angle(blk.points) >= thresh_rad]
    return pv.MultiBlock(curved)                


def fieldlines_3D(pos = 40, ood = False, save = False, pad = 0.2, vlas_lines = True, RBF_lines = True):
    """
    interactive 3D plot of field lines traces from RBF and Vlasiator data. Currently easiest way 
    to interact with the plot is to use ood.cs.helsinki.fi and running the script on there.
    Set ood to True to show plots 
    Code modified from/inspired by: https://magpylib.readthedocs.io/en/latest/_pages/user_guide/examples/examples_vis_pv_streamlines.html 
    TODO: Keep both field lines when one qualifies 
    """

    pos_idx = pos  
    sc_now = df.iloc[pos_idx][pos_cols].to_numpy().reshape(7, 3)/1000.0  

    padding = pad
    #For the integration set resolution to be more directly correlated to step size
    bounds_km = np.array([
        sc_now[:, 0].min() - padding * R_e_km,
        sc_now[:, 0].max() + padding * R_e_km,
        sc_now[:, 1].min() - padding * R_e_km,
        sc_now[:, 1].max() + padding * R_e_km,
        sc_now[:, 2].min() - padding * R_e_km,
        sc_now[:, 2].max() + padding * R_e_km,
    ])
    spacing = (200.0, 200.0, 200.0)
    dims = (
        int((bounds_km[1] - bounds_km[0]) // spacing[0]) + 1,
        int((bounds_km[3] - bounds_km[2]) // spacing[1]) + 1,
        int((bounds_km[5] - bounds_km[4]) // spacing[2]) + 1,
    )
    
    #Origin in pv.ImageData sets the starting corner
    #thats why bary center not used here
    grid = pv.ImageData(
        dimensions=dims,
        spacing=spacing,
        origin=(bounds_km[0], bounds_km[2], bounds_km[4]),
    )
    grid_pts_km = grid.points
    grid_pts_m = grid_pts_km*1000.0

    grid["B_RBF"] = rbf(grid_pts_km)
    B_vlas = vlsvfile.read_interpolated_variable("vg_b_vol", grid_pts_m)
    grid["B_VLAS"] = B_vlas

    
    #INTEGRAL CURVE SEED POINTS
    #spacing of field line seeds
    buffer = 0.2*R_e_km
    #buffer so that integration doesn't start at edge
    #Notice buffer sign
    grid_seed_spacing = 4
    x = np.linspace(bounds_km[0]+buffer, bounds_km[1]-buffer, grid_seed_spacing)
    y = np.linspace(bounds_km[2]+buffer, bounds_km[3]-buffer, grid_seed_spacing)
    z = np.linspace(bounds_km[4]+buffer, bounds_km[5]-buffer, grid_seed_spacing)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid_seeds = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    pl = pv.Plotter()
    all_seeds = np.vstack([sc_now, grid_seeds])
    seed_points = pv.PolyData(all_seeds)
    #seed_points = pv.PolyData(np.vstack([sc_now]))

    streamlines_RBF = grid.streamlines_from_source(
        seed_points,
        vectors="B_RBF",
        max_step_length=400.0,
        integration_direction="both",
    )
    streamlines_vlas = grid.streamlines_from_source(
        seed_points,
        vectors="B_VLAS",
        max_step_length=400,
        integration_direction="both"
    )
    
    curv_thresh = np.deg2rad(2)     
    streamlines_vlas = keep_only_curved(streamlines_vlas, curv_thresh)
    streamlines_RBF  = keep_only_curved(streamlines_RBF,  curv_thresh)
   
    if vlas_lines:
        pl.add_mesh(streamlines_vlas, color="blue", label="Vlasiator")
    if RBF_lines:
        pl.add_mesh(streamlines_RBF, color="red",  label="RBF")    
    """
    if vlas_lines: 
        pl.add_mesh(streamlines_vlas.tube(radius=60), color="blue", label="Vlasiator")
    if RBF_lines:
        pl.add_mesh(streamlines_RBF.tube(radius=60), color="red", label = "RBF")
    """
    pl.add_points(sc_now, color="black", point_size=10)
    pl.add_axes()
    pl.add_legend()
    pl.show_grid(
        xtitle="X [km]",
        ytitle="Y [km]",
        ztitle="Z [km]",
        fmt="%.0f",              
        font_size=15
    )
    if save:
        pl.save_graphic(f"/home/leeviloi/fluxrope_thesis/fly_through_high_res_z=-2_inner=0.14/RBF_Vlas_3D_fieldlines_pos={pos_idx}.svg")
    if ood:
        pl.show(title=f"RBF vs Vlasiator Streamlines at Position_Index={pos_idx}")
    
    return

#Animation functions
def full_comp_anim():
    """
    Function to loop:full_vlas_RBF_error()
    Logic: Loop through all position indecies and save a plot into folder
    Later create animation of photos in said folder using ffmpeg command
    Needed: extract planes at each position
    """
    return

######
#Main#
######

#CHECK WHICH FILE USED AND OUTPUT FILE NAMES

#plot_vlas_RBF_error(vlas_planes,RBF_planes, points=points_incl, rel_error=True)
#full_Wasser_hist(vlas_planes,RBF_planes)
#Wasser_3D_hist(all_points, pos_idx="All Points", save = False, error_cutoff=100.0)
#extrapolation_limit(points, error_cutoff=50, inner = True)
#limit_plot(error_cut = 10, steps = 25, shells=False, pos= 35)
#W_rel_stats(anim = True)
fieldlines_3D(save=False,pos=50,ood = True)