#This script was created by Leevi Loikkanen to do error analysis between Vlasiator simulation files 
#and reconstructed magnetic fields from virtual spacecraft data using Radial Basis Functions  
#by means of point-wise error and Wasserstein distances 
#RBF method and eroor analysis method"s used here outlined in paper: 
#https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2023EA003369

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
import pytools as pt
import matplotlib as mpl

#Shared info
df = pd.read_csv("/home/leeviloi/plas_obs_vg_b_full_1432_fly_up+pos.csv")
t = 1432
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
pos_idx = 20           
nx, ny  = 200, 200    

times = df["Position_Index"].to_numpy() 
T = len(times)
sc_names  = [f"sc{i}" for i in range(1, 8)]
pos_cols  = sum([[f"{sc}_pos_x", f"{sc}_pos_y", f"{sc}_pos_z"] for sc in sc_names], [])
B_cols    = sum([[f"{sc}_vg_B_x", f"{sc}_vg_B_y", f"{sc}_vg_B_z"] for sc in sc_names], [])

#####
#RBF#
#####

centers = (df[pos_cols].to_numpy().reshape(T * 7, 3) / 1000.0)   # km
values  =  df[B_cols].to_numpy().reshape(T * 7, 3)   

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


row = df.loc[df["Position_Index"] == pos_idx].iloc[0]
cluster_now = row[pos_cols].to_numpy().reshape(7, 3) / 1000.0     
bary_rbf = cluster_now.mean(axis=0)                             

                                      

def sample_slice(coord1, coord2, const_coord, plane):
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
bary_vlas    = points.mean(axis=0)                              

                             
x = np.linspace(bary_vlas[0]-L_vlas, bary_vlas[0]+L_vlas, nx)
y = np.linspace(bary_vlas[1]-L_vlas, bary_vlas[1]+L_vlas, ny)
z = np.linspace(bary_vlas[2]-L_vlas, bary_vlas[2]+L_vlas, ny)

#function to extract meshgrids of simulation data in different planes for reference 
def sample_slice_vlas(coord1, coord2, const_coord, plane):

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

def plane_wasserstein(rbf_tuple, sim_tuple, plane_name):
    COMP_INDEX = {
    "xy": (2, 3, 4),   # tuple = (X, Y,  Bx, By, Bz)
    "xz": (2, 4, 3),   # tuple = (X, Z,  Bx, Bz, By)
    "yz": (4, 2, 3),   # tuple = (Y, Z,  By, Bz, Bx)
    }
    id_x, id_y, id_z = COMP_INDEX[plane_name]

    # reorder each tuple into canonical (Bx, By, Bz)
    Br = [rbf_tuple[idx] for idx in (id_x, id_y, id_z)]
    Bs = [sim_tuple[idx] for idx in (id_x, id_y, id_z)]
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


results = []
results.append(plane_wasserstein(XY_RBF, XY_Vlas, "xy"))
results.append(plane_wasserstein(XZ_RBF, XZ_Vlas, "xz"))
results.append(plane_wasserstein(YZ_RBF, YZ_Vlas, "yz"))

df_W = pd.DataFrame(results).set_index("plane")
print(f"Wasserstein values at position {pos_idx}")
print(df_W)

vlas_planes = [XY_Vlas,XZ_Vlas,YZ_Vlas]
RBF_planes = [XY_RBF,XZ_RBF,YZ_RBF]  

def error_perscentage(B_plane_RBF, B_plane_vlas):

  

    Pr, Qr, Bxr, Byr, Bzr = B_plane_RBF
    Pv, Qv, Bxv, Byv, Bzv = B_plane_vlas     
    if not np.allclose(Pr,Pv/1e3) and np.allclose(Qr,Qv/1e3):
        raise ValueError("fields dont match")
    dBx = Bxr - Bxv
    dBy = Byr - Byv
    dBz = Bzr - Bzv

    dB_mag = np.sqrt(dBx**2 + dBy**2 + dBz**2)   
    Bv_mag = np.sqrt(Bxv**2+Byv**2+Bzv**2)
    error_per = 100*dB_mag/Bv_mag
    return Pr, Qr, error_per

#test = error_perscentage(XY_RBF, XY_Vlas)
"""
threshold = 220.0     # %
mask = error_per > threshold

if mask.any():
    idx      = np.argwhere(mask)          
    npoints  = idx.shape[0]
    print(f"{npoints} points with error {threshold}%:")
    print("    X   Y    dB   B_vlas   %")
    for k, (i, j) in enumerate(idx):
        if k == 20:                       
            break
        x_km = Xr[i, j] / 1e3            
        y_km = Yr[i, j] / 1e3
        print(f"    {x_km:7.1f}  {y_km:7.1f}   {dB_mag[i,j]:8.3g}   "
              f"{Bv_mag[i,j]:7.3g}   {error_per[i,j]:6.1f}")
else:
    print(f"No points have {threshold}% error.")
"""
"""
fig, axs = plt.subplots(1,3, figsize = (7,6))

contour = axs.contourf(Xr/1e3,Yr/1e3, error_per, 40, cmap = "viridis")

fig.colorbar(contour, ax=axs, label = " {%} error")
axs.set_aspect("equal")
axs.set_xlabel("X 10³ km")
axs.set_ylabel("Y 10³ km")
axs.margins(0)
#plt.savefig("/home/leeviloi/scripts_useful/error_test_xy_%_L=1.2_pos=50.png")
"""
##########
#PLOTTING#
##########
def plot_point_wise_error():
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

    #plt.savefig(f"/home/leeviloi/fluxrope_thesis/error_threeslice_L=1.2_pos={pos_idx}_1432s.png")
    return

def plot_hist_component_comparison(plane_RBF, plane_Vlas, plane, type = "filled"):
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
        comp_med = np.median(B_comp_RBF)

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

#plot_hist_component_comparison(XY_RBF,XY_Vlas, "xy")


def plot_vlas_RBF_error(vlas_planes, rbf_planes):
    err_xy =  error_perscentage(rbf_planes[0],vlas_planes[0])
    err_xz = error_perscentage(rbf_planes[1],vlas_planes[1])
    err_yz =error_perscentage(rbf_planes[2],vlas_planes[2])
    fig, axes = plt.subplots(3,3,figsize = (13,11), constrained_layout=True)
    fig.dpi = 500
    panels = [
    ("X-Y  (z = {:.0f} km)".format(bary_vlas[2]/1e3),("X","Y","z"),err_xy[-1]),
    ("X-Z  (y = {:.0f} km)".format(bary_vlas[1]/1e3),("X","Z","y"),err_xz[-1]),
    ("Y-Z  (x = {:.0f} km)".format(bary_vlas[0]/1e3),("Y","Z","x"),err_yz[-1]),
    ]
    err_vmin, err_vmax = 0.0, 50.0                     
    levels   = np.linspace(err_vmin, err_vmax, 31)     
    norm     = mpl.colors.Normalize(vmin=err_vmin, vmax=err_vmax)
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
                orientation="vertical", label="Error %", shrink = 0.8)
    #Label each row
    row_y = [0.96, 0.64, 0.32]    
    for y, txt in zip(row_y,
                    ["Vlasiator",
                    "RBF",
                    "point-wise error (%)"]):
        fig.text(0.5, y, txt, ha="center", va="center", fontsize=20)
    #fig.tight_layout()
    fig.suptitle(f"Comparison of Vlasiator and RBF reconstruction at Pos={pos_idx}, time = 1432s", fontsize = 20)
    #plt.savefig(f"/home/leeviloi/fluxrope_thesis/full_vlas_rbf_comp_pos:{pos_idx}_time=1432_L={Lsize}_7.png")
    return
def full_Wasser_hist(vlas_planes, rbf_planes, type = "filled"):

    fig, axes = plt.subplots(3,3, figsize = (12,10), constrained_layout = True)
    n_bins = 40
    fig.dpi = 150
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
    plt.savefig(f"/home/leeviloi/fluxrope_thesis/histogram_3x3_L={Lsize}_time=1432s_pos={pos_idx}_CORRECT.png")
    return
def Wasser_3D_hist(sc_points, type = "filled"):
    nx, ny, nz = 60, 60, 60
    lil = 0.1*R_e
    x = np.linspace(sc_points[:,0].min(),sc_points[:,0].max(),nx)
    y = np.linspace(sc_points[:,1].min(),sc_points[:,1].max(),ny)
    z = np.linspace(sc_points[:,2].min(),sc_points[:,2].max(),nz)
    
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
        ax.set_xlabel(f"{label}")
        #ax.set_title(rf"$W_{{\mathrm{{rel}}}} = {W_val}$")
        ax.grid(alpha=0.3)
        axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.suptitle(f"Histograms of component counts at Pos={pos_idx}, Convex Hull")
    print("Everything worked")
    #plt.savefig(f"/home/leeviloi/fluxrope_thesis/histogram_comparision_comp_counts_3D_pos_{pos_idx}_no_den_type={type}_3.png")
  
    return
#plot_vlas_RBF_error(vlas_planes,RBF_planes)
#full_Wasser_hist(vlas_planes,RBF_planes)
Wasser_3D_hist(points)
