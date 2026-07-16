"""

Move/adapt all plotting scripts from RBF_time_reconstruction.py into plotting.py
Needed: RBF, SC locations, etc 

"""
import numpy as np

from sampling import sample_slice, sample_slice_any_plane, sample_slice_vlas, sample_slice_vlas_coords
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import wasserstein_distance
import scipy

sys.path.insert(0, "/home/leeviloi/analysator-dev")
import analysator as pt; print(pt.__file__)

#TODO needs sc_inits and rbf 

def plot_vlas_slices(time, nx = 200, ny = 200, L_Re = 1.2, output_dir = None, output_file = None, save = True):
    """
    Plotting vlasiators slices at the barycenter of the spacecraft constellations

    :kword time: Time of reconstruction 
    :kword save: Save figure 
    :kword L_Re: Set size of slice
    :kword output_dir: Output file directory for saving figure
    :kword output_file: Output file name 
    
    """

    file = f"/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.000{time}.vlsv"
    print(file)
    vlsvfile = pt.vlsvfile.VlsvReader(file)

    XY = sample_slice_vlas(vlsvfile = vlsvfile, plane = "xy", nx=nx, ny=ny, L_Re=L_Re)
    XZ = sample_slice_vlas(vlsvfile = vlsvfile, plane = "xz", nx=nx, ny=ny, L_Re=L_Re)
    YZ = sample_slice_vlas(vlsvfile = vlsvfile, plane = "yz", nx=nx, ny=ny, L_Re=L_Re)

    init_pts = np.vstack(list(sc_init.values()))
    """
    bulkpath_FHA = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/bulk1/"
    file_readers = []
    for t in df["Timeframe"].values:
        file_readers.append(pt.vlsvfile.VlsvReader(bulkpath_FHA+"bulk1.{}.vlsv".format(str(int(t)).zfill(7)),indexer="dict"))

    injection_points = np.array([sc_init[f"{sc}"] for sc in included_sc])

    time_interpolator = pt.calculations.VlsvTInterpolator(vlsvReaders_list = file_readers)

    streakobj = pt.calculations.fieldtracer.streaklines(vlsvTObject = time_interpolator, seed_points = injection_points, direction = "+",dt_step = 0.1, 
                                                        points_per = 1, method = "RK4", tracked_vars = ["vg_b_vol"])
    i_rel = streakobj._time_to_idx(time = df["Timeframe"].iloc[-1])
    j_rel = streakobj._time_to_idx(time = time)
    ref_points = streakobj.M[:,i_rel,j_rel]
    bary_vlas  = ref_points.mean(axis=0)                              

    L_vlas = L_Re*R_e
                             
    x = np.linspace(bary_vlas[0]-L_vlas, bary_vlas[0]+L_vlas, nx)
    y = np.linspace(bary_vlas[1]-L_vlas, bary_vlas[1]+L_vlas, ny)
    z = np.linspace(bary_vlas[2]-L_vlas, bary_vlas[2]+L_vlas, ny)   
    init_pts = ref_points
    
    XY = sample_slice_vlas_coords(df["Timeframe"].iloc[-1], x, y, bary_vlas[2], "xy")
    XZ = sample_slice_vlas_coords(df["Timeframe"].iloc[-1], x, z, bary_vlas[1], "xz")
    YZ = sample_slice_vlas_coords(df["Timeframe"].iloc[-1], y, z, bary_vlas[0], "yz")
    """

    fig, axs = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)
    for ax, (data, title) in zip(axs, zip([XY,XZ,YZ], ["X-Y","X-Z","Y-Z"])):
       
        C1, C2, U, V, W = data

        mag = np.hypot(U, V)
        cf  = ax.contourf(C1, C2, W, 30, cmap="coolwarm")
        ax.streamplot(C1, C2, U, V,
                      color=mag, cmap="magma", density=1.5, linewidth=0.5)
        if title == "X-Y":
            sc_u = init_pts[:,0]; sc_v = init_pts[:,1]
            out_comp = "Z"
        elif title == "X-Z":
            sc_u = init_pts[:,0]; sc_v = init_pts[:,2]
            out_comp = "Y"
        else:  
            sc_u = init_pts[:,1]; sc_v = init_pts[:,2]
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
    fig.suptitle(f"Vlasiator slices at Time = {time} s")
    
    if output_dir == None:
        output_dir = "~/"

    if output_file == None:
        output_file = f"vlasitor_slices_{time}s.png"
    
    output_file = output_dir+output_file
    if save: 
        plt.savefig(output_file)
    plt.close()

    return    

def plot_rbf_slices(time, nx = 200, ny = 200, L_Re = 1.2, output_dir = None, output_file = None):
    """
    Plots the RBF reconstruction at the bary center of the spacecraft constellation

    :kword time: Time of reconstruction 
    :kword L_Re: Set size of slice
    :kword output_dir: Output file directory for saving figure
    :kword output_file: Output file name 
    
    """
    #Find coordinates of times
    row = df[df["Timeframe"] == time].iloc[0]
    cluster = row[pos_cols].to_numpy().reshape(-1,3)
    bary    = cluster.mean(axis=0)

    L_m = L_Re * R_e
    xs = np.linspace(bary[0]-L_m, bary[0]+L_m, nx)
    ys = np.linspace(bary[1]-L_m, bary[1]+L_m, ny)
    zs = np.linspace(bary[2]-L_m, bary[2]+L_m, ny)

    #Sample the coordinates 
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

def plot_vlas_RBF_error(time, save = True, rel_error = True, L_Re = 1.2, output_dir = None, output_file = None, nx = 200, ny = 200, err_vmax = 1.5e-8, ref_plane_streak = False):
    """
    Creates a 3x3 plot of countours  (First row Vlasiator xy, xz and yz planes with streamlines,
    Second row RBF xy, xz, yz planes with streamliens, Third row point-wise error comparison of 
    the magnetic field strenght of the first two rows)  
    
    time : Time of reconstruction 
    save : Save figure
    L_Re : Set size of slice
    nx, ny : Grid resolution
    rel_error : Toggle between relative error and absolute error
    err_vmax : Set max error for the absolute error (should also be implemented for relative error)
    output_dir : Output file directory for saving figure
    output_file : Output file name 
    ref_plane_streak : Boolean that determines whether RBF slices are compared to the advected vlasiator points at the end time 
                       or the planes the seed points at the time of measurement 
    
    TODO: Recenter RBF points to original points. Could be just set vlasiator grid for RBF grid
    Currently I guess SCs moving in the flux rope rest frame?!?
    """
    #Vlasitor DATA
    

    
    if ref_plane_streak:
        #measures at advected SC locations at the refrence time
        bulkpath_FHA = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/bulk1/"
        file_readers = []
        for t in df["Timeframe"].values:
            file_readers.append(pt.vlsvfile.VlsvReader(bulkpath_FHA+"bulk1.{}.vlsv".format(str(int(t)).zfill(7)),indexer="dict"))

        injection_points = np.array([sc_init[f"{sc}"] for sc in included_sc])

        time_interpolator = pt.calculations.VlsvTInterpolator(vlsvReaders_list = file_readers)

        streakobj = pt.calculations.fieldtracer.streaklines(vlsvTObject = time_interpolator, seed_points = injection_points, direction = "+",dt_step = 0.1, 
                                                            points_per = 1, method = "RK4", tracked_vars = ["vg_b_vol"])
        i_rel = streakobj._time_to_idx(time = df["Timeframe"].iloc[-1])
        j_rel = streakobj._time_to_idx(time = time)
        ref_points = streakobj.M[:,i_rel,j_rel]
        bary_vlas  = ref_points.mean(axis=0)                              

        L_vlas = L_Re*R_e
                                
        x = np.linspace(bary_vlas[0]-L_vlas, bary_vlas[0]+L_vlas, nx)
        y = np.linspace(bary_vlas[1]-L_vlas, bary_vlas[1]+L_vlas, ny)
        z = np.linspace(bary_vlas[2]-L_vlas, bary_vlas[2]+L_vlas, ny)   
        init_pts = ref_points
        
        XY_vlas = sample_slice_vlas_coords(df["Timeframe"].iloc[-1], x, y, bary_vlas[2], "xy")
        XZ_vlas = sample_slice_vlas_coords(df["Timeframe"].iloc[-1], x, z, bary_vlas[1], "xz")
        YZ_vlas = sample_slice_vlas_coords(df["Timeframe"].iloc[-1], y, z, bary_vlas[0], "yz")
        
    else:
        #Samples at seed points coordinates at the measurement time
        file = f"/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.000{time}.vlsv"
        print(file)
        vlsvfile = pt.vlsvfile.VlsvReader(file)
        XY_vlas = sample_slice_vlas(vlsvfile = vlsvfile, plane = "xy", nx=nx, ny=ny, L_Re=L_Re)
        XZ_vlas = sample_slice_vlas(vlsvfile = vlsvfile, plane = "xz", nx=nx, ny=ny, L_Re=L_Re)
        YZ_vlas = sample_slice_vlas(vlsvfile = vlsvfile, plane = "yz", nx=nx, ny=ny, L_Re=L_Re)

        init_pts = np.vstack(list(sc_init.values()))
        
    vlas_planes = [XY_vlas, XZ_vlas, YZ_vlas]
    
    #RBF DATA
    row = df[df["Timeframe"] == time].iloc[0]
    cluster = row[pos_cols].to_numpy().reshape(-1,3)
    bary    = cluster.mean(axis=0)

    L_m = L_Re * R_e
    xs = np.linspace(bary[0]-L_m, bary[0]+L_m, nx)
    ys = np.linspace(bary[1]-L_m, bary[1]+L_m, ny)
    zs = np.linspace(bary[2]-L_m, bary[2]+L_m, ny)

    XY_rbf = sample_slice(xs, ys, bary[2], "xy", nx, ny)
    XZ_rbf = sample_slice(xs, zs, bary[1], "xz", nx, ny)
    YZ_rbf = sample_slice(ys, zs, bary[0], "yz", nx, ny)
    
    rbf_planes = [XY_rbf,XZ_rbf,YZ_rbf]

    fig, axes = plt.subplots(3,3,figsize = (13,11), constrained_layout=True)
    fig.dpi = 500
    panels = [
    ("X-Y",("X","Y","z")),
    ("X-Z",("X","Z","y")),
    ("Y-Z",("Y","Z","x")),
    ]
    if rel_error:
        err_vmin, err_vmax = 0.0, 50.0                     
        levels   = np.linspace(err_vmin, err_vmax, 31)     
        norm     = mpl.colors.Normalize(vmin=err_vmin, vmax=err_vmax)
        error_lbl = "Error (%)"
        error_title = "point-wise error (%)"
    else:

        err_vmin, err_vmax = 0.0, err_vmax
        levels   = np.linspace(err_vmin, err_vmax, 31)
        norm     = mpl.colors.Normalize(vmin=err_vmin, vmax=err_vmax)
        error_lbl = "|ΔB|"  
        error_title = "Absolute point-wise error"          

    clus_size = 20

    for i, (vlas_plane,rbf_plane, panel) in enumerate(zip(vlas_planes,rbf_planes,panels)):

        #Component naming here wrong but makes no difference with absolute error
        Pr, Qr, Bxr, Byr, Bzr = rbf_plane
        Pv, Qv, Bxv, Byv, Bzv = vlas_plane    

        dBx = Bxr - Bxv
        dBy = Byr - Byv
        dBz = Bzr - Bzv

        dB_mag = np.sqrt(dBx**2 + dBy**2 + dBz**2)   
        Bv_mag = np.sqrt(Bxv**2+Byv**2+Bzv**2)
        if rel_error:
            error = 100*dB_mag/Bv_mag
        else: 
            error = dB_mag
        title, (lab1, lab2, lab3) = panel
        Pr, Qr = rbf_plane[0], rbf_plane[1]
    
        #
        #Vlasiator Plotting
        #
        cont_0 = axes[0,i].contourf(Pv,Qv,vlas_plane[-1], 30, cmap="coolwarm")
        speed = np.hypot(vlas_plane[2], vlas_plane[3])
        axes[0,i].streamplot(Pv, Qv, vlas_plane[2], vlas_plane[3],
                    color=speed, cmap="magma", density=2, linewidth = 0.4)
        
        if lab1 == "X":             
            u_v = init_pts[:,0]
            u_r = cluster[:,0] 
        elif lab1 == "Y":           
            u_v = init_pts[:,1]
            u_r = cluster[:,1]
        else:
            u_v = init_pts[:,2]
            u_r = cluster[:,2]
        if lab2 == "Y":             
            v_v = init_pts[:,1]
            v_r = cluster[:,1]
        elif lab2 == "Z":           
            v_v = init_pts[:,2]
            v_r = cluster[:,2]
        else:
            v_v = init_pts[:,0]
            v_r = cluster[:,0]

        def scatter_mask(u,v, P, Q):
            xmin, xmax = np.min(P), np.max(P)
            ymin, ymax = np.min(Q), np.max(Q)

            #masking the points such that they are shown only
            # if theyre within the grid
            mask = (u>= xmin)&(u<= xmax)&(v>=ymin)&(v<=ymax)

            return u[mask], v[mask]
        
        u_v, v_v = scatter_mask(u_v, v_v, Pv, Qv)
        u_r, v_r = scatter_mask(u_r, v_r, Pr, Qr)

        cbar = fig.colorbar(cont_0, ax=axes[0,i], orientation="vertical", shrink = 0.8)
        cbar.set_label(f"$B_{lab3}$")
        axes[0,i].scatter(u_v, v_v, c="k", s=clus_size, label="spacecraft")
        axes[0,i].margins(0)
        #axes[0,i].set_xlabel(f"{lab1}  (m)")
        axes[0,i].set_ylabel(f"{lab2} (m)")
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
        axes[1,i].scatter(u_r, v_r, c="k", s=clus_size, label="spacecraft")
        axes[1,i].margins(0)
        #axes[1,i].set_xlabel(f"{lab1}  (10³ km)")
        axes[1,i].set_ylabel(f"{lab2} (m)")
        axes[1,i].set_aspect("equal")
        axes[1,0].legend(loc="upper right",fontsize="small")
        #axes[1,i].set_title(title)
        #
        #POINT-WISE ERROR
        #
        cf = axes[2,i].contourf(Pv, Qv, error, levels = levels, cmap="viridis", norm = norm, extend = "max")
      
        axes[2,i].margins(0) 
        axes[2,i].scatter(u_v, v_v, c="k", s=clus_size, label="spacecraft")
        axes[2,i].set_xlabel(f"{lab1} (m)")
        axes[2,i].set_ylabel(f"{lab2} (m)")
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
    fig.suptitle(f"Comparison of Vlasiator and RBF reconstruction at time {t_ref}, release time = {time}s", fontsize = 20)
    if save:
        if output_dir == None:
            output_dir = "~/"

        if output_file == None:
            if rel_error:
                output_file = f"full_vlas_rbf_comp_time={time}_L={L_Re}_GOOD_scale.png"
            else: 
                output_file = f"full_vlas_rbf_comp_time={time}_L={L_Re}_abs_error.png"
        
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
    print(f"W_rel 50th percentiles: W_x = {np.percentile(W_x,50)} W_y = {np.percentile(W_y,50)}, W_z = {np.percentile(W_z,50)}")
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
        if flow.static_vel is not None:
            fig.suptitle(f"Bulk velocity: ({np.round(flow.static_vel[0],1)},{np.round(flow.static_vel[1],1)},{np.round(flow.static_vel[2],1)}) m/s")
        else:
            fig.suptitle("Dynamic Bulk velocity")
        fig.tight_layout()

        if output_dir == None:
            output_dir = "~/"

        if output_file == None:
            if flow.static_vel is not None:
                output_file = f"Wasserstein_vs_Time+error_abs_bulk={np.sqrt(flow.static_vel[0]**2+flow.static_vel[1]**2+flow.static_vel[2]**2)}.png"
            else:
                output_file = f"Wasserstein_vs_Time+error_abs_dynamic.png"
        
        output_file = output_dir+output_file

        plt.savefig(output_file)

    return 
