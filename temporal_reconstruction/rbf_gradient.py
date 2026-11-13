import numpy as np
from config import Config, tail_config, magnetopause_config, tail_flipped_centered_config
from itertools import combinations
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/leeviloi/analysator-dev")
import analysator as pt;# print(pt.__file__)

"""
Calculate the least squared gradient and RBF gradients with varying constellation sizes 
and compare to Vlasiator gradients 
"""
R_e = 6371000
bulk_path = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/bulk1/bulk1.{:07d}.vlsv"

def rbf_jacobian(rbf, x):
    """
    J[m,i,k] = dB_i/dx_k of a scipy RBFInterpolator.
    with multiquadric kernel, can be degree 0 or 1
    """
    x = np.atleast_2d(x)
    eps = rbf.epsilon
    N = rbf.y.shape[0]
    c, cp = rbf._coeffs[:N], rbf._coeffs[N:]
    diff = x[:, None, :] - rbf.y[None, :, :]
    #derivative
    w = -eps**2 / np.sqrt(1 + eps**2 * np.sum(diff**2, axis=-1))
    J = np.einsum("mn,mnk,ni->mik", w, diff, c)
    for row, p in enumerate(rbf.powers):
        if p.sum() == 1:
            k = int(np.argmax(p))
            J[:, :, k] += cp[row] / rbf._scale[k]
    return J

def lsq_gradient(pos, field):
    """
    Least squares gradient function
    """
    N = len(pos)
    #distance to bary center
    dr0 = pos - pos.mean(axis=0)
    R = dr0.T @ dr0 / N
    S = np.zeros((3, 3))
    for a in range(N):
        for b in range(a + 1, N):
            S += np.outer(field[a] - field[b], pos[a] - pos[b])
    return S / N**2 @ np.linalg.inv(R)

def constellation_wellness(cfg:Config):
    """
    Function to check the planarity, elongation and characteric size of a constellation 
    according to "Multi-Point Gradient Estimation in Turbulence" Broern and Klein 2024
    """

    init_pts = np.vstack(list(cfg.sc_init.values()))

    N = init_pts.shape[0]
    bary = init_pts.mean(axis=0)
    d = init_pts-bary

    #The volumetric tensor matrix
    R = d.T @ d/N

    #Eigenvalues
    lam, vecs = np.linalg.eigh(R)                  
    lam, vecs = lam[::-1], vecs[:, ::-1]          
    lam = np.clip(lam, 0.0, None)                  
    a, b, c = 2.0 * np.sqrt(lam)                    

    #Elongation
    E = 1.0 - b / a if a > 0 else np.nan          
    #Planarity
    P = 1.0 - c / b if b > 0 else np.nan            
    #The single scalar metric
    chi = np.sqrt(E**2 + P**2)                       

    return {"R": R, "barycenter": bary, "eigvals": lam, "eigvecs": vecs,
            "a": a, "b": b, "c": c,
            "L": a,                                 
            "E": E, "P": P, "chi": chi}

def read(reader, var, pts):
    return np.asarray(reader.read_interpolated_variable(var, pts), dtype=float)



def summarize_results(G_ref, G_rbf, G_lsq, info=True):
    """Relative Frobenius error of the RBF and LSQ Jacobians against the reference.
    G_ref, G_rbf, G_lsq : (3, 3) Jacobians, G[i, j] = dB_i/dx_j
    Returns (err_rbf, err_lsq).
    """
    norm_ref = np.linalg.norm(G_ref)
    err_rbf = np.linalg.norm(G_rbf - G_ref) / norm_ref
    err_lsq = np.linalg.norm(G_lsq - G_ref) / norm_ref
    if info:
        print(f"{'':<10}{'rel err ∇B':>12}")
        print(f"{'RBF':<10}{err_rbf:>12.3f}")
        print(f"{'LSQ':<10}{err_lsq:>12.3f}")
    return err_rbf, err_lsq

def plot_gradient_errors(cfg:Config, save = None):
    times = range(cfg.start_time+5, cfg.end_time-5)
    errors = {"RBF":[], "LSQ":[]}
    for time in times:
        cfg.t_ref = time
        pos = np.vstack(list(cfg.sc_init.values()))
        
    
        flow = flow_type(sc_init=cfg.sc_init, velocity_path=cfg.vg_v_file, B_field_path=cfg.b_field_file, 
                        start_time=cfg.start_time, end_time=cfg.end_time)
    
        pos_cols, B_cols = flow.steady_flow_velocity(t_ref=cfg.t_ref,true_times=False)
    
        df = flow.df
            
        #Build the interpolator
        from rbf import RBF_missing_data
    
        missing_sc = None
        #missing_sc = cfg.missing_sc
        rbf, included_pos_cols, included_B_cols, included_sc =  RBF_missing_data(df=df, sc_names=cfg.sc_names(), missing_sc=missing_sc,
                                                                                eps_method=cfg.rbf_eps_method, kernel=cfg.rbf_kernel,
                                                                                )
        
        init_pts = np.vstack(list(cfg.sc_init.values()))
        
        N = init_pts.shape[0]
        bary = init_pts.mean(axis=0)
    
        f = pt.vlsvfile.VlsvReader(bulk_path.format(cfg.t_ref))
        B_sc = read(f, "vg_b_vol", pos)   
    
        
        G_rbf = rbf_jacobian(rbf, bary)[0]
        G_lsq = lsq_gradient(pos, B_sc)
        G_sim = read(f, "vg_jacobian_b", bary[None]).reshape(3, 3)
        
        rbf_error, lsq_error = summarize_results(G_sim, G_rbf, G_lsq, info=False)
        errors["RBF"].append(rbf_error)
        errors["LSQ"].append(lsq_error)
    
    
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(times, errors["RBF"], "-o", ms=3, label="RBF")
    ax.plot(times, errors["LSQ"], "-o", ms=3, label="LSQ")
    ax.set_xlabel("time frame")
    ax.set_ylabel("Relative error")
    #ax.set_yscale("log")
    ax.set_title("Relative gradient error at the barycenter")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    if save:
        fig.savefig(save, dpi=150)
        
    return fig, times, errors

if __name__ == "__main__":

    from flow_type import flow_type
    cfg  = magnetopause_config()
    plot_gradient_errors(cfg=cfg, save ="./gradient_errors_magnetopause2.png")
    
    pos = np.vstack(list(cfg.sc_init.values()))
    cfg.t_ref = 1450
    time = cfg.t_ref

    flow = flow_type(sc_init=cfg.sc_init, velocity_path=cfg.vg_v_file, B_field_path=cfg.b_field_file, 
                    start_time=cfg.start_time, end_time=cfg.end_time)

    pos_cols, B_cols = flow.steady_flow_velocity(t_ref=cfg.t_ref,true_times=False)

    df = flow.df
    df_v = flow.df_v
    #print(constellation_wellness(magnetopause_config()))
        
    #Build the interpolator
    from rbf import RBF_missing_data

    missing_sc = None
    #missing_sc = cfg.missing_sc
    rbf, included_pos_cols, included_B_cols, included_sc =  RBF_missing_data(df=df, sc_names=cfg.sc_names(), missing_sc=missing_sc,
                                                                            eps_method=cfg.rbf_eps_method, kernel=cfg.rbf_kernel,
                                                                            )
    
    init_pts = np.vstack(list(cfg.sc_init.values()))
    
    N = init_pts.shape[0]
    bary = init_pts.mean(axis=0)

    f = pt.vlsvfile.VlsvReader(bulk_path.format(cfg.t_ref))
    B_sc = read(f, "vg_b_vol", pos)   

    
    G_rbf = rbf_jacobian(rbf, bary)[0]
    G_lsq = lsq_gradient(pos, B_sc)
    G_sim = read(f, "vg_jacobian_b", bary[None]).reshape(3, 3)
    
    np.set_printoptions(precision=3, suppress=True)
    # T/m -> nT/R_E
    to = 1e9 * R_e                                                     
    print("rows: B_x, B_y, B_z   columns: d/dx, d/dy, d/dz   [nT/R_E]\n")
    print("RBF:\n", G_rbf*to, "\n")
    print("LSQ:\n", G_lsq*to, "\n")
    print("Vlasiator:\n", G_sim*to)
    summarize_results(G_sim,G_rbf,G_lsq)
    
    pass