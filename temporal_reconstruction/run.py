"""

Simple script that just calls other functions in other to run wanted feature
-load config
-set what is to be plotted/parameters changed
-plot/print

Would be nice to have a print of the parameters when running as well
if info: 
LOADED CONFIG X
--------------------
Vlasiator File: bulk1.0005000.vlsv
--------------------
Time range: X-Y
--------------------
Reconstruction region: 
    x: 0-100 R_E
    y: -40-40 R_E
    z: -40-40 R_E
--------------------
Flow type: Individual Ballistic
--------------------
RUNNING FUNCTION: Plotting whatever
--------------------
Reconstruction accuracy in the domain
101% OMG 
--------------------
DONE!
"""

#Load a plasma environment's config
from config import tail_config, magnetopause_config

cfg = tail_config()

#make changes to config file variables if needed

cfg.start_time = 1340
cfg.end_time = 1372
cfg.t_ref = 1360
#cfg.rbf_eps_method = "LOOCV"
#Load the data and move the centers 

from flow_type import flow_type

flow = flow_type(sc_init=cfg.sc_init, velocity_path=cfg.vg_v_file, B_field_path=cfg.b_field_file, 
                 start_time=cfg.start_time, end_time=cfg.end_time)

pos_cols, B_cols = flow.steady_flow_velocity(t_ref=cfg.t_ref)

df = flow.df
df_v = flow.df_v

#Build the interpolator
from rbf import RBF_missing_data

missing_sc = None
#missing_sc = ["sc2"]
rbf, included_pos_cols, included_B_cols, included_sc =  RBF_missing_data(df=df, sc_names=cfg.sc_names(), missing_sc=missing_sc,
                                                                         eps_method=cfg.rbf_eps_method, kernel=cfg.rbf_kernel)


#Plot the data
import plotting

time = 1360

plotting.plot_vlas_RBF_error(time = time, df = df, cfg=cfg, rbf=rbf, pos_cols=included_pos_cols, 
                             included_sc=included_sc, output_dir= "./", ref_plane_streak=False, 
                             output_file=f"RBF_recon_data_{cfg.start_time}-{cfg.end_time}s_t_ref_{cfg.t_ref}_tau_{time}_eps_inv.png")