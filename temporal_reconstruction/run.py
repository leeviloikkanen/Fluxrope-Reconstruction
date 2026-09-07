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
import argparse
from dataclasses import fields
import numpy as np

#Load a plasma environment's config
from config import tail_config, magnetopause_config, Config, tail_flipped_config, tail_flipped_centered_config


#cfg = tail_config()
#cfg = magnetopause_config()
#cfg = tail_flipped_centered_config()
cfg = tail_flipped_config()
#make changes to config file variables if needed
#cfg.start_time = 1340
#cfg.end_time = 1372
cfg.t_ref = 1360
#cfg.rbf_eps_method = "LOSO"
#cfg.rbf_kernel = "gaussian"
#Passed in arguments overwrite cfg values
parser = argparse.ArgumentParser()
#Find available arguments from the Config dataclass

import typing

for field in fields(Config):

    #This addition before the else is need to accept --missing_sc "sc1" "sc2" as a input
    field_type = field.type
    if isinstance(field_type, str):
        field_type = typing.get_type_hints(Config)[field.name]

    origin = typing.get_origin(field_type) 

    if origin is typing.Union:
        non_none_args = [a for a in typing.get_args(field_type) if a is not type(None)]
        if len(non_none_args) == 1:
            field_type = non_none_args[0]
            origin = typing.get_origin(field_type)
    if field_type is list or origin is list:
        args_inner = typing.get_args(field_type)
        elem_type = args_inner[0] if args_inner else str
        parser.add_argument(
            f"--{field.name}",
            type=elem_type,
            nargs="+",
            default=None
        )
    else:
        parser.add_argument(
            f"--{field.name}",
            type=field.type,
            default=None
        )

#If variable passed --> Set the variable in the cfg
args = parser.parse_args()
for key, value in vars(args).items():
    if value is not None and hasattr(cfg, key):
        setattr(cfg, key, value)

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
#missing_sc = cfg.missing_sc
rbf, included_pos_cols, included_B_cols, included_sc =  RBF_missing_data(df=df, sc_names=cfg.sc_names(), missing_sc=missing_sc,
                                                                         eps_method=cfg.rbf_eps_method, kernel=cfg.rbf_kernel,
                                                                         )

#Plot the data
import plotting

time = cfg.t_ref

output_dir = "/home/leeviloi/fluxrope_thesis/timeseries_tail/dataset_constrained/centered_at_1360_flipped_z=0.5/radius_wass/"
#output_dir = "./"

file_name = f"RBF_recon_data_t_ref_{cfg.t_ref}_{cfg.end_time-cfg.start_time}s_flipped.png"
wass_file = f"Wasserstein_dist_at_{cfg.t_ref}_{cfg.end_time-cfg.start_time}s_flipped_comparison.png"
wass_sphere_file = f"Wasserstein_dist_at_{cfg.t_ref}_{cfg.end_time-cfg.start_time}s_radius_{cfg.radius}.png"
#output_dir = "/home/leeviloi/fluxrope_thesis/timeseries_magnetopause/dataset_constrained/centred_at_1450/all_wass/"
#plotting.plot_vlas_RBF_error(time = time, df = df, cfg=cfg, rbf=rbf, pos_cols=included_pos_cols, 
#                             included_sc=included_sc, output_dir= output_dir, ref_plane_streak=False, 
#                             output_file=file_name, background = "mag", rel_error=True, err_vmax= 10, stream_color = True)


#wass = plotting.Wasserstein_Hull(time = time, df= df, rbf = rbf, cfg=cfg, pos_cols=pos_cols, error_cutoff=20,
#                                  save = True, output_dir=output_dir, output_file = wass_file) 
#print(wass)

wass_sphere = plotting.Wasserstein_sphere(time = time, df=df, rbf = rbf, cfg = cfg, pos_cols = pos_cols,
                                    save = True, output_dir = output_dir, domain = True, output_file = wass_sphere_file,
                                    domain_file = f"tail_sampling_domain_R={cfg.radius}.png", R_RE=cfg.radius)
print(wass_sphere)