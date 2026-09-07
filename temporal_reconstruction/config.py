"""

Script where basic running parameters can be saved for different locations/times
Reason for this is that with multiple runs it will be difficult to remember all the settings

"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
R_E = 6371000

@dataclass
class Config:

    vg_v_file: str = ""
    b_field_file: str = ""
    bulk_path: str = "/turso/group/spacephysics/vlasiator/data/L1/3D/FHA/bulk1/"
    output_dir: str = "./"

    t_ref: int = -1
    start_time: int = None
    end_time: int = None 

    sc_init: dict = field(default_factory=dict)

    rbf_kernel: str = "multiquadric"
    rbf_eps_method: str = "neighbour"
    rbf_smoothing: float = 0.0
    missing_sc: List[str] = None

    radius:float = 1.2

    def sc_names(self):
        return list(self.sc_init.keys())

    pass

def tail_config():
    """
    Configuration for a magnetotail fluxrope that passes over the SC constellation between 
    1340-1372s in the Vlasiator FHA simulation run.

    The constellation is (near) aligned with the flow direction though the flux rope axis is 
    (near) perpendicular direction of travel
    """
    return Config(
        vg_v_file="./temporal_reconstruction/data/plas_obs_vir_vg_v_full_tail_right_Z=0.5_GOOD.csv",
        b_field_file= "./temporal_reconstruction/data/plas_obs_vg_b_timeseries_tail_right_z=0.5.csv",
        output_dir="./",
        t_ref = 1372,
        start_time=1340,
        end_time=1372,
        sc_init = {
            "sc1": np.array([-27.0, 3.0, 0.5]) * R_E,
            "sc2": np.array([-26.0, 3.0, 1.5]) * R_E,
            "sc3": np.array([-26.0, 3.86602540, 0.0]) * R_E,
            "sc4": np.array([-26.0, 2.13397460, 0.0]) * R_E,
            "sc5": np.array([-26.85714286, 3.0, 0.64285714]) * R_E,
            "sc6": np.array([-26.85714286, 3.12371791, 0.42857143]) * R_E,
            "sc7": np.array([-26.85714286, 2.87628209, 0.42857143]) * R_E,
        },

    )

def tail_flipped_config():
    return Config(
        vg_v_file="/home/leeviloi/Fluxrope-Reconstruction/temporal_reconstruction/data/plas_obs_vir_vg_v_full_tail_z=0.5_1330-1380_flipped_GOOD.csv",
        b_field_file="/home/leeviloi/Fluxrope-Reconstruction/temporal_reconstruction/data/plas_obs_vg_b_timeseries_tail_z=0.5_1330-1380s_flipped.csv",
        t_ref=1360,
        start_time=1330,
        end_time=1380,
        sc_init = {
            "sc1": np.array([-27.0, 3.0, 0.5]) * R_E,
            "sc2": np.array([-26.0, 3.0, -0.5]) * R_E,
            "sc3": np.array([-26.0, 2.1339746, 1.0]) * R_E,
            "sc4": np.array([-26.0, 3.8660254, 1.0]) * R_E,
            "sc5": np.array([-26.85714286, 3.0, 0.35714286]) * R_E,
            "sc6": np.array([-26.85714286, 2.87628209, 0.57142857]) * R_E,
            "sc7": np.array([-26.85714286, 3.12371791, 0.57142857]) * R_E,
        },

    )

def tail_flipped_centered_config():
    return Config(
        vg_v_file="/home/leeviloi/Fluxrope-Reconstruction/temporal_reconstruction/data/plas_obs_vir_vg_v_full_tail_z=0.8_1330-1380_flipped_GOOD.csv",
        b_field_file="/home/leeviloi/Fluxrope-Reconstruction/temporal_reconstruction/data/plas_obs_vg_b_timeseries_tail_z=0.8_1330-1380s_flipped.csv",
        t_ref=1360,
        start_time=1330,
        end_time=1380,
        sc_init = {
            "sc1": np.array([-27.0, 3.0, 0.8]) * R_E,
            "sc2": np.array([-26.0, 3.0, -0.2]) * R_E,
            "sc3": np.array([-26.0, 2.1339746, 1.3]) * R_E,
            "sc4": np.array([-26.0, 3.8660254, 1.3]) * R_E,
            "sc5": np.array([-26.85714286, 3.0, 0.65714286]) * R_E,
            "sc6": np.array([-26.85714286, 2.87628209, 0.87142857]) * R_E,
            "sc7": np.array([-26.85714286, 3.12371791, 0.87142857]) * R_E,
        },

    )

def magnetopause_config():
    """
    Configuration for dayside magnetopause fluxrope that passes over the SC constellation between
    1400-1500s in the Vlasiator FHA simulation run. 

    The constellation and flux rope axis are a both (near) parallel to the flow direction.
    """ 
    return Config(
        vg_v_file="./temporal_reconstruction/data/plas_obs_vir_vg_v_full_magnetopause_z=-1_1400-1500_GOOD.csv",
        b_field_file="./temporal_reconstruction/data/plas_obs_vg_b_timeseries_magnetopause_z=-1_1400-1500s.csv",
        t_ref= 1452,
        start_time = 1400,
        end_time = 1500,
        sc_init = {
            "sc1": np.array([6.0, -11.0, -1.0]) * R_E,
            "sc2": np.array([6.52532199, -10.14909648,  0.0]) * R_E,
            "sc3": np.array([5.78841792,  -9.69415429, -1.5]) * R_E,
            "sc4": np.array([7.26222606, -10.60403866, -1.5]) * R_E,
            "sc5": np.array([6.07504600, -10.87844235, -0.85714286]) * R_E,
            "sc6": np.array([5.96977399, -10.81345061, -1.07142857]) * R_E,
            "sc7": np.array([6.18031801, -10.94343409, -1.07142857]) * R_E,
        },
    )